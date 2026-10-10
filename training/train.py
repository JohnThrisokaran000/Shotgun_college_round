"""Train the PatchWork model from scratch (single process), then evaluate and export it.

Architecture: a LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, tied embeddings) built with
transformers, so the exported safetensors load in Rust with candle's llama implementation.

Example (CPU smoke test):
    python -m training.train --data data/patchwork --out runs/smoke --hidden 128 --layers 2 \
        --steps 200 --batch 16 --max-len 512
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from training import pwdata


def build_model(vocab: int, hidden: int, layers: int, heads: int, ctx: int, tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=int(hidden * 8 / 3) // 32 * 32 or 32,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=ctx,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        pad_token_id=pwdata.ids(tok, pwdata.PAD),
        bos_token_id=pwdata.ids(tok, pwdata.BOS),
        eos_token_id=pwdata.ids(tok, pwdata.EOS),
    )
    return LlamaForCausalLM(cfg)


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model, optimizer, items, steps: int, batch: int, pad_id: int, base_lr: float,
               warmup: int, device, rng: random.Random, start_step: int = 0, total_steps: int | None = None,
               log_every: int = 50, use_amp: bool = False, scaler=None) -> float:
    """Run `steps` optimizer steps over `items`, cycling through shuffled epochs."""
    model.train()
    total_steps = total_steps or steps
    losses, step = [], 0
    epoch_iter = None
    while step < steps:
        if epoch_iter is None:
            epoch_iter = pwdata.batches(items, batch, pad_id, rng)
        try:
            input_ids, labels, mask = next(epoch_iter)
        except StopIteration:
            epoch_iter = None
            continue
        ids_t = torch.tensor(input_ids, device=device)
        lab_t = torch.tensor(labels, device=device)
        mask_t = torch.tensor(mask, device=device)
        lr = lr_at(start_step + step, total_steps, base_lr, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=ids_t, attention_mask=mask_t)
            logits = out.logits[:, :-1, :]
            targets = lab_t[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=pwdata.IGNORE)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
        step += 1
        if log_every and step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {start_step + step}/{total_steps} lr {lr:.2e} loss {avg:.4f}", flush=True)
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def heldout_loss(model, items, pad_id: int, device, batch: int = 32) -> float:
    model.eval()
    total, count = 0.0, 0
    rng = random.Random(0)
    for input_ids, labels, mask in pwdata.batches(items, batch, pad_id, rng, shuffle=False):
        ids_t = torch.tensor(input_ids, device=device)
        lab_t = torch.tensor(labels, device=device)
        mask_t = torch.tensor(mask, device=device)
        logits = model(input_ids=ids_t, attention_mask=mask_t).logits[:, :-1, :]
        targets = lab_t[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=pwdata.IGNORE, reduction="sum")
        total += loss.item()
        count += int((targets != pwdata.IGNORE).sum().item())
    return total / max(1, count)


@torch.no_grad()
def generate_answer(model, tok, prompt: str, device, max_new: int = 24) -> str:
    model.eval()
    bos, eos = pwdata.ids(tok, pwdata.BOS), pwdata.ids(tok, pwdata.EOS)
    seq = [bos] + tok.encode(prompt).ids
    ids_t = torch.tensor([seq], device=device)
    out = model.generate(input_ids=ids_t, max_new_tokens=max_new, do_sample=False, eos_token_id=eos,
                         pad_token_id=pwdata.ids(tok, pwdata.PAD))
    new_tokens = out[0, len(seq):].tolist()
    if eos in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(eos)]
    return tok.decode(new_tokens).strip()


def normalise_answer(task: str, text: str):
    """Compare answers by meaning: JSON for both tasks, with a fallback to the raw string."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return ("raw", text.strip())
    return ("json", json.dumps(value, sort_keys=True))


def evaluate(model, tok, rows: list[dict], device, limit: int) -> dict:
    relocate = [r for r in rows if r["task"] == "relocate"][:limit]
    author = [r for r in rows if r["task"] == "author"][:limit]
    report = {"relocate_n": len(relocate), "author_n": len(author)}
    for name, group in (("relocate", relocate), ("author", author)):
        if not group:
            continue
        hits = 0
        null_hits = null_n = match_hits = match_n = 0
        for row in group:
            pred = normalise_answer(name, generate_answer(model, tok, row["prompt"], device))
            gold = normalise_answer(name, row["completion"])
            ok = pred == gold
            hits += ok
            if name == "relocate":
                is_null = gold == ("json", json.dumps({"match": None}, sort_keys=True))
                if is_null:
                    null_n += 1
                    null_hits += ok
                else:
                    match_n += 1
                    match_hits += ok
        report[f"{name}_exact"] = round(hits / len(group), 4)
        if name == "relocate":
            report["relocate_match_exact"] = round(match_hits / max(1, match_n), 4)
            report["relocate_null_exact"] = round(null_hits / max(1, null_n), 4)
            report["relocate_null_share"] = round(null_n / len(group), 4)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="directory with train.jsonl and heldout.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpus", type=Path, nargs="*", default=[], help="extra text files for the tokenizer and LM")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    train_rows = list(pwdata.read_jsonl(args.data / "train.jsonl"))
    held_rows = list(pwdata.read_jsonl(args.data / "heldout.jsonl"))

    tok_path = args.out / "tokenizer.json"
    if not tok_path.exists():
        texts = (t for r in train_rows for t in (r["prompt"], r["completion"]))
        extra = (line.rstrip("\n") for path in args.corpus for line in path.open(encoding="utf-8", errors="ignore"))
        print("training tokenizer", flush=True)
        pwdata.train_tokenizer(list(texts) + list(extra), tok_path, args.vocab)
    tok = pwdata.load_tokenizer(tok_path)

    train_items, dropped = pwdata.encode_rows(tok, train_rows, args.max_len)
    held_items, held_dropped = pwdata.encode_rows(tok, held_rows, args.max_len)
    print(f"train {len(train_items)} rows (dropped {dropped} over {args.max_len} tokens); heldout {len(held_items)}", flush=True)

    model = build_model(tok.get_vocab_size(), args.hidden, args.layers, args.heads, args.ctx, tok).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model: {params / 1e6:.2f}M parameters on {device}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    t0 = time.time()
    train_loss = train_loop(model, optimizer, train_items, args.steps, args.batch, pwdata.ids(tok, pwdata.PAD),
                            args.lr, args.warmup, device, rng, total_steps=args.steps, use_amp=use_amp, scaler=scaler)
    elapsed = time.time() - t0
    held = heldout_loss(model, held_items, pwdata.ids(tok, pwdata.PAD), device)
    metrics = {"params": params, "steps": args.steps, "train_loss_avg": round(train_loss, 4),
               "heldout_loss": round(held, 4), "train_seconds": round(elapsed, 1), "device": str(device),
               "dropped_too_long": dropped}
    metrics.update(evaluate(model, tok, held_rows, device, args.eval_n))

    model.save_pretrained(args.out / "model")
    tok.save(str(args.out / "model" / "tokenizer.json"))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
