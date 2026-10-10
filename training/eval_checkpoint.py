"""Evaluate a checkpoint, and optionally export it as a Hugging Face model folder.

    python -m training.eval_checkpoint --serve runs/serve --weights runs/state/global.safetensors \
        --patchwork data/pw-v2/heldout.jsonl --general data/corpus/general/heldout.jsonl \
        --limit 300 --export runs/final-model

Reports:
    general_heldout_loss    teacher-forced loss on whole packages never seen in training (nats per token)
    patchwork_heldout_loss  the same on held-out PatchWork sites, completion tokens only
    relocate_*              greedy answers on held-out sites: exact match, split into "pick an element"
                            and "refuse" (null) cases. wrong_pick_rate counts the harmful errors: a pick
                            when the answer was a refusal, or the wrong element.
    author_*                exact ops match, and the share of answers that are valid DSL JSON
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from training import arch, pwdata

NULL = json.dumps({"match": None}, sort_keys=True)


def load_model(serve: Path, weights: Path, device: torch.device):
    from transformers import Gemma3TextConfig

    cfg = Gemma3TextConfig.from_json_file(str(serve / "config.json"))
    model = arch.build_model(cfg)
    state = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied to the embeddings
    if missing or unexpected:
        raise SystemExit(f"weights do not match the architecture: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.tie_weights()
    return model.to(device).eval(), cfg


@torch.no_grad()
def teacher_forced_loss(model, tok, rows: list[dict], device, max_len: int = 512, batch: int = 16) -> float:
    bos, eos, pad = (pwdata.ids(tok, n) for n in (pwdata.BOS, pwdata.EOS, pwdata.PAD))
    items = []
    for row in rows:
        prompt, completion = tok.encode(row["prompt"]).ids, tok.encode(row["completion"]).ids + [eos]
        seq = [bos] + prompt + completion
        if len(seq) <= max_len:
            items.append((seq, 1 + len(prompt)))
    total, count = 0.0, 0
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        width = max(len(s) for s, _ in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, (seq, comp) in enumerate(chunk):
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
            labels[r, comp:len(seq)] = torch.tensor(seq[comp:])
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, :-1].float()
        targets = labels[:, 1:].to(device)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
                                 reduction="sum").item()
        count += int((targets != -100).sum().item())
    return total / max(1, count)


@torch.no_grad()
def greedy(model, tok, prompt: str, device, max_new: int) -> str:
    bos, eos, pad = (pwdata.ids(tok, n) for n in (pwdata.BOS, pwdata.EOS, pwdata.PAD))
    seq = [bos] + tok.encode(prompt).ids
    out = model.generate(input_ids=torch.tensor([seq], device=device), max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos, pad_token_id=pad)
    new = out[0, len(seq):].tolist()
    if eos in new:
        new = new[:new.index(eos)]
    return tok.decode(new).strip()


def _canonical(text: str) -> str | None:
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except json.JSONDecodeError:
        return None


def score_patchwork(model, tok, rows: list[dict], device, limit: int) -> dict:
    relocate = [r for r in rows if r["task"] == "relocate"][:limit]
    author = [r for r in rows if r["task"] == "author"][:limit]
    report: dict = {"relocate_n": len(relocate), "author_n": len(author)}
    pick_ok = pick_n = null_ok = null_n = wrong_pick = 0
    for row in relocate:
        gold, pred = _canonical(row["completion"]), _canonical(greedy(model, tok, row["prompt"], device, 16))
        if gold == NULL:
            null_n += 1
            null_ok += pred == NULL
            wrong_pick += pred is not None and pred != NULL
        else:
            pick_n += 1
            pick_ok += pred == gold
            wrong_pick += pred is not None and pred not in (gold, NULL)
    if relocate:
        report.update({
            "relocate_exact": round((pick_ok + null_ok) / len(relocate), 4),
            "relocate_pick_exact": round(pick_ok / max(1, pick_n), 4),
            "relocate_refusal_exact": round(null_ok / max(1, null_n), 4),
            "relocate_wrong_pick_rate": round(wrong_pick / len(relocate), 4),
            "relocate_refusal_share": round(null_n / len(relocate), 4),
        })
    exact = valid = 0
    for row in author:
        pred = greedy(model, tok, row["prompt"], device, 128)
        canonical = _canonical(pred)
        exact += canonical is not None and canonical == _canonical(row["completion"])
        try:
            valid += isinstance(json.loads(pred).get("ops"), list)
        except (json.JSONDecodeError, AttributeError):
            pass
    if author:
        report["author_exact"] = round(exact / len(author), 4)
        report["author_valid_json"] = round(valid / len(author), 4)
    return report


def export(model, serve: Path, out: Path, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in ("tokenizer.json", "run.json"):
        if (serve / name).exists():
            shutil.copy(serve / name, out / (name if name != "run.json" else "init_run.json"))
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serve", type=Path, required=True, help="folder with config.json and tokenizer.json")
    p.add_argument("--weights", type=Path, required=True, help="f32 safetensors with the model's parameter names")
    p.add_argument("--patchwork", type=Path, default=None, help="held-out PatchWork rows (jsonl)")
    p.add_argument("--general", type=Path, default=None, help="held-out general rows (jsonl)")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--loss-rows", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--export", type=Path, default=None, help="write a Hugging Face model folder here")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    tok = pwdata.load_tokenizer(args.serve / "tokenizer.json")
    model, _ = load_model(args.serve, args.weights, device)
    report: dict = {"weights": str(args.weights), "params": sum(p.numel() for p in model.parameters())}
    if args.general:
        rows = list(pwdata.read_jsonl(args.general))[:args.loss_rows]
        loss = teacher_forced_loss(model, tok, rows, device)
        report.update(general_heldout_loss=round(loss, 4), general_heldout_ppl=round(math.exp(min(loss, 50)), 2))
    if args.patchwork:
        rows = list(pwdata.read_jsonl(args.patchwork))
        report["patchwork_heldout_loss"] = round(teacher_forced_loss(model, tok, rows[:args.loss_rows], device), 4)
        report.update(score_patchwork(model, tok, rows, device, args.limit))
    print(json.dumps(report, indent=2), flush=True)
    if args.export:
        export(model, args.serve, args.export, report)
        print(f"exported to {args.export}")


if __name__ == "__main__":
    main()
