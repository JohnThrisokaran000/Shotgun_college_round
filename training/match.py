"""Train the PatchWork matcher from scratch.

The matcher is a small BERT-style cross-encoder. It reads an anchor's description and a candidate element's
description (written by `pw train match-data`, with the text built by pw_core::describe), and scores whether the
candidate is the element the anchor was made from. Nothing is pretrained: the tokenizer and the weights start
from random and learn only from the synthetic pairs.

Output folder, read by pw-model's matcher: config.json, model.safetensors, tokenizer.json, meta.json.

    python -m training.match --data data/match --out models/matcher-v2 --device cuda

Performance: every pair is tokenised once, before training starts, and each batch is padded to its own longest
pair. On a GPU, mixed precision (bf16 autocast) is on by default. The GPU path was measured against the CPU path
on the same data (docs/DESIGN.md, section 6.10).
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from torch import nn
from transformers import BertConfig, BertModel

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenizer(texts, vocab_size, max_len):
    """WordPiece over the descriptions, with a pair template: [CLS] anchor [SEP] candidate [SEP]."""
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIALS))
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tok.enable_truncation(max_length=max_len)
    return tok


class Matcher(nn.Module):
    """BERT encoder, then one linear unit on the [CLS] vector. The score is a logit: sigmoid gives a probability."""

    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, ids, types, mask):
        cls = self.bert(input_ids=ids, token_type_ids=types, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


class Encoded:
    """Every pair, tokenised once. Batches are built from these arrays, so the tokenizer never runs in the loop."""

    def __init__(self, tok, rows):
        encs = tok.encode_batch([(r["a"], r["c"]) for r in rows])
        self.ids = [e.ids for e in encs]
        self.types = [e.type_ids for e in encs]
        self.labels = [float(r["y"]) for r in rows]
        self.groups = [r["g"] for r in rows]

    def __len__(self):
        return len(self.ids)

    def batch(self, indices, device):
        """Pad the chosen pairs to the longest of them. Returns ids, token types, mask, and labels on `device`."""
        width = max(len(self.ids[i]) for i in indices)
        ids = torch.zeros(len(indices), width, dtype=torch.long)
        types = torch.zeros_like(ids)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(indices):
            n = len(self.ids[i])
            ids[row, :n] = torch.tensor(self.ids[i])
            types[row, :n] = torch.tensor(self.types[i])
            mask[row, :n] = 1
        labels = torch.tensor([self.labels[i] for i in indices])
        return ids.to(device, non_blocking=True), types.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(device)


def summarize(scores, labels, groups):
    """Top-1 accuracy on variants whose target is a candidate, and refusal on variants whose target is not.

    A refusal is a variant where no candidate scores at least 0.5. The bench decides the real thresholds; these
    numbers only show whether training is going somewhere."""
    by_group = {}
    for s, y, g in zip(scores, labels, groups):
        by_group.setdefault(g, []).append((s, y))
    present = [rows for rows in by_group.values() if any(y for _, y in rows)]
    absent = [rows for rows in by_group.values() if not any(y for _, y in rows)]
    top1 = sum(1 for rows in present if max(rows, key=lambda r: r[0])[1] == 1) / max(1, len(present))
    refused = sum(1 for rows in absent if all(s < 0.5 for s, _ in rows)) / max(1, len(absent))
    return {"variants_with_target": len(present), "top1_present": round(top1, 4),
            "variants_without_target": len(absent), "refused_absent": round(refused, 4)}


def pick_device(name):
    """`auto` uses the first CUDA GPU when torch can see one, and the CPU otherwise."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, data, batch, device, amp):
    model.eval()
    scores = []
    order = list(range(len(data)))
    for start in range(0, len(order), batch):
        ids, types, mask, _ = data.batch(order[start:start + batch], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(ids, types, mask)
        scores += torch.sigmoid(logits.float()).cpu().tolist()
    model.train()
    return summarize(scores, data.labels, data.groups)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device {device} amp {amp}", flush=True)

    data_dir = Path(args.data)
    train_rows = read_rows(data_dir / "train.jsonl")
    val_rows = read_rows(data_dir / "val.jsonl")
    tok = build_tokenizer([r["a"] for r in train_rows] + [r["c"] for r in train_rows], args.vocab, args.max_len)
    train_data = Encoded(tok, train_rows)
    val_data = Encoded(tok, val_rows)

    cfg = BertConfig(
        vocab_size=tok.get_vocab_size(), hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, intermediate_size=4 * args.hidden, max_position_embeddings=args.max_len,
        type_vocab_size=2, hidden_act="gelu",
        hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout,
    )
    model = Matcher(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    per_epoch = (len(train_data) + args.batch - 1) // args.batch
    steps = args.epochs * per_epoch
    warmup = max(1, int(steps * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / (steps - warmup + 1))))
    pos_weight = torch.tensor(args.pos_weight, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch):
            ids, types, mask, target = train_data.batch(order[start:start + args.batch], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(ids, types, mask)
            loss = loss_fn(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(target)
        metrics = evaluate(model, val_data, args.batch * 2, device, amp)
        history.append({"epoch": epoch + 1, "train_loss": round(total / len(train_data), 4), **metrics})
        print(json.dumps(history[-1]), f"({time.time() - started:.0f}s)", flush=True)
    elapsed = time.time() - started

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k[len("bert."):]: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k.startswith("bert.")}
    state["head.weight"] = model.head.weight.detach().cpu().contiguous()
    state["head.bias"] = model.head.bias.detach().cpu().contiguous()
    save_file(state, str(out / "model.safetensors"))
    cfg.to_json_file(str(out / "config.json"))
    tok.save(str(out / "tokenizer.json"))
    meta = {
        "device": str(device), "amp": amp, "seconds": round(elapsed, 1),
        "train_rows": len(train_data), "val_rows": len(val_data), "epochs": args.epochs, "vocab": tok.get_vocab_size(),
        "hidden": args.hidden, "layers": args.layers, "heads": args.heads, "lr": args.lr, "batch": args.batch,
        "dropout": args.dropout, "pos_weight": args.pos_weight, "warmup": args.warmup, "max_len": args.max_len,
        "seed": args.seed, "parameters": sum(p.numel() for p in model.parameters()), "history": history,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({meta['parameters']} parameters, {elapsed:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/match")
    p.add_argument("--out", default="models/matcher-v2")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--vocab", type=int, default=6000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos-weight", type=float, default=1.0, help="weight on the positive (target) rows")
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--warmup", type=float, default=0.05, help="share of steps spent warming the learning rate up")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast on a GPU")
    train(p.parse_args())


if __name__ == "__main__":
    main()
