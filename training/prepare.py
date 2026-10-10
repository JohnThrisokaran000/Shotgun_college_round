"""Prepare everything a run needs, in one folder that the coordinator serves.

    python -m training.prepare --data-dir data/patchwork --tokenizer runs/x/tokenizer.json --out runs/serve

Writes flat file names, because the coordinator serves a single folder with no subfolders:
    config.json          Gemma-3-style architecture (see training/arch.py)
    tokenizer.json       the PatchWork tokenizer
    init.safetensors     the starting weights (f32). The coordinator reads this one.
    train_*, heldout_*   pre-tokenized, memory-mapped data (see training/pack.py)
    run.json             what was written, for the record

Architecture overrides come from the flags below, or from a reference config.json that holds
architecture keys only (weights and tokenizer in it are ignored). The EmbeddingGemma config is
gated, so this build uses its architecture as a reference only, with the assumed dimensions in
arch.TEMPLATE unless you provide one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from training import arch, pack, pwdata


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="folder with train.jsonl and heldout.jsonl")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None, help="optional architecture-only config.json")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--limit-train", type=int, default=0, help="use only the first N training rows (0 = all)")
    p.add_argument("--limit-heldout", type=int, default=0)
    p.add_argument("--hidden", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--ffn", type=int)
    p.add_argument("--sliding-window", type=int)
    p.add_argument("--ctx", type=int)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tokenizer = pwdata.load_tokenizer(args.tokenizer)
    vocab = tokenizer.get_vocab_size()
    bos, eos = pwdata.ids(tokenizer, pwdata.BOS), pwdata.ids(tokenizer, pwdata.EOS)
    pad = pwdata.ids(tokenizer, pwdata.PAD)
    flags = {
        "hidden_size": args.hidden,
        "num_hidden_layers": args.layers,
        "num_attention_heads": args.heads,
        "num_key_value_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "intermediate_size": args.ffn,
        "sliding_window": args.sliding_window,
        "max_position_embeddings": args.ctx,
    }
    overrides = {k: v for k, v in flags.items() if v is not None}
    cfg = arch.make_config(vocab, pad_id=pad, bos_id=bos, eos_id=eos, reference=args.reference, **overrides)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(out)
    shutil.copy(args.tokenizer, out / "tokenizer.json")

    torch.manual_seed(args.seed)
    model = arch.build_model(cfg)
    state = {name: p.detach().contiguous() for name, p in model.named_parameters()}
    save_file(state, str(out / "init.safetensors"))
    params = sum(t.numel() for t in state.values())

    summary = {"vocab": vocab, "params": params, "architecture": {k: getattr(cfg, k) for k in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "sliding_window", "max_position_embeddings")}, "splits": {}}
    limits = {"train": args.limit_train, "heldout": args.limit_heldout}
    for split in ("train", "heldout"):
        rows = pack.iter_jsonl(args.data_dir / f"{split}.jsonl")
        if limits[split]:
            rows = itertools.islice(rows, limits[split])
        tmp = out / f"_pack_{split}"
        meta = pack.pack(rows, tokenizer, tmp, max_len=args.max_len, bos=bos, eos=eos, pad=pad)
        for name in ("meta.json", "offsets.i64", "comp.i32", "tokens.u16"):
            shutil.move(str(tmp / name), str(out / f"{split}_{name}"))
        shutil.rmtree(tmp)
        summary["splits"][split] = {k: meta[k] for k in ("examples", "tokens", "dropped_too_long")}
    (out / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
