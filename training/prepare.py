"""Prepare what the coordinator serves: the architecture, the tokenizer, and the starting weights.

    python -m training.prepare --tokenizer runs/tokenizer.json --out runs/serve

Writes:
    config.json          Gemma-3-style architecture (see training/arch.py)
    tokenizer.json       the tokenizer, copied from --tokenizer
    init.safetensors     the starting weights, f32, from training/initrng.py (deterministic)
    run.json             what was written, with the canonical hash that every group checks

The training data is not here. Each group gets its own dataset, built by training/shards.py,
so the coordinator never sends bulk data. The starting weights come from a seeded generator that
gives the same bits on any machine, and every group checks its own copy against the hash in run.json.

Architecture overrides come from the flags below, or from a reference config.json that holds
architecture keys only (weights and tokenizer in it are ignored). The EmbeddingGemma config is
gated, so this build uses its architecture as a reference only, with the assumed dimensions in
arch.TEMPLATE unless you provide one.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from training import arch, initrng, pwdata


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None, help="optional architecture-only config.json")
    p.add_argument("--seed", type=int, default=0, help="seed of the starting weights; groups must use the same one")
    p.add_argument("--hidden", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--ffn", type=int)
    p.add_argument("--sliding-window", type=int)
    p.add_argument("--ctx", type=int)
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

    with torch.device("meta"):
        model = arch.build_model(cfg)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    state = initrng.init_state(shapes, args.seed)
    save_file({name: torch.from_numpy(values) for name, values in state.items()}, str(out / "init.safetensors"))
    params = sum(int(v.size) for v in state.values())

    summary = {
        "vocab": vocab,
        "params": params,
        "tensors": len(state),
        "init_seed": args.seed,
        "global_sha256": initrng.canonical_sha256(state),
        "architecture": {k: getattr(cfg, k) for k in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "sliding_window", "max_position_embeddings")},
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
