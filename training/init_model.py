"""Create the shared starting point for a distributed run: config, tokenizer, initial weights.

    python -m training.init_model --tokenizer runs/x/tokenizer.json --out runs/dist \
        --train-rows data/patchwork/train.jsonl --hidden 512 --layers 8 --heads 8

Writes config.json, tokenizer.json, init.safetensors (the coordinator's starting weights) and
serve/ (the files workers download: config, tokenizer, training rows).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import save_file

from training import pwdata, train as tr


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--train-rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    tok = pwdata.load_tokenizer(args.tokenizer)
    model = tr.build_model(tok.get_vocab_size(), args.hidden, args.layers, args.heads, args.ctx, tok)
    args.out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(args.out)
    shutil.copy(args.tokenizer, args.out / "tokenizer.json")
    state = {k: v.detach().contiguous() for k, v in model.named_parameters()}
    save_file(state, str(args.out / "init.safetensors"))
    serve = args.out / "serve"
    serve.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.out / "config.json", serve / "config.json")
    shutil.copy(args.tokenizer, serve / "tokenizer.json")
    shutil.copy(args.train_rows, serve / "train.jsonl")
    total = sum(v.numel() for v in state.values())
    print(f"initial model: {total / 1e6:.2f}M parameters in {len(state)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
