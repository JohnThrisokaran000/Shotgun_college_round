"""Evaluate a checkpoint on the held-out split: teacher-forced loss and greedy exact match.

    python -m training.eval_checkpoint --config runs/dist/config.json --weights runs/dist/state/global.safetensors \
        --tokenizer runs/dist/tokenizer.json --data data/patchwork/heldout.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from training import pwdata
from training import train as tr


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="safetensors with the parameter names of the model")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    from transformers import LlamaConfig, LlamaForCausalLM

    device = tr.pick_device(args.device)
    tok = pwdata.load_tokenizer(args.tokenizer)
    model = LlamaForCausalLM(LlamaConfig.from_json_file(str(args.config)))
    missing, unexpected = model.load_state_dict(load_file(str(args.weights)), strict=False)
    model.to(device).eval()
    rows = list(pwdata.read_jsonl(args.data))
    items, _ = pwdata.encode_rows(tok, rows, args.max_len)
    pad = pwdata.ids(tok, pwdata.PAD)
    report = {
        "weights": str(args.weights),
        "missing_keys": [k for k in missing if "lm_head" not in k],
        "unexpected_keys": unexpected,
        "heldout_loss": round(tr.heldout_loss(model, items, pad, device), 4),
    }
    report.update(tr.evaluate(model, tok, rows, device, args.limit))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
