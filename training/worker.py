"""DiLoCo worker: trains locally for H steps, sends a pseudo-gradient to the coordinator, resyncs.

Runs on each Kaggle notebook (or on any machine). It needs only the coordinator URL. The model
config, tokenizer and training rows are downloaded from the coordinator's /v1/files route.

    python -m training.worker --coordinator http://HOST:8765 --worker-id kaggle-01 \
        --rank 0 --world 14 --inner-steps 500 --rounds 40 --max-minutes 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from training import pwdata
from training import train as tr

FILES = ("config.json", "tokenizer.json", "train.jsonl")


def http(method: str, url: str, token: str | None = None, data: bytes | None = None, timeout: float = 300) -> bytes:
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def status(coord: str, token: str | None) -> dict:
    return json.loads(http("GET", f"{coord}/v1/status", token, timeout=60))


def fetch_global(coord: str, token: str | None) -> dict[str, torch.Tensor]:
    return st_load(http("GET", f"{coord}/v1/global", token))


def post_delta(coord: str, token: str | None, worker: str, round_no: int, delta: dict[str, torch.Tensor]) -> int:
    body = st_save({k: v.detach().half().cpu().contiguous() for k, v in delta.items()})
    try:
        http("POST", f"{coord}/v1/delta?worker={worker}&round={round_no}", token, data=body)
        return 202
    except urllib.error.HTTPError as e:
        # 409 means the round already moved on, which is normal after a timeout. Anything else is a
        # real failure (for example 413, a body over the coordinator's limit) and must stop the worker.
        if e.code == 409:
            return e.code
        detail = e.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"coordinator refused the delta: HTTP {e.code} {detail}") from e


def download_files(coord: str, token: str | None, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if not target.exists():
            target.write_bytes(http("GET", f"{coord}/v1/files/{name}", token, timeout=1800))


def wait_for_round(coord: str, token: str | None, after: int, poll_s: float, give_up_s: float) -> int:
    waited = 0.0
    while True:
        s = status(coord, token)
        if s["round"] > after:
            return s["round"]
        if waited >= give_up_s:
            raise TimeoutError(f"round {after + 1} did not open within {give_up_s:.0f}s")
        time.sleep(poll_s)
        waited += poll_s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coordinator", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("worker-data"))
    p.add_argument("--inner-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=0, help="planned total inner steps, for the LR schedule")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--max-minutes", type=float, default=600.0, help="stop cleanly before a notebook time limit")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--round-timeout-minutes", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda when present)")
    args = p.parse_args(argv)

    token = os.environ.get("PW_TOKEN") or None
    coord = args.coordinator.rstrip("/")
    started = time.time()
    torch.manual_seed(args.seed + args.rank)
    rng = random.Random(args.seed + args.rank)

    download_files(coord, token, args.work_dir)
    from transformers import LlamaConfig, LlamaForCausalLM

    device = tr.pick_device(args.device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    config = LlamaConfig.from_json_file(str(args.work_dir / "config.json"))
    model = LlamaForCausalLM(config).to(device)
    tok = pwdata.load_tokenizer(args.work_dir / "tokenizer.json")
    rows = list(pwdata.read_jsonl(args.work_dir / "train.jsonl"))
    items, dropped = pwdata.encode_rows(tok, rows, args.max_len)
    shard = items[args.rank :: args.world]
    print(f"worker {args.worker_id}: shard {len(shard)} of {len(items)} rows ({dropped} too long)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    pad = pwdata.ids(tok, pwdata.PAD)
    total = args.total_steps or args.inner_steps * args.rounds
    inner_done = 0
    round_no = status(coord, token)["round"]

    for _ in range(args.rounds):
        if (time.time() - started) / 60.0 > args.max_minutes:
            print("time budget reached; stopping cleanly", flush=True)
            break
        global_state = fetch_global(coord, token)
        model.load_state_dict(global_state, strict=False)
        base = {k: v.detach().clone() for k, v in model.named_parameters()}
        loss = tr.train_loop(model, optimizer, shard, args.inner_steps, args.batch, pad, args.lr, args.warmup,
                             device, rng, start_step=inner_done, total_steps=total, log_every=0,
                             use_amp=use_amp, scaler=scaler)
        inner_done += args.inner_steps
        now = {k: v.detach() for k, v in model.named_parameters()}
        delta = {k: base[k] - now[k] for k in base}
        code = post_delta(coord, token, args.worker_id, round_no, delta)
        print(f"round {round_no}: inner loss {loss:.4f}, delta sent ({code})", flush=True)
        round_no = wait_for_round(coord, token, round_no, args.poll_seconds, args.round_timeout_minutes * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
