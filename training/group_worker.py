"""Group worker: one DiLoCo replica per Kaggle group.

A group is one Kaggle notebook with two T4s. Inside a group the GPUs train one replica with
DistributedDataParallel. Across groups nothing is shared directly. Each group sends an outer
delta to the coordinator and receives the merged global weights back.

Communication overlaps with compute:
  - Rank 0 uploads a delta in a background thread. The group keeps training meanwhile.
  - Rank 0 downloads a newer global in a background thread, whenever the coordinator has one.
  - At a sync point the group merges it:  local <- global + (local - snapshot)
    so the progress made since the snapshot is kept, and nothing is counted twice.
  - Every decision is broadcast from rank 0, so both GPUs of a group apply the same change.

Deltas may be one global version old (the coordinator's MAX_STALENESS is 1).

Every line meant for monitoring starts with "PW " followed by one JSON object. Nothing else
is needed to follow a run from the notebook output.

    python -m training.group_worker --coordinator http://HOST:8765 --group k01 --replica 0 --replicas 14
    torchrun --standalone --nproc_per_node 2 -m training.group_worker ...   (two GPUs in one group)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save
from torch.nn.parallel import DistributedDataParallel as DDP

from training import arch, initrng, pack
from training.train import lr_at

IGNORE = pack.IGNORE
SERVED = ["config.json", "tokenizer.json"] + [
    f"{split}_{name}" for split in ("train", "heldout") for name in ("meta.json", "offsets.i64", "comp.i32", "tokens.u16")
]

# Decisions broadcast from rank 0 at each sync point.
NOTHING, SEND, MERGE, FINISH = 0, 1, 2, 3
SCALE_SUFFIX = ".__scale"  # per-row scales in an int8 global (see coordinator.to_int8_safetensors)

GROUP = "?"
EVENTS = None  # set on rank 0 when events are posted to the coordinator


def log(ev: str, **fields) -> None:
    fields["ev"] = ev
    fields["t"] = round(time.time(), 1)
    fields.setdefault("group", GROUP)
    print("PW " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)
    if EVENTS is not None:
        EVENTS.put(fields)


class Events:
    """Posts PW events to the coordinator from a background thread, so the monitor sees them live.
    Monitoring never slows training: when the queue is full or the coordinator is unreachable, the
    event is dropped and counted. The notebook log still has every line."""

    def __init__(self, coord: str, token: str, batch: int = 40, flush_s: float = 5.0):
        self.coord, self.token, self.batch, self.flush_s = coord, token, batch, flush_s
        self.queue: queue.Queue = queue.Queue(maxsize=5000)
        self.dropped = 0
        self.failures = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def put(self, record: dict) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + self.flush_s
            while len(batch) < self.batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._send(batch)

    def _send(self, batch: list[dict]) -> None:
        req = urllib.request.Request(f"{self.coord}/v1/events", data=json.dumps(batch, default=str).encode(),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                response.read()
        except Exception as err:  # noqa: BLE001 - monitoring must not stop training
            self.dropped += len(batch)
            self.failures += 1
            if self.failures == 1 or self.failures % 20 == 0:
                print("PW " + json.dumps({"ev": "event_post_failed", "failures": self.failures,
                                          "dropped": self.dropped, "error": repr(err)[:160]}), flush=True)


# ---------------------------------------------------------------- HTTP


def _request(method: str, url: str, token: str | None, data: bytes | None = None, timeout: float = 600):
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers


def get_json(coord: str, token: str | None, path: str) -> dict:
    body, _ = _request("GET", f"{coord}{path}", token, timeout=60)
    return json.loads(body)


def fetch_file(coord: str, token: str | None, name: str, dest: Path) -> None:
    body, _ = _request("GET", f"{coord}/v1/files/{name}", token, timeout=1800)
    dest.write_bytes(body)


def decode_global(raw: dict, names: list[str]) -> dict:
    """int8 rows dequantise as q * scale. f16 and f32 tensors convert to float32. Keys are the parameter names."""
    out = {}
    for name in names:
        tensor, scale = raw[name], raw.get(name + SCALE_SUFFIX)
        out[name] = tensor.float() * scale.float().unsqueeze(1) if scale is not None else tensor.float()
    return out


def fetch_global(coord: str, token: str | None, dtype: str, names: list[str]) -> tuple[int, dict]:
    body, headers = _request("GET", f"{coord}/v1/global?dtype={dtype}", token, timeout=1800)
    return int(headers.get("x-pw-round", "0")), decode_global(st_load(body), names)


def load_into(model, state: dict) -> None:
    """Copy float32 values into the model's parameters, by name."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def post_delta(coord: str, token: str | None, group: str, base_version: int, seq: int, delta16: dict):
    """Returns (202, body) when accepted, or (status, message) when refused. Safe to repeat: same seq, same result."""
    payload = st_save({k: v.contiguous() for k, v in delta16.items()})
    try:
        body, _ = _request("POST", f"{coord}/v1/delta?worker={group}&round={base_version}&seq={seq}", token, data=payload)
        return 202, json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode("utf-8", "replace")


# ---------------------------------------------------------------- background network (rank 0 only)


class Net:
    """Uploads deltas and fetches globals in background threads, so the GPUs never wait on the network."""

    def __init__(self, coord: str, token: str | None, group: str, global_dtype: str, names: list[str], poll_s: float = 5.0):
        self.coord, self.token, self.group = coord, token, group
        self.global_dtype, self.names = global_dtype, names
        self.poll_s = poll_s
        self.lock = threading.Lock()
        self.uploads: queue.Queue = queue.Queue()
        self.result: dict | None = None  # outcome of the last upload
        self.want: int | None = None  # the smallest global version the group needs
        self.ready: tuple[int, dict] | None = None  # (version, state) waiting to be merged
        self.closed = False
        self.errors = 0
        threading.Thread(target=self._upload_loop, daemon=True).start()
        threading.Thread(target=self._fetch_loop, daemon=True).start()

    def submit(self, seq: int, base_version: int, delta16: dict) -> None:
        with self.lock:
            self.result = None
        self.uploads.put((seq, base_version, delta16))

    def upload_result(self) -> dict | None:
        with self.lock:
            return self.result

    def request_global(self, min_version: int) -> None:
        with self.lock:
            self.want = min_version

    def take_global(self, min_version: int) -> tuple[int, dict] | None:
        with self.lock:
            if self.ready is not None and self.ready[0] >= min_version:
                got, self.ready, self.want = self.ready, None, None
                return got
        return None

    def close(self) -> None:
        self.closed = True
        self.uploads.put(None)

    def _upload_loop(self) -> None:
        while True:
            item = self.uploads.get()
            if item is None:
                return
            seq, base_version, delta16 = item
            outcome: dict = {"accepted": None, "refused": False, "base": base_version, "seq": seq}
            for attempt in range(4):
                try:
                    code, body = post_delta(self.coord, self.token, self.group, base_version, seq, delta16)
                    if code == 202:
                        outcome["accepted"] = int(body["accepted_into"])
                        outcome["version_now"] = int(body["round"])
                    else:
                        outcome["refused"] = True
                        outcome["reason"] = f"HTTP {code}: {body}"
                    break
                except Exception as e:  # network trouble: retry (the same seq is safe), then report
                    self.errors += 1
                    outcome["reason"] = repr(e)
                    log("net_error", where="upload", group=self.group, seq=seq, attempt=attempt + 1, error=repr(e))
                    time.sleep(5 * (attempt + 1))
            else:
                outcome["refused"] = True
            with self.lock:
                self.result = outcome

    def _fetch_loop(self) -> None:
        while not self.closed:
            with self.lock:
                want = self.want
                have = self.ready[0] if self.ready is not None else None
            if want is None or (have is not None and have >= want):
                time.sleep(self.poll_s)
                continue
            try:
                status = get_json(self.coord, self.token, "/v1/status")
                if int(status["round"]) >= want:
                    version, state = fetch_global(self.coord, self.token, self.global_dtype, self.names)
                    with self.lock:
                        self.ready = (version, state)
                    continue
            except Exception as e:
                self.errors += 1
                log("net_error", where="fetch", error=repr(e))
            time.sleep(self.poll_s)


# ---------------------------------------------------------------- pure helpers (tested)


@torch.no_grad()
def delayed_merge(local: list[torch.Tensor], snapshot: list[torch.Tensor], global_: list[torch.Tensor]) -> None:
    """In place: local <- global + (local - snapshot). The progress since the snapshot is kept."""
    for l, s, g in zip(local, snapshot, global_):
        progress = l - s
        l.copy_(g)
        l.add_(progress)


def delta_fp16(base: list[torch.Tensor], local: list[torch.Tensor], names: list[str]) -> dict:
    """base - local, in fp16, on the CPU, keyed by parameter name."""
    return {n: (b - l).detach().half().cpu() for n, b, l in zip(names, base, local)}


class Batches:
    """Endless stream of length-bucketed micro-batches over one rank's shard."""

    def __init__(self, ds: pack.PackedDataset, indices: list[int], batch: int, rng: random.Random):
        if not indices:
            raise ValueError("this rank has no training examples; use fewer ranks or more data")
        self.ds, self.indices, self.batch, self.rng = ds, indices, batch, rng
        self.it = None
        self.epoch = 0

    def next(self) -> list[int]:
        while True:
            if self.it is None:
                self.it = pack.length_bucketed_batches(self.ds, self.indices, self.batch, self.rng)
            try:
                return next(self.it)
            except StopIteration:
                self.it = None
                self.epoch += 1


@torch.no_grad()
def heldout_loss(model, ds: pack.PackedDataset, pad: int, vocab: int, device, amp: bool, limit: int, batch: int) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    n = min(limit, len(ds))
    for start in range(0, n, batch):
        ids_np, labels_np, mask_np = pack.collate(ds, list(range(start, min(n, start + batch))), pad)
        ids = torch.from_numpy(ids_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        targets = labels[:, 1:]
        total += F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=IGNORE, reduction="sum").item()
        count += int((targets != IGNORE).sum().item())
    model.train(was_training)
    return total / max(1, count)


def broadcast_tensors(tensors, shapes: list[torch.Size], device, src: int = 0) -> list[torch.Tensor]:
    """Send rank `src`'s tensors to every rank in one collective. Returns the received copies."""
    if not dist.is_initialized():
        return tensors
    total = sum(math.prod(s) for s in shapes)
    if dist.get_rank() == src:
        flat = torch.cat([t.reshape(-1).float() for t in tensors])
    else:
        flat = torch.empty(total, device=device, dtype=torch.float32)
    dist.broadcast(flat, src=src)
    out, off = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(flat[off : off + n].view(s))
        off += n
    return out


def time_lr(step: int, progress: float, base_lr: float, warmup: int) -> float:
    """Warm up over the first steps, then cosine-decay to 10% of the base rate by the end of the time budget.
    Rank 0 computes it and broadcasts it at each sync point, so every GPU uses the same rate."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def checksum(params: list[torch.Tensor]) -> float:
    return float(sum(p.detach().double().abs().sum().item() for p in params))


# ---------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coordinator", required=True)
    p.add_argument("--group", required=True, help="unique id of this group (or GPU, in independent mode)")
    p.add_argument("--replica", type=int, required=True, help="index of this replica among all replicas")
    p.add_argument("--replicas", type=int, required=True, help="number of replicas (groups, or GPUs in independent mode)")
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p.add_argument("--backend", default="auto", help="nccl, gloo, or auto (used when torchrun starts several ranks)")
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
    p.add_argument("--inner-steps", type=int, default=500, help="optimizer steps between sends")
    p.add_argument("--sync-every", type=int, default=10, help="how often to check for decisions")
    p.add_argument("--max-rounds", type=int, default=0, help="stop after this many sends (0 = until time runs out)")
    p.add_argument("--max-minutes", type=float, default=540.0)
    p.add_argument("--final-wait-minutes", type=float, default=10.0)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--total-steps", type=int, default=0, help="planned optimizer steps for the cosine schedule")
    p.add_argument("--global-dtype", default="int8", choices=["f32", "f16", "int8"],
                   help="how the global is downloaded: int8 is a quarter of the f32 bytes")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="this group's dataset (config.json, train/, heldout/). Each group's data is already its own shard.")
    p.add_argument("--init-seed", type=int, default=0, help="must match training.prepare --seed (checked against the coordinator)")
    p.add_argument("--events", action=argparse.BooleanOptionalAction, default=True,
                   help="post monitoring events to the coordinator as well as printing them")
    p.add_argument("--log-every-s", type=float, default=60.0)
    p.add_argument("--eval-every", type=int, default=1, help="heldout evaluation after every N merges")
    p.add_argument("--eval-examples", type=int, default=512)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--probe-only", action="store_true", help="run a few steps and report activation magnitudes")
    p.add_argument("--probe-steps", type=int, default=20)
    p.add_argument("--debug-checksum", action="store_true", help="log a parameter checksum after every merge, per rank")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def setup_distributed(args) -> tuple[int, int, int]:
    """Returns (rank, world, local_rank). Single process when torchrun did not start several."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), local_rank


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank, world, local_rank = setup_distributed(args)
    is_root = rank == 0
    token = os.environ.get("PW_TOKEN") or None
    coord = args.coordinator.rstrip("/")
    started = time.time()
    torch.manual_seed(args.seed + args.replica)
    rng = random.Random(args.seed + 1000 * args.replica + rank)

    if args.device == "auto":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = device.type == "cuda" and args.precision in ("auto", "fp16")
    if args.precision == "fp16" and device.type != "cuda":
        raise SystemExit("fp16 needs a CUDA device")

    global GROUP, EVENTS
    GROUP = args.group
    if args.events and is_root and token:
        EVENTS = Events(coord, token)

    # ---- configuration: from the group's dataset, or from the coordinator's served folder
    if args.data_dir is None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if is_root:
            for name in SERVED[:2]:
                if not (args.work_dir / name).exists():
                    fetch_file(coord, token, name, args.work_dir / name)
        if world > 1:
            dist.barrier()
    config_dir = args.data_dir if args.data_dir is not None else args.work_dir
    from transformers import Gemma3TextConfig

    cfg = Gemma3TextConfig.from_json_file(str(config_dir / "config.json"))
    model = arch.build_model(cfg)
    vocab = cfg.vocab_size
    names = [n for n, _ in model.named_parameters()]
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    model = model.to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    pad = int(cfg.pad_token_id or 0)

    # ---- data
    if args.data_dir is None:
        for split in ("train", "heldout"):
            d = args.work_dir / split
            d.mkdir(exist_ok=True)
            for name in ("meta.json", "offsets.i64", "comp.i32", "tokens.u16"):
                target = d / name
                if is_root and not target.exists():
                    fetch_file(coord, token, f"{split}_{name}", target)
        if world > 1:
            dist.barrier()
        train_root, held_root = args.work_dir / "train", args.work_dir / "heldout"
    else:
        train_root, held_root = args.data_dir / "train", args.data_dir / "heldout"
        if not (train_root / "meta.json").exists():
            raise SystemExit(f"no training data at {train_root}; attach the group's dataset to the notebook")
    train_ds = pack.PackedDataset(train_root)
    held_ds = pack.PackedDataset(held_root)
    if args.data_dir is not None:
        my_shard = pack.shard_indices(len(train_ds), rank, world)  # the group's own shard, split across its GPUs
    else:
        my_shard = pack.shard_indices(len(train_ds), args.replica * world + rank, args.replicas * world)
    batches = Batches(train_ds, my_shard, args.micro_batch, rng)

    # ---- starting weights. Round 0: this group's deterministic init, checked against the coordinator.
    # Later rounds: join at the coordinator's current global.
    net: Net | None = None
    version = 0
    if is_root:
        net = Net(coord, token, args.group, args.global_dtype, names)
        status = get_json(coord, token, "/v1/status")
        version = int(status["round"])
        if version == 0:
            init = initrng.init_state(shapes, args.init_seed)
            init_sha = initrng.canonical_sha256(init)
            if init_sha != status["global_sha256"]:
                raise SystemExit(f"starting weights do not match the coordinator (ours {init_sha[:12]}, coordinator "
                                 f"{status['global_sha256'][:12]}). Check --init-seed and the torch and transformers versions.")
            load_into(model, {n: torch.from_numpy(v) for n, v in init.items()})
            del init
        else:
            version, state = fetch_global(coord, token, args.global_dtype, names)
            load_into(model, state)
            del state
        log("start", group=args.group, replica=args.replica, replicas=args.replicas, world=world, device=str(device),
            params=sum(p.numel() for p in model.parameters()), vocab=vocab, version=version,
            train_examples=len(train_ds), shard_examples=len(my_shard), precision="fp16" if amp else "fp32",
            inner_steps=args.inner_steps, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
            global_sha256=status["global_sha256"], data=str(args.data_dir or args.work_dir),
            events=EVENTS is not None)
    if world > 1:
        # Every rank must start from the same weights: broadcast rank 0's copy.
        params_now = [p for p in model.parameters()]
        synced = broadcast_tensors([p.detach() for p in params_now], [p.shape for p in params_now], device)
        with torch.no_grad():
            for p, s in zip(params_now, synced):
                p.copy_(s)
        del synced

    model_ddp = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
    params = [p for p in model.parameters()]
    shapes = [p.shape for p in params]
    base = [p.detach().clone() for p in params]  # last merged global, on device
    base_version = version if is_root else 0
    snapshot: list[torch.Tensor] | None = None
    outstanding = False
    required: int | None = None  # the global version that will include our delta
    steps_since_send = 0
    step = 0
    sends = merges = 0
    final_sent = False
    final_deadline = None
    seq_counter = 0

    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                                  fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pad_id = pad
    last_log = time.time()
    window_loss, window_tokens, window_t0 = [], 0, time.time()
    total_steps = args.total_steps or args.inner_steps * max(1, args.max_rounds or 100)

    # ---- probe mode: activation magnitudes and fp16 health, then stop
    if args.probe_only:
        maxima: list[float] = []

        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            maxima.append(float(t.detach().abs().max().item()))

        handles = [layer.register_forward_hook(hook) for layer in model.model.layers]
        for i in range(args.probe_steps):
            idxs = batches.next()
            ids_np, labels_np, mask_np = pack.collate(train_ds, idxs, pad_id)
            ids = torch.from_numpy(ids_np).to(device)
            labels = torch.from_numpy(labels_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            maxima.clear()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
            finite = bool(torch.isfinite(loss).item())
            if is_root:
                log("probe", group=args.group, step=i, loss=round(float(loss.item()), 4), finite=finite,
                    max_activation=round(max(maxima) if maxima else float("nan"), 2),
                    tokens=int(mask.sum().item()))
            loss.backward()
            model.zero_grad(set_to_none=True)
        for h in handles:
            h.remove()
        if is_root:
            log("done", reason="probe")
        if world > 1:
            dist.destroy_process_group()
        return 0

    current_lr = time_lr(0, 0.0, args.lr, args.warmup_steps)  # time-based schedule; updated at sync points
    try:
        while True:
            # ---------------- one optimizer step (all ranks)
            lr = lr_at(step, total_steps, args.lr, args.warmup_steps) if args.total_steps else current_lr
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for micro in range(args.grad_accum):
                idxs = batches.next()
                ids_np, labels_np, mask_np = pack.collate(train_ds, idxs, pad_id)
                ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
                labels = torch.from_numpy(labels_np).to(device, non_blocking=True)
                mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
                sync_ctx = model_ddp.no_sync() if (world > 1 and micro < args.grad_accum - 1) else contextlib.nullcontext()
                with sync_ctx:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        logits = model_ddp(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                    loss = F.cross_entropy(logits.reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
                    scaled = loss / args.grad_accum
                    if amp:
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                loss_acc += loss.item() / args.grad_accum
                window_tokens += int(mask.sum().item())
            if amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            step += 1
            steps_since_send += 1
            window_loss.append(loss_acc)

            # ---------------- monitoring (rank 0)
            if is_root and time.time() - last_log >= args.log_every_s:
                elapsed = max(1e-6, time.time() - window_t0)
                log("step", group=args.group, step=step, lr=float(f"{lr:.3g}"), loss=round(sum(window_loss) / len(window_loss), 4),
                    tok_s=round(window_tokens / elapsed), merges=merges, sends=sends, outstanding=outstanding,
                    version=base_version, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
                    epoch=batches.epoch)
                last_log = time.time()
                window_loss, window_tokens, window_t0 = [], 0, time.time()

            if step % args.sync_every != 0:
                continue

            # ---------------- sync point: rank 0 decides, everyone follows
            decision = NOTHING
            got_global = None
            if is_root:
                out_of_time = (time.time() - started) / 60.0 > args.max_minutes
                rounds_done = args.max_rounds > 0 and sends >= args.max_rounds
                if outstanding:
                    res = net.upload_result()
                    if res is not None and required is None:
                        if res.get("accepted") is not None:
                            required = int(res["accepted"]) + 1  # the global that includes this delta
                        else:
                            required = base_version + 1  # refused or failed: resync from the next global
                            log("delta_dropped", group=args.group, reason=res.get("reason"))
                        net.request_global(required)
                    if required is not None:
                        got_global = net.take_global(required)  # taken once, used below
                        if got_global is not None:
                            decision = MERGE
                    if decision == NOTHING and final_deadline is not None and time.time() > final_deadline:
                        log("final_timeout", group=args.group)
                        decision = FINISH
                elif out_of_time and not final_sent and steps_since_send > 0:
                    decision = SEND  # the last leftover steps, then wait for the merge
                    final_sent = True
                    final_deadline = time.time() + args.final_wait_minutes * 60.0
                elif out_of_time or rounds_done:
                    decision = FINISH  # no more sends: the last one has been merged
                elif steps_since_send >= args.inner_steps:
                    decision = SEND
            lr_next = float(lr)
            if is_root and not args.total_steps:
                lr_next = time_lr(step, (time.time() - started) / 60.0 / max(args.max_minutes, 1e-6), args.lr, args.warmup_steps)
            flag = torch.tensor([float(decision), lr_next], device=device, dtype=torch.float64)
            if world > 1:
                dist.broadcast(flag, src=0)
            decision = int(flag[0].item())
            if not args.total_steps:
                current_lr = float(flag[1].item())

            if decision == SEND:
                snapshot = [p.detach().clone() for p in params]
                if is_root:
                    # Time-based sequence numbers stay increasing even across worker restarts.
                    seq_counter = max(seq_counter + 1, int(time.time() * 1000))
                    net.submit(seq_counter, base_version, delta_fp16(base, params, names))
                    sends += 1
                    log("send", group=args.group, round=sends, base_version=base_version, steps=steps_since_send,
                        step=step, final=final_sent)
                outstanding = True
                required = None
                steps_since_send = 0

            elif decision == MERGE:
                if is_root:
                    version_new, state = got_global
                    gl = [state[n].to(device=device, dtype=torch.float32) for n in names]
                    del state
                    gl = broadcast_tensors(gl, shapes, device)
                else:
                    gl = broadcast_tensors(None, shapes, device)
                assert snapshot is not None, "a merge always follows a send"
                delayed_merge(params, snapshot, gl)
                for b, g in zip(base, gl):
                    b.copy_(g)
                del gl
                snapshot = None
                outstanding = False
                required = None
                merges += 1  # every rank counts, so checksum lines line up across ranks
                if is_root:
                    base_version = version_new
                    log("merge", group=args.group, version=version_new, merges=merges, step=step)
                    if args.eval_every and merges % args.eval_every == 0:
                        loss_h = heldout_loss(model, held_ds, pad_id, vocab, device, amp, args.eval_examples, args.micro_batch)
                        log("eval", group=args.group, version=version_new, heldout_loss=round(loss_h, 4), step=step)
                if args.debug_checksum:
                    print(f"PW {json.dumps({'ev': 'checksum', 'rank': rank, 'value': round(checksum(params), 6), 'merges': merges})}", flush=True)

            elif decision == FINISH:
                if is_root:
                    log("done", group=args.group, step=step, sends=sends, merges=merges, version=base_version,
                        minutes=round((time.time() - started) / 60.0, 1))
                break
    except Exception:
        if is_root:
            log("error", group=args.group, step=step, trace=traceback.format_exc()[-1500:])
        raise
    finally:
        if net is not None:
            net.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
