"""Follow the coordinator's events.jsonl and print one short line per notable event.

    python -m training.live runs/state/events.jsonl

Start, probe summary, send, merge, eval, drops, network errors, errors and done are always shown.
Step events are shown once per --step-minutes per group, with the fleet-wide numbers.
The coordinator's own "outer" lines (new global version) are in runs/coordinator.log.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def fmt(e: dict) -> str | None:
    g, ev = e.get("group", "?"), e.get("ev")
    if ev == "start":
        return (f"{g}: START gpus={e.get('world')} device={e.get('device')} {e.get('precision')} "
                f"v{e.get('version')} shard={e.get('train_examples')} ex")
    if ev == "probe":
        return None
    if ev == "send":
        return f"{g}: SEND round {e.get('round')} base v{e.get('base_version')} after {e.get('steps')} steps"
    if ev == "merge":
        return f"{g}: MERGE -> v{e.get('version')} (merge #{e.get('merges')}, step {e.get('step')})"
    if ev == "eval":
        return f"{g}: EVAL heldout loss {e.get('heldout_loss')} at v{e.get('version')}"
    if ev == "done":
        return f"{g}: DONE {json.dumps({k: e[k] for k in ('step', 'sends', 'merges', 'minutes') if k in e})}"
    if ev in ("delta_dropped", "net_error", "error", "final_timeout", "event_post_failed"):
        detail = e.get("trace") or e.get("error") or e.get("reason") or ""
        return f"{g}: !! {ev} {str(detail)[-240:]}"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events", type=Path)
    p.add_argument("--step-minutes", type=float, default=10.0)
    args = p.parse_args()
    last_step: dict[str, float] = {}
    probe: dict[str, list] = {}
    while not args.events.exists():
        time.sleep(2)
    buffer = b""
    with args.events.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                time.sleep(1)
                continue
            buffer += chunk
            lines, buffer = buffer.rsplit(b"\n", 1) if b"\n" in buffer else ([], buffer)
            for raw in (lines.split(b"\n") if lines else []):
                handle(raw, last_step, probe, args.step_minutes)


def handle(raw: bytes, last_step: dict, probe: dict, step_minutes: float) -> None:
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        return
    g, ev = e.get("group", "?"), e.get("ev")
    if ev == "probe":
        probe.setdefault(g, []).append(e)
        if len(probe[g]) == 20:
            rows = probe[g]
            print(f"{g}: PROBE 20 steps finite={all(r['finite'] for r in rows)} "
                  f"loss {rows[0]['loss']}->{rows[-1]['loss']} max_act={max(r['max_activation'] for r in rows)}", flush=True)
        return
    if ev == "step":
        now = time.time()
        if now - last_step.get(g, 0) >= step_minutes * 60:
            last_step[g] = now
            print(f"{g}: step {e.get('step')} loss {e.get('loss')} tok/s {e.get('tok_s')} "
                  f"mem {e.get('mem_gb')}GB v{e.get('version')}", flush=True)
        return
    out = fmt(e)
    if out:
        print(out, flush=True)


if __name__ == "__main__":
    main()
