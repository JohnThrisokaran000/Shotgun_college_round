"""One table of the whole run, from the coordinator's events.jsonl.

    python -m training.status runs/state/events.jsonl            # print once
    python -m training.status runs/state/events.jsonl --watch 30  # redraw every 30 s

Per group: state, last step, loss, tokens/s, sends, merges, global version, held-out loss
(first -> latest), and how long ago it last reported. A group that has not reported for several
minutes shows STALE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    if not path.exists():
        return groups
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        g = groups.setdefault(e.get("group", "?"), {"evals": [], "state": "starting", "errors": 0, "gpus": None})
        g["seen"] = e.get("received", e.get("t", 0))
        ev = e.get("ev")
        if ev == "start":
            g["state"], g["gpus"] = "training", e.get("world")
        elif ev == "step":
            g.update(step=e.get("step"), loss=e.get("loss"), tok_s=e.get("tok_s"), version=e.get("version"),
                     sends=e.get("sends"), merges=e.get("merges"))
        elif ev == "eval":
            g["evals"].append(e.get("heldout_loss"))
        elif ev == "done":
            g["state"] = "done"
        elif ev == "error":
            g["state"], g["errors"] = "ERROR", g["errors"] + 1
        elif ev in ("net_error", "delta_dropped"):
            g["errors"] += 1
    return groups


def render(groups: dict[str, dict]) -> str:
    now = time.time()
    head = f"{'group':<8}{'state':<10}{'gpus':>4}{'step':>8}{'loss':>8}{'tok/s':>8}{'sent':>5}{'merged':>7}{'ver':>4}  {'heldout':<16}{'seen':>6}  issues"
    lines = [head]
    for name, g in sorted(groups.items()):
        age = int(now - g.get("seen", now))
        ev = g["evals"]
        held = f"{ev[0]}->{ev[-1]}" if len(ev) > 1 else (str(ev[0]) if ev else "-")
        state = g["state"] if age < 600 or g["state"] == "done" else "STALE"
        lines.append(f"{name:<8}{state:<10}{g['gpus'] or '-':>4}{g.get('step', '-'):>8}{g.get('loss', '-'):>8}"
                     f"{g.get('tok_s', '-'):>8}{g.get('sends', '-'):>5}{g.get('merges', '-'):>7}{g.get('version', '-'):>4}  "
                     f"{held:<16}{age:>5}s  {g['errors'] or ''}")
    live = [g for g in groups.values() if g.get("tok_s")]
    if live:
        lines.append(f"\nfleet: {len(live)} groups reporting, {sum(g['tok_s'] for g in live):,} tokens/s in total, "
                     f"{max((g.get('version') or 0) for g in live)} newest global version")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events", type=Path)
    p.add_argument("--watch", type=float, default=0.0)
    args = p.parse_args()
    while True:
        out = render(load(args.events))
        print(("\033[2J\033[H" if args.watch else "") + (out if load(args.events) else "no events yet"), flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
