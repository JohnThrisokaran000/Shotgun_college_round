"""Summarise group logs. The notebook output is the only feedback channel, so every monitoring line
is "PW {json}". This reads one or more log files and prints one status line per group.

    python -m training.logparse runs/*/g0.log runs/*/g1.log
    python -m training.logparse --json kaggle-output/*.log
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _new_group() -> dict:
    return {
        "start": None, "last_step": None, "loss": None, "tok_s": None, "merges": 0, "sends": 0,
        "version": None, "evals": [], "errors": [], "net_errors": 0, "dropped": 0, "done": False,
        "tracebacks": 0, "probe": [], "files": [],
    }


def parse(paths: list[Path]) -> dict:
    """One entry per group. Lines without a group field (for example per-rank checksums) are skipped."""
    groups: dict[str, dict] = defaultdict(_new_group)
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        seen: set[str] = set()
        for line in text.splitlines():
            if not line.startswith("PW "):
                continue
            try:
                d = json.loads(line[3:])
            except json.JSONDecodeError:
                continue
            if "group" not in d:
                continue
            name = d["group"]
            seen.add(name)
            g = groups[name]
            g["files"] = sorted(set(g["files"]) | {str(path)})
            ev = d.get("ev")
            if ev == "start":
                g["start"] = {k: d.get(k) for k in ("replica", "world", "device", "params", "precision", "version")}
            elif ev == "step":
                g["last_step"], g["loss"], g["tok_s"] = d.get("step"), d.get("loss"), d.get("tok_s")
                g["merges"] = max(g["merges"], d.get("merges", 0))
                g["sends"] = max(g["sends"], d.get("sends", 0))
                g["version"] = d.get("version", g["version"])
            elif ev == "send":
                g["sends"] = max(g["sends"], d.get("round", 0))
            elif ev == "merge":
                g["merges"] = max(g["merges"], d.get("merges", 0))
                g["version"] = d.get("version", g["version"])
            elif ev == "eval":
                g["evals"].append((d.get("version"), d.get("heldout_loss")))
            elif ev == "delta_dropped":
                g["dropped"] += 1
            elif ev == "net_error":
                g["net_errors"] += 1
            elif ev == "error":
                g["errors"].append(d.get("trace", "")[-300:])
            elif ev == "done":
                g["done"] = True
        tracebacks = text.count("Traceback (most recent call last)")
        for name in seen:
            groups[name]["tracebacks"] += tracebacks
    return dict(groups)


def render(groups: dict) -> str:
    lines = []
    for name, g in sorted(groups.items()):
        evals = g["evals"]
        trend = f"{evals[0][1]} -> {evals[-1][1]}" if len(evals) >= 2 else (str(evals[-1][1]) if evals else "-")
        status = "done" if g["done"] else ("ERROR" if g["errors"] or g["tracebacks"] else "running")
        # tok/s is rank 0's throughput: a DDP group also runs other ranks that are not counted here.
        lines.append(
            f"{name:<10} {status:<8} step {g['last_step']}  loss {g['loss']}  tok/s {g['tok_s']}  "
            f"sends {g['sends']}  merges {g['merges']}  version {g['version']}  heldout {trend}  "
            f"dropped {g['dropped']}  net_errors {g['net_errors']}  tracebacks {g['tracebacks']}"
        )
    return "\n".join(lines) if lines else "(no PW lines found)"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    groups = parse(args.logs)
    if args.json:
        print(json.dumps(groups, indent=2, default=str))
    else:
        print(render(groups))


if __name__ == "__main__":
    main()
