"""One table of the whole run, from the coordinator's events.jsonl.

    python -m training.status runs/state/events.jsonl            # print once
    python -m training.status runs/state/events.jsonl --watch 30  # redraw every 30 s

Per group: state, last step, loss, tokens/s, sends, merges, global version, held-out loss
(first -> latest), how long ago it last reported, and when its current session started. A group that has
not reported for several minutes shows STALE. A group whose step counts go backwards after its newest session
started has two sessions posting at once, and gets a note: a re-push starts a new session, and nothing stops
the old one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


RECENT_S = 15 * 60  # errors older than this are history, not current issues
SESSION_GAP_S = 10 * 60  # a session logs its probe start and its training start a few minutes apart
SETTLE_S = 60  # ignore the first minute after a new session starts, when the old one may still be finishing


def load(path: Path, errors_after: float = 0.0) -> dict[str, dict]:
    """errors_after: errors received before this time are history and are not counted. The events file keeps them."""
    groups: dict[str, dict] = {}
    now = time.time()
    if not path.exists():
        return groups
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        g = groups.setdefault(e.get("group", "?"), {"evals": [], "starts": [], "steps": [], "state": "starting", "errors": 0, "gpus": None})
        g["seen"] = e.get("received", e.get("t", 0))
        ev = e.get("ev")
        if ev == "start":
            g["state"], g["gpus"] = "training", e.get("world")
            g["starts"].append(float(e.get("received", e.get("t", 0)) or 0))
        elif ev == "step":
            g.update(step=e.get("step"), loss=e.get("loss"), tok_s=e.get("tok_s"), version=e.get("version"),
                     sends=e.get("sends"), merges=e.get("merges"))
            if e.get("step") is not None:
                g["steps"].append((float(e.get("received", e.get("t", 0)) or 0), int(e["step"])))
        elif ev == "eval":
            g["evals"].append(e.get("heldout_loss"))
        elif ev == "done" and e.get("reason") != "probe":  # the probe process also logs a done
            g["state"] = "done"
        elif ev == "error":
            g["state"], g["errors"] = "ERROR", g["errors"] + 1
        elif ev in ("net_error", "delta_dropped") and now - e.get("received", e.get("t", 0)) < RECENT_S and e.get("received", e.get("t", 0)) >= errors_after:
            g["errors"] += 1  # only recent ones: old network blips are history
    return groups


def sessions(starts: list[float]) -> list[float]:
    """Start time of each session. Starts closer together than SESSION_GAP_S belong to one session."""
    out: list[float] = []
    last = None
    for t in sorted(starts):
        if last is None or t - last > SESSION_GAP_S:
            out.append(t)
        last = t
    return out


def interleaved(steps: list[tuple[float, int]], since: float) -> bool:
    """True when step counts go backwards after `since`. One session only ever counts up, so this means two are posting."""
    seq = [s for t, s in sorted(steps) if t >= since + SETTLE_S]
    return any(b < a for a, b in zip(seq, seq[1:]))


def render(groups: dict[str, dict], now: float | None = None) -> str:
    now = time.time() if now is None else now
    head = (f"{'group':<8}{'state':<10}{'gpus':>4}{'step':>8}{'loss':>8}{'tok/s':>8}{'sent':>5}{'merged':>7}{'ver':>4}  "
            f"{'heldout':<16}{'seen':>6}  {'since':>5}  issues")
    lines = [head]
    notes = []
    for name, g in sorted(groups.items()):
        age = int(now - g.get("seen", now))
        ev = g["evals"]
        held = f"{ev[0]}->{ev[-1]}" if len(ev) > 1 else (str(ev[0]) if ev else "-")
        state = g["state"] if age < 600 or g["state"] == "done" else "STALE"
        starts = sessions(g.get("starts", []))
        since = time.strftime("%H:%M", time.localtime(starts[-1])) if starts else "-"
        if starts and interleaved(g.get("steps", []), starts[-1]):
            notes.append(f"{name}: step counts go backwards since its newest session started, so two sessions are posting. "
                         "The older one keeps using GPU quota until it ends (there is no cancel command).")
        lines.append(f"{name:<8}{state:<10}{g['gpus'] or '-':>4}{g.get('step', '-'):>8}{g.get('loss', '-'):>8}"
                     f"{g.get('tok_s', '-'):>8}{g.get('sends', '-'):>5}{g.get('merges', '-'):>7}{g.get('version', '-'):>4}  "
                     f"{held:<16}{age:>5}s  {since:>5}  {g['errors'] or ''}")
    out = "\n".join(lines)
    live = [g for g in groups.values() if g.get("tok_s")]
    if live:
        out += (f"\n\nfleet: {len(live)} groups reporting, {sum(g['tok_s'] for g in live):,} tokens/s in total, "
                f"{max((g.get('version') or 0) for g in live)} newest global version")
    for note in notes:
        out += f"\nnote: {note}"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events", type=Path)
    p.add_argument("--watch", type=float, default=0.0)
    p.add_argument("--errors-after", type=float, default=0.0, help="unix time; earlier errors are not counted")
    args = p.parse_args()
    while True:
        groups = load(args.events, args.errors_after)
        out = render(groups) if groups else "no events yet"
        print(("\033[2J\033[H" if args.watch else "") + out, flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
