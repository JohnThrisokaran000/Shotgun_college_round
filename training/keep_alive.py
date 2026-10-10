"""Keep every group's Kaggle kernel running.

    python -m training.keep_alive --plan kaggle/plan4 --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env

Every --interval seconds, for each account: if its kernel is RUNNING, leave it alone. If it is QUEUED (or in
any state other than RUNNING and COMPLETE) for longer than --grace seconds since it was last pushed, delete it
and push the same notebook again. COMPLETE is the normal end of a run and is never restarted. A group that
crashes mid-run is restarted too, and rejoins at the coordinator's current global version.

Each action is printed as one line. Keys go only into the kaggle CLI's environment and are masked in output.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.kaggle_ops import _run
from training.kaggle_upload import read_accounts, read_keys


def state(user: str, alias: str, key: str) -> str:
    text = _run(["kaggle", "kernels", "status", f"{user}/pw-worker-{alias}"], key)
    for name in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"):
        if name in text:
            return name
    return "UNKNOWN"


def last_reports(events: Path) -> dict[str, float]:
    """Latest time each group posted an event. A group that is RUNNING but silent is stalled."""
    seen: dict[str, float] = {}
    if not events.exists():
        return seen
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines()[-5000:]:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        group = e.get("group")
        if group:
            seen[group] = max(seen.get(group, 0.0), float(e.get("received", e.get("t", 0)) or 0))
    return seen


def restart(user: str, alias: str, key: str, plan: Path) -> str:
    _run(["kaggle", "kernels", "delete", "-y", f"{user}/pw-worker-{alias}"], key)
    return _run(["kaggle", "kernels", "push", "-p", str(plan / alias)], key)[:40]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--accounts", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--grace", type=int, default=30, help="seconds a pushed kernel may take to reach RUNNING")
    p.add_argument("--max-restarts", type=int, default=40, help="per account, so a broken account cannot loop forever")
    p.add_argument("--events", type=Path, default=Path("runs/state/events.jsonl"))
    p.add_argument("--silent", type=float, default=10.0, help="minutes a RUNNING kernel may stay silent before it counts as stalled")
    args = p.parse_args()
    accounts, keys = read_accounts(args.accounts), read_keys(args.keys)
    pushed_at = {alias: time.time() for alias, _ in accounts}  # assume a kernel already on Kaggle was pushed just now
    restarts = {alias: 0 for alias, _ in accounts}
    seen: dict[str, str] = {}

    def check(item):
        alias, user = item
        now = state(user, alias, keys[alias])
        reports = last_reports(args.events)
        silent_since = max(pushed_at[alias], reports.get(alias, 0.0))
        stalled = now == "RUNNING" and time.time() - silent_since > args.silent * 60
        label = "RUNNING (silent, stalled)" if stalled else now
        if label != seen.get(alias):
            print(f"{time.strftime('%H:%M:%S')} {alias}: {seen.get(alias, '?')} -> {label}", flush=True)
            seen[alias] = label
        if now == "COMPLETE" or (now == "RUNNING" and not stalled):
            return
        if time.time() - pushed_at[alias] < args.grace:
            return
        if restarts[alias] >= args.max_restarts:
            return
        restarts[alias] += 1
        result = restart(user, alias, keys[alias], args.plan)
        pushed_at[alias] = time.time()
        print(f"{time.strftime('%H:%M:%S')} {alias}: was {now}, restarted (#{restarts[alias]}): {result}", flush=True)

    while True:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, accounts))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
