"""Restore the Kaggle groups after a coordinator URL change, without running two sessions at once.

For each account in the accounts file, wait until its kernel has finished (COMPLETE, ERROR, or CANCEL), then push
that account's folder of the plan once. A kernel that is still RUNNING or QUEUED is never pushed: a push starts a new
session while the old one keeps running, and that doubles the account's GPU use.

    python -m training.resume_when_complete --plan kaggle/plan8 --accounts kaggle/active2.txt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from training.keep_alive import restart, state  # noqa: E402
from training.kaggle_upload import read_accounts, read_keys  # noqa: E402

FINISHED = ("COMPLETE", "ERROR", "CANCEL")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=Path("kaggle/plan8"))
    p.add_argument("--accounts", type=Path, default=Path("kaggle/active2.txt"))
    p.add_argument("--keys", type=Path, default=Path("kaggle/secrets/keys.env"))
    p.add_argument("--interval", type=int, default=120, help="seconds between status checks")
    args = p.parse_args()
    accounts = read_accounts(args.accounts)
    keys = read_keys(args.keys)
    pending = dict(accounts)
    while pending:
        for alias, user in list(pending.items()):
            now = state(user, alias, keys[alias])
            if now in FINISHED:
                out = restart(user, alias, keys[alias], args.plan)
                print(time.strftime("%H:%M:%S"), alias, f"was {now}, pushed once:", out, flush=True)
                del pending[alias]
        if pending:
            time.sleep(args.interval)
    print(time.strftime("%H:%M:%S"), "all accounts pushed once with the new coordinator URL", flush=True)


if __name__ == "__main__":
    main()
