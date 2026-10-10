"""Status checks across all accounts: dataset readiness and kernel state.

    python -m training.kaggle_ops datasets --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env --wait 1800
    python -m training.kaggle_ops kernels  --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env

Keys go only into the environment of the kaggle CLI, and any key that shows up in its output is masked.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from training.kaggle_upload import read_accounts, read_keys, slug_for


def _run(args: list[str], key: str) -> str:
    proc = subprocess.run(args, env=dict(os.environ, KAGGLE_API_TOKEN=key), capture_output=True, text=True)
    text = (proc.stdout + proc.stderr).strip().replace(key, "***")
    return text.splitlines()[-1] if text else f"exit {proc.returncode}"


def dataset_states(accounts, keys) -> dict[str, str]:
    return {alias: _run(["kaggle", "datasets", "status", f"{user}/{slug_for(alias)}"], keys[alias]).lower()
            for alias, user in accounts}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["datasets", "kernels"])
    p.add_argument("--accounts", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--wait", type=int, default=0, help="datasets: poll until all are ready, up to this many seconds")
    args = p.parse_args(argv)
    accounts, keys = read_accounts(args.accounts), read_keys(args.keys)

    if args.what == "datasets":
        deadline = time.time() + args.wait
        while True:
            states = dataset_states(accounts, keys)
            ready = sum("ready" in s for s in states.values())
            print(f"{time.strftime('%H:%M:%S')} datasets ready {ready}/{len(states)}: "
                  + ", ".join(f"{a}={s[:24]}" for a, s in states.items()), flush=True)
            if ready == len(states) or time.time() > deadline:
                raise SystemExit(0 if ready == len(states) else 1)
            time.sleep(30)
    for alias, user in accounts:
        print(f"{alias:<8} {_run(['kaggle', 'kernels', 'status', f'{user}/pw-worker-{alias}'], keys[alias])}", flush=True)


if __name__ == "__main__":
    main()
