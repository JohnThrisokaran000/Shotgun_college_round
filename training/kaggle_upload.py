"""Upload each account's shard as a private Kaggle dataset owned by that same account.

Dry run (default): prints what it would do, and sends nothing.

    python -m training.kaggle_upload --shards data/accounts --accounts kaggle/accounts.txt \
        --keys kaggle/secrets/keys.env

Upload: every gate must be given, and the account count must match the accounts file.

    python -m training.kaggle_upload --shards data/accounts --accounts kaggle/accounts.txt \
        --keys kaggle/secrets/keys.env --execute --confirm-accounts 14 --i-have-checked-kaggle-rules

Each shard folder is uploaded as it is (flat files, no subfolders). The folder gets a
dataset-metadata.json that names the account's dataset. The key goes only into the environment
of the kaggle CLI. Keys are never printed, and any key that shows up in CLI output is masked.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LICENCE = "other"  # third-party permissive sources, each listed in LICENSES.md inside the dataset


def read_accounts(path: Path) -> list[tuple[str, str]]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            alias, username = (part.strip() for part in line.split(",", 1))
            out.append((alias, username))
    if not out:
        raise SystemExit(f"no accounts in {path}")
    return out


def read_keys(path: Path) -> dict[str, str]:
    keys = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            alias, key = (part.strip() for part in line.split("=", 1))
            keys[alias] = key
    return keys


def slug_for(alias: str) -> str:
    return f"patchwork-shard-{alias}"


def write_metadata(folder: Path, alias: str, username: str) -> None:
    meta = {
        "id": f"{username}/{slug_for(alias)}",
        "title": f"PatchWork shard {alias}",
        "subtitle": "Training shard for one PatchWork group (private)",
        "description": ("Packed tokens (uint16), tokenizer and architecture for one group of the PatchWork "
                        "training run. Third-party sources and their licences are listed in LICENSES.md."),
        "licenses": [{"name": LICENCE}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _masked(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def upload(folder: Path, key: str) -> str:
    """Create the dataset. If it already exists, add a new version instead."""
    env = dict(os.environ, KAGGLE_API_TOKEN=key)
    created = subprocess.run(["kaggle", "datasets", "create", "-p", str(folder), "-r", "skip"],
                             env=env, capture_output=True, text=True)
    if created.returncode == 0:
        return "created"
    versioned = subprocess.run(["kaggle", "datasets", "version", "-p", str(folder), "-m", "shard refresh",
                                "-r", "skip"], env=env, capture_output=True, text=True)
    if versioned.returncode == 0:
        return "new version"
    detail = _masked(versioned.stderr or versioned.stdout or created.stderr, key).strip().splitlines()
    return f"failed: {detail[-1][:200] if detail else 'no output'}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=Path, required=True, help="folder with acctNN subfolders (training.shards build)")
    p.add_argument("--accounts", type=Path, required=True, help="alias,username per line")
    p.add_argument("--keys", type=Path, required=True, help="alias=KEY per line (never printed)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-accounts", type=int, default=0)
    p.add_argument("--i-have-checked-kaggle-rules", action="store_true")
    args = p.parse_args(argv)

    accounts = read_accounts(args.accounts)
    keys = read_keys(args.keys)
    for alias, username in accounts:
        folder = args.shards / alias
        if not (folder / "train_meta.json").exists():
            raise SystemExit(f"{folder} has no shard; run training.shards build first")
        if alias not in keys:
            raise SystemExit(f"no key for account {alias} in the keys file")
        print(f"{alias:<8} {username:<24} {folder}  ->  {slug_for(alias)}", flush=True)

    if not args.execute:
        print(f"\ndry run: {len(accounts)} datasets would be created (private, one per account). Nothing was sent.")
        return
    if not args.i_have_checked_kaggle_rules:
        raise SystemExit("refused: pass --i-have-checked-kaggle-rules after confirming the upload is allowed")
    if args.confirm_accounts != len(accounts):
        raise SystemExit(f"refused: --confirm-accounts {args.confirm_accounts} does not match {len(accounts)} accounts")

    failures = 0
    for alias, username in accounts:
        folder = args.shards / alias
        write_metadata(folder, alias, username)
        result = upload(folder, keys[alias])
        failures += result.startswith("failed")
        print(f"{alias:<8} {result}", flush=True)
    if failures:
        raise SystemExit(f"{failures} upload(s) failed")


if __name__ == "__main__":
    sys.exit(main())
