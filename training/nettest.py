"""Find out which Kaggle accounts have working internet, and can reach the coordinator.

    PW_TOKEN=... python -m training.nettest --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env \
        --url https://coordinator.example [--only acct15 acct16]

Kaggle silently disables internet on accounts it has not verified, even when a kernel asks for it. This pushes a
tiny CPU-only kernel to each account that tries PyPI and the coordinator, waits, and prints one line per account.
Keys go only into the kaggle CLI's environment, and are masked in any output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from training.kaggle_ops import _run
from training.kaggle_upload import read_accounts, read_keys

CODE = """import urllib.request
def probe(name, req):
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return name + "=ok(" + str(r.status) + ")"
    except Exception as e:
        return name + "=FAIL(" + type(e).__name__ + ")"
req = urllib.request.Request("URL/v1/status", headers={"Authorization": "Bearer TOKEN"})
print("NETTEST", probe("pypi", "https://pypi.org/simple/pip/"), probe("coordinator", req), flush=True)
"""


def write_kernel(root: Path, alias: str, user: str, url: str, token: str) -> Path:
    folder = root / alias
    folder.mkdir(parents=True, exist_ok=True)
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": CODE.replace("URL", url).replace("TOKEN", token).splitlines(True)}
    notebook = {"cells": [cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 4}
    (folder / "nettest.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    meta = {"id": f"{user}/pw-nettest-{alias}", "title": f"pw-nettest-{alias}", "code_file": "nettest.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accounts", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--wait", type=int, default=900)
    p.add_argument("--purge", action="store_true",
                   help="remove accounts without internet from the accounts and keys files (backups are kept)")
    args = p.parse_args()
    token = os.environ["PW_TOKEN"]
    accounts = [(a, u) for a, u in read_accounts(args.accounts) if not args.only or a in args.only]
    keys, root = read_keys(args.keys), Path("kaggle/nettest")
    users = dict(accounts)
    from concurrent.futures import ThreadPoolExecutor

    def push(item):
        alias, user = item
        folder = write_kernel(root, alias, user, args.url, token)
        return alias, _run(["kaggle", "kernels", "push", "-p", str(folder)], keys[alias])[:50]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for alias, result in pool.map(push, accounts):
            print(f"{alias:<8} push: {result}", flush=True)
    pending, deadline = set(users), time.time() + args.wait
    results: dict[str, bool] = {}
    while pending and time.time() < deadline:
        time.sleep(10)
        for alias in sorted(pending):
            ref = f"{users[alias]}/pw-nettest-{alias}"
            status = _run(["kaggle", "kernels", "status", ref], keys[alias])
            if "COMPLETE" in status or "ERROR" in status:
                out = root / "out" / alias
                out.mkdir(parents=True, exist_ok=True)
                _run(["kaggle", "kernels", "output", ref, "-p", str(out)], keys[alias])
                text = "".join(f.read_text(errors="replace") for f in out.glob("*.log"))
                at = text.find("NETTEST")
                line = text[at:at + 80].split(chr(92))[0] if at >= 0 else "no result in log"
                results[alias] = "pypi=ok" in line and "coordinator=ok" in line
                print(f"{alias:<8} {line}", flush=True)
                pending.discard(alias)
    for alias in sorted(pending):
        print(f"{alias:<8} still running after {args.wait}s", flush=True)
    working = sorted(a for a, ok in results.items() if ok)
    print(f"\nWORKING {len(working)}/{len(users)}: {' '.join(working)}", flush=True)
    if args.purge:
        dead = {a for a, ok in results.items() if not ok}
        for path, sep in ((args.accounts, ","), (args.keys, "=")):
            kept = [l for l in path.read_text(encoding="utf-8").splitlines()
                    if not l.strip() or l.startswith("#") or l.split(sep)[0].strip() not in dead]
            path.with_name(path.name + ".bak-before-purge").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"purged {len(dead)} accounts without internet: {' '.join(sorted(dead))}", flush=True)


if __name__ == "__main__":
    main()
