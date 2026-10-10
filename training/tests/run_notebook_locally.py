"""Rehearse a generated Kaggle notebook on this machine, against a local coordinator.

    python -m training.tests.run_notebook_locally kaggle/plan/k01/pw_worker.ipynb rehearsal/k01 \
        --replace-url https://coord.example.org --with-url http://127.0.0.1:8770

The cells run in order: the pip cell is skipped (this machine already has the libraries), the
%%writefile cells write the training package into the rehearsal folder, and the launch cell runs
with the coordinator URL replaced. The launch cell starts the same processes it would start on
Kaggle (torchrun for DDP, one process per GPU for independent mode), with gloo on the CPU here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", type=Path)
    p.add_argument("workdir", type=Path)
    p.add_argument("--replace-url", required=True)
    p.add_argument("--with-url", required=True)
    args = p.parse_args(argv)

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    sys.path.insert(0, str(args.workdir))
    ran = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(line.lstrip().startswith("!") for line in src.splitlines()):
            ran.append(f"cell {i}: skipped (shell line, pip)")
            continue
        m = re.match(r"%%writefile (\S+)\n", src)
        if m:
            target = Path(m.group(1))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(src[m.end():], encoding="utf-8")
            ran.append(f"cell {i}: wrote {target}")
            continue
        src = src.replace(args.replace_url, args.with_url)
        ran.append(f"cell {i}: running")
        print(f"--- {ran[-1]}", flush=True)
        exec(compile(src, f"<notebook cell {i}>", "exec"), {"__name__": "__main__"})
    print("\n".join(ran), flush=True)


if __name__ == "__main__":
    main()
