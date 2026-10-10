"""Tokenizer training, and one dataset per Kaggle account.

    python -m training.shards tokenizer --general data/corpus/general --patchwork data/patchwork \
        --out runs/tokenizer.json --vocab 32768 --sample-mb 100

    python -m training.shards build --general data/corpus/general --patchwork data/patchwork \
        --tokenizer runs/tokenizer.json --serve runs/serve --out data/accounts --accounts 14 --pw-share 0.2

Each account folder gets:
    train_*.{u16,i64,...}, heldout_*   packed tokens: its general shard plus PatchWork rows (--pw-share of the
                 tokens), and the held-out rows, which are the same in every account (flat names, see below)
    config.json, tokenizer.json, run.json, LICENSES.md

Training order is random at run time (length-bucketed shuffling in training/pack.py), so the rows
are written in order and not shuffled here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Iterator

from training import pack, pwdata

GENERAL_SHARD = "shard_{:02d}.jsonl"


def _rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _pair(row: dict) -> dict:
    return {"prompt": row["prompt"], "completion": row["completion"]}


def _sample_texts(general: Path, patchwork: Path, sample_bytes: int, pw_share: float, seed: int) -> Iterator[str]:
    """Texts for tokenizer training: a random sample of general chunks, plus PatchWork rows."""
    rng = random.Random(seed)
    budget_general = int(sample_bytes * (1 - pw_share))
    used = 0
    for shard in sorted(general.glob("shard_*.jsonl")):
        for row in _rows(shard):
            if used >= budget_general:
                break
            if rng.random() < 0.05:  # thin out, so the sample spreads over every shard
                used += len(row["completion"].encode("utf-8"))
                yield row["completion"]
    used = 0
    budget_pw = sample_bytes - budget_general
    with (patchwork / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if used >= budget_pw:
                return
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["prompt"] + row["completion"]
            used += len(text.encode("utf-8"))
            yield text


def train_tokenizer(args: argparse.Namespace) -> None:
    texts = _sample_texts(args.general, args.patchwork, int(args.sample_mb * 1_000_000), args.pw_share_sample, args.seed)
    pwdata.train_tokenizer(texts, args.out, vocab_size=args.vocab)
    tok = pwdata.load_tokenizer(args.out)
    print(json.dumps({"tokenizer": str(args.out), "vocab": tok.get_vocab_size(), "sample_mb": args.sample_mb}, indent=2))


def _count_tokens(tok, rows: Iterable[dict], batch: int = 4096) -> tuple[int, int]:
    """(rows, tokens) as pack would count them: prompt, completion, and two special tokens per row."""
    n_rows = n_tokens = 0
    buf: list[dict] = []

    def flush() -> None:
        nonlocal n_rows, n_tokens
        prompts = tok.encode_batch([r["prompt"] for r in buf])
        completions = tok.encode_batch([r["completion"] for r in buf])
        for a, b in zip(prompts, completions):
            n_rows += 1
            n_tokens += len(a.ids) + len(b.ids) + 2
        buf.clear()

    for row in rows:
        buf.append(row)
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    return n_rows, n_tokens


def _split_patchwork(patchwork: Path, counts: list[int], work: Path) -> list[Path]:
    """One pass over the PatchWork rows. Account i gets the next counts[i] rows, written to its own file."""
    work.mkdir(parents=True, exist_ok=True)
    paths = [work / f"pw_{i:02d}.jsonl" for i in range(len(counts))]
    ends = list(itertools.accumulate(counts))
    handles = [p.open("w", encoding="utf-8") for p in paths]
    try:
        account, index = 0, 0
        for line in (patchwork / "train.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            while account < len(counts) and index >= ends[account]:
                account += 1
            if account >= len(counts):
                break
            handles[account].write(line)
            index += 1
    finally:
        for handle in handles:
            handle.close()
    return paths


def build(args: argparse.Namespace) -> None:
    tok = pwdata.load_tokenizer(args.tokenizer)
    bos, eos, pad = pwdata.ids(tok, pwdata.BOS), pwdata.ids(tok, pwdata.EOS), pwdata.ids(tok, pwdata.PAD)
    accounts = args.accounts
    for account in range(accounts):
        if not (args.general / GENERAL_SHARD.format(account)).exists():
            raise SystemExit(f"missing general shard {GENERAL_SHARD.format(account)}; run training.corpus.build first")

    # Tokens per account for the general part, and the average size of a PatchWork row.
    general_tokens = []
    for account in range(accounts):
        _, tokens = _count_tokens(tok, (_pair(r) for r in _rows(args.general / GENERAL_SHARD.format(account))))
        general_tokens.append(tokens)
        print(f"account {account:02d}: general tokens {tokens:,}", flush=True)
    sample = [_pair(r) for r in itertools.islice(_rows(args.patchwork / "train.jsonl"), 2000)]
    sample_rows, sample_tokens = _count_tokens(tok, sample)
    avg_pw_tokens = sample_tokens / max(1, sample_rows)
    pw_total = sum(1 for _ in _rows_count(args.patchwork / "train.jsonl"))

    # PatchWork rows per account, so their tokens are about pw_share of the account's total.
    ratio = args.pw_share / (1 - args.pw_share)
    pw_per_account = [int(ratio * t / avg_pw_tokens) for t in general_tokens]
    if args.max_rows:
        pw_per_account = [min(n, args.max_rows) for n in pw_per_account]
    if sum(pw_per_account) > pw_total:
        scale = pw_total / sum(pw_per_account)
        pw_per_account = [int(n * scale) for n in pw_per_account]
        print(f"PatchWork rows are short: using {scale:.0%} of the target share", flush=True)

    # Held-out: the general held-out rows and a fixed slice of PatchWork held-out rows, the same for every account.
    heldout = [_pair(r) for r in _rows(args.general / "heldout.jsonl")] if (args.general / "heldout.jsonl").exists() else []
    if (args.patchwork / "heldout.jsonl").exists():
        heldout += [_pair(r) for r in itertools.islice(_rows(args.patchwork / "heldout.jsonl"), args.pw_heldout_rows)]
    random.Random(args.seed).shuffle(heldout)

    work = args.out / "_pw_split"
    pw_paths = _split_patchwork(args.patchwork, pw_per_account, work)
    licences = args.general / "LICENSES.md"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"accounts": accounts, "pw_share_target": args.pw_share, "avg_pw_tokens": round(avg_pw_tokens, 1),
               "pw_rows_total": pw_total, "pw_rows_used": sum(pw_per_account), "accounts_detail": []}
    aliases = [f"acct{i:02d}" for i in range(accounts)]
    if args.accounts_file is not None:
        listed = [line.split(",")[0].strip() for line in args.accounts_file.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
        if len(listed) != accounts:
            raise SystemExit(f"{args.accounts_file} lists {len(listed)} accounts, but --accounts is {accounts}")
        aliases = listed
    for account in range(accounts):
        folder = args.out / aliases[account]
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        general_rows = _limited((_pair(r) for r in _rows(args.general / GENERAL_SHARD.format(account))), args.max_rows)
        pw_rows = (_pair(r) for r in _rows(pw_paths[account]))
        meta = pack.pack(itertools.chain(general_rows, pw_rows), tok, folder / "_stage_train",
                         max_len=args.max_len, bos=bos, eos=eos, pad=pad)
        held = pack.pack(iter(heldout), tok, folder / "_stage_heldout", max_len=args.max_len, bos=bos, eos=eos, pad=pad)
        # Flat names (train_tokens.u16 ...): Kaggle datasets are flat, and the notebook links them back into data/.
        for split in ("train", "heldout"):
            for name in ("tokens.u16", "offsets.i64", "comp.i32", "meta.json"):
                shutil.move(str(folder / f"_stage_{split}" / name), str(folder / f"{split}_{name}"))
            shutil.rmtree(folder / f"_stage_{split}")
        for name in ("config.json", "tokenizer.json"):
            shutil.copy(args.serve / name, folder / name)
        if licences.exists():
            shutil.copy(licences, folder / "LICENSES.md")
        info = {"account": account, "alias": aliases[account], "train_examples": meta["examples"], "train_tokens": meta["tokens"],
                "train_dropped_too_long": meta["dropped_too_long"], "pw_rows": pw_per_account[account],
                "pw_token_share": round(pw_per_account[account] * avg_pw_tokens / max(1, meta["tokens"]), 3),
                "heldout_examples": held["examples"], "heldout_tokens": held["tokens"],
                "max_len": args.max_len, "vocab": meta["vocab"]}
        (folder / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        summary["accounts_detail"].append(info)
        print(json.dumps(info), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    (args.out / "build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "accounts_detail"}, indent=2))


def _limited(rows: Iterable[dict], max_rows: int) -> Iterator[dict]:
    return itertools.islice(rows, max_rows) if max_rows else iter(rows)


def _rows_count(path: Path) -> Iterator[int]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokenizer", help="train the BPE tokenizer on a sample")
    t.add_argument("--general", type=Path, required=True)
    t.add_argument("--patchwork", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--vocab", type=int, default=32768)
    t.add_argument("--sample-mb", type=float, default=100.0)
    t.add_argument("--pw-share-sample", type=float, default=0.1, help="share of the sample taken from PatchWork")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train_tokenizer)

    b = sub.add_parser("build", help="build one dataset per account")
    b.add_argument("--general", type=Path, required=True)
    b.add_argument("--patchwork", type=Path, required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--serve", type=Path, required=True, help="folder with config.json from training.prepare")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--accounts", type=int, default=14)
    b.add_argument("--accounts-file", type=Path, default=None, help="alias,username lines: shard folders take these aliases")
    b.add_argument("--pw-share", type=float, default=0.2, help="PatchWork share of training tokens")
    b.add_argument("--pw-heldout-rows", type=int, default=1000)
    b.add_argument("--max-len", type=int, default=512)
    b.add_argument("--max-rows", type=int, default=0, help="cap rows per account (rehearsals only; 0 = no cap)")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=build)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
