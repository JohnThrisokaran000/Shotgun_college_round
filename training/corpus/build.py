"""Build the general corpus shards from the downloaded archives.

    python -m training.corpus.build --corpus data/corpus --out data/corpus/general

Phase 1 reads every allowed archive in parallel and writes rows to staging files. A row goes to
one of N shards by a hash of its text, so identical text always lands in the same shard. Whole
packages whose name hashes into the held-out bucket go to a held-out file instead.
Phase 2 removes duplicate text inside each shard, which removes it globally too.
Phase 3 drops held-out rows whose text also occurs in training, then caps the held-out set.
Writes out/shard_NN.jsonl, out/heldout.jsonl, out/build.json, out/attribution.json and out/LICENSES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from training.corpus import extract
from training.corpus.fetch import load_manifest


def digest(text: str) -> int:
    """64-bit content hash. Stable across runs and machines, which the shard assignment depends on."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def is_held_out(key: str, every: int) -> bool:
    return every > 0 and digest(key) % every == 0


def final_row(row: dict) -> dict:
    """The row as training sees it: a FILE header, then the chunk, with no prompt."""
    completion = f"FILE: {row['src']}@{row['version']}/{row['file']}\n{row['body']}"
    return {"prompt": "", "completion": completion, "src": row["src"], "file": row["file"], "lic": row["lic"]}


def process_archive(job: dict) -> dict:
    """Worker: extract one allowed source into rows. Errors are returned, not raised, so one bad archive is recorded and skipped."""
    key = job["key"]
    try:
        if job["ext"] == "dir":
            members = extract.iter_directory(Path(job["path"]))
        else:
            members = extract.iter_archive(Path(job["path"]), job["ext"])
        if job.get("include"):
            prefixes = tuple(job["include"])
            members = ((rel, data) for rel, data in members if rel.startswith(prefixes))
        rows = list(extract.package_rows(key, job["version"], job["licence"], members))
        return {"key": key, "rows": rows, "error": None}
    except Exception as err:  # noqa: BLE001 - recorded per source, and the run continues
        return {"key": key, "rows": [], "error": f"{type(err).__name__}: {err}"}


def _jobs(manifest: dict[str, dict]) -> list[dict]:
    jobs = []
    for key in sorted(manifest):
        entry = manifest[key]
        if not entry.get("allowed"):
            continue
        path = entry.get("path") if entry["ext"] == "dir" else entry.get("file")
        if not path or not Path(path).exists():
            continue
        jobs.append({"key": key, "version": entry["version"], "licence": entry["licence"],
                     "ext": entry["ext"], "path": path, "include": entry.get("include")})
    return jobs


def phase_one(jobs: list[dict], staging: Path, shards: int, heldout_every: int, workers: int) -> dict:
    staging.mkdir(parents=True, exist_ok=True)
    shard_files = [(staging / f"shard_{i:02d}.jsonl").open("w", encoding="utf-8") for i in range(shards)]
    heldout_file = (staging / "heldout.jsonl").open("w", encoding="utf-8")
    report: dict[str, dict] = {}
    try:
        with mp.Pool(workers) as pool:
            for result in pool.imap_unordered(process_archive, jobs, chunksize=1):
                key = result["key"]
                if result["error"]:
                    report[key] = {"error": result["error"]}
                    continue
                held = is_held_out(key, heldout_every)
                chars = 0
                for row in result["rows"]:
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    chars += len(row["body"])
                    if held:
                        heldout_file.write(line)
                    else:
                        shard_files[digest(row["body"]) % shards].write(line)
                report[key] = {"rows": len(result["rows"]), "chars": chars, "heldout": held}
                print(f"  extracted {key:<44} rows {len(result['rows']):>7}  chars {chars / 1e6:8.1f} M", flush=True)
    finally:
        for handle in shard_files:
            handle.close()
        heldout_file.close()
    return report


def phase_two(staging: Path, out: Path, shards: int) -> tuple[np.ndarray, list[int], dict[str, int]]:
    """Deduplicate each shard. Returns the sorted training hashes, rows per shard, and rows per package."""
    seen_all: list[np.ndarray] = []
    rows_per_shard: list[int] = []
    per_package: dict[str, int] = {}
    for i in range(shards):
        seen: set[int] = set()
        kept_hashes: list[int] = []
        with (staging / f"shard_{i:02d}.jsonl").open(encoding="utf-8") as src, \
                (out / f"shard_{i:02d}.jsonl").open("w", encoding="utf-8") as dst:
            for line in src:
                row = json.loads(line)
                h = digest(row["body"])
                if h in seen:
                    continue
                seen.add(h)
                kept_hashes.append(h)
                per_package[row["src"]] = per_package.get(row["src"], 0) + 1
                dst.write(json.dumps(final_row(row), ensure_ascii=False) + "\n")
        rows_per_shard.append(len(kept_hashes))
        seen_all.append(np.fromiter(seen, dtype=np.uint64, count=len(seen)))
        print(f"  shard {i:02d}: {len(kept_hashes)} unique rows", flush=True)
    train_hashes = np.sort(np.concatenate(seen_all)) if seen_all else np.zeros(0, dtype=np.uint64)
    return train_hashes, rows_per_shard, per_package


def phase_three(staging: Path, out: Path, train_hashes: np.ndarray, cap: int) -> tuple[int, int]:
    """Drop held-out rows whose text is in training, dedupe within the held-out set, and cap its size."""
    rows: dict[int, dict] = {}
    leaked = 0
    with (staging / "heldout.jsonl").open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            h = digest(row["body"])
            pos = int(np.searchsorted(train_hashes, np.uint64(h)))
            if pos < len(train_hashes) and train_hashes[pos] == h:
                leaked += 1
                continue
            rows.setdefault(h, row)
    ordered = [final_row(rows[h]) for h in sorted(rows)]
    if cap and len(ordered) > cap:
        step = len(ordered) / cap
        ordered = [ordered[int(i * step)] for i in range(cap)]
    with (out / "heldout.jsonl").open("w", encoding="utf-8") as dst:
        for row in ordered:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(ordered), leaked


def write_attribution(manifest: dict[str, dict], report: dict, per_package: dict, out: Path) -> None:
    rows = []
    for key in sorted(manifest):
        entry = manifest[key]
        if not entry.get("allowed"):
            continue
        info = report.get(key, {})
        rows.append({"key": key, "version": entry["version"], "licence": entry["licence"],
                     "rows_kept": per_package.get(key, 0), "error": info.get("error"),
                     "heldout": info.get("heldout", False)})
    (out / "attribution.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    lines = ["# Third-party sources in the general corpus", "",
             "Each source is distributed under the licence named here. Rows are chunks of the source files,",
             "each with a `FILE:` header. Sources with no kept rows are listed for completeness.", "",
             "| source | version | licence | rows kept | held out |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['key']} | {row['version']} | {row['licence']} | {row['rows_kept']} | "
                     f"{'yes' if row['heldout'] else ''} |")
    (out / "LICENSES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--out", type=Path, default=Path("data/corpus/general"))
    p.add_argument("--shards", type=int, default=14)
    p.add_argument("--heldout-every", type=int, default=40, help="one package in this many is held out (0 = none)")
    p.add_argument("--heldout-rows", type=int, default=4000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--keep-staging", action="store_true")
    args = p.parse_args(argv)

    started = time.time()
    manifest = load_manifest(args.corpus / "manifest.jsonl")
    jobs = _jobs(manifest)
    if not jobs:
        raise SystemExit("no allowed sources with files on disk; run training.corpus.fetch first")
    args.out.mkdir(parents=True, exist_ok=True)
    staging = args.out / "staging"
    print(f"phase 1: extracting {len(jobs)} sources with {args.workers} workers", flush=True)
    report = phase_one(jobs, staging, args.shards, args.heldout_every, args.workers)
    print("phase 2: deduplicating shards", flush=True)
    train_hashes, rows_per_shard, per_package = phase_two(staging, args.out, args.shards)
    print("phase 3: held-out split", flush=True)
    heldout_rows, leaked = phase_three(staging, args.out, train_hashes, args.heldout_rows)
    write_attribution(manifest, report, per_package, args.out)
    summary = {
        "sources": len(jobs), "sources_failed": sorted(k for k, v in report.items() if v.get("error")),
        "shards": args.shards, "rows_per_shard": rows_per_shard, "rows_train": int(sum(rows_per_shard)),
        "rows_heldout": heldout_rows, "heldout_leaked_removed": leaked,
        "heldout_packages": sorted(k for k, v in report.items() if v.get("heldout")),
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
