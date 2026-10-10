"""Download the general corpus archives, with licence checks and a resumable manifest.

    python -m training.corpus.fetch --out data/corpus --dry-run     # resolve and check licences only
    python -m training.corpus.fetch --out data/corpus               # download what is allowed

Writes data/corpus/raw/<ecosystem>/<file> and appends one line per source to
data/corpus/manifest.jsonl. Every source gets a line, allowed or not, with the reason, so skips
can be audited. Re-running skips archives already downloaded and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from training.corpus import sources

USER_AGENT = "patchwork-corpus/0.1 (training data builder)"
MAX_ARCHIVE_BYTES = 450 * 1024 * 1024
MAX_GITHUB_ARCHIVE_BYTES = 1200 * 1024 * 1024
TIMEOUT_S = 90


class Skip(Exception):
    """A source that cannot be used. The message is the reason recorded in the manifest."""


@dataclass
class Archive:
    eco: str
    name: str
    version: str
    licence: str
    url: str
    ext: str
    size_hint: int | None = None
    verify: str | None = None  # text that must appear in the archive's top-level LICENSE after download
    include: list[str] | None = None  # keep only members under these top-level folders (GitHub archives)

    @property
    def key(self) -> str:
        return f"{self.eco}:{self.name}"


class Http:
    """HTTP client with retries on transient errors and a reserved time slot per call, so calls
    from several threads stay at least `min_interval_s` apart."""

    def __init__(self, min_interval_s: float = 0.1):
        self._min = min_interval_s
        self._lock = threading.Lock()
        self._next = 0.0

    def _pace(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._min
        if start > now:
            time.sleep(start - now)

    def open(self, url: str, accept: str | None = None):
        headers = {"User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        delay = 1.0
        for attempt in range(6):
            self._pace()
            try:
                return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S)
            except urllib.error.HTTPError as err:
                if not (err.code >= 500 or err.code == 429) or attempt == 5:
                    raise
                if err.code == 429:  # rate limited: back off much longer than for server errors
                    time.sleep(15 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    def json(self, url: str):
        with self.open(url, accept="application/json") as response:
            return json.load(response)


def _registry_errors_are_skips(fn: Callable) -> Callable:
    """A 404 means the name is not there. Other failures are recorded too, so one bad name cannot stop a run."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except urllib.error.HTTPError as err:
            raise Skip(f"not found (HTTP 404)" if err.code == 404 else f"registry HTTP {err.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as err:
            raise Skip(f"registry unreachable: {type(err).__name__}") from None
    wrapper.__name__ = fn.__name__
    return wrapper


@_registry_errors_are_skips
def resolve_npm(http: Http, name: str) -> Archive:
    data = http.json(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}/latest")
    dist = data.get("dist") or {}
    if not dist.get("tarball"):
        raise Skip("no tarball")
    ok, licence = sources.allowed_licence(data.get("license"))
    if not ok:
        raise Skip(f"licence not allowed: {licence or 'missing'}")
    return Archive("npm", name, str(data.get("version")), licence, dist["tarball"], "tgz", dist.get("unpackedSize"))


@_registry_errors_are_skips
def resolve_pypi(http: Http, name: str) -> Archive:
    data = http.json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    info = data.get("info") or {}
    sdists = [u for u in data.get("urls") or [] if u.get("packagetype") == "sdist"]
    if not sdists:
        raise Skip("no source distribution")
    release = sdists[0]
    filename = release.get("filename", "")
    if filename.endswith((".tar.gz", ".tgz")):
        ext = "tgz"
    elif filename.endswith(".zip"):
        ext = "zip"
    else:
        raise Skip(f"unsupported sdist format: {filename}")
    ok, licence = sources.licence_from_pypi(info)
    if not ok:
        raise Skip(f"licence not allowed: {licence or 'missing'}")
    return Archive("pypi", name, str(info.get("version")), licence, release["url"], ext, release.get("size"))


@_registry_errors_are_skips
def resolve_crate(http: Http, name: str) -> Archive:
    quoted = urllib.parse.quote(name)
    crate = http.json(f"https://crates.io/api/v1/crates/{quoted}")["crate"]
    version = crate.get("max_stable_version") or crate.get("max_version")
    if not version:
        raise Skip("no published version")
    detail = http.json(f"https://crates.io/api/v1/crates/{quoted}/{urllib.parse.quote(version)}")["version"]
    ok, licence = sources.allowed_licence(detail.get("license"))
    if not ok:
        raise Skip(f"licence not allowed: {licence or 'missing'}")
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    return Archive("crates", name, version, licence, url, "crate", detail.get("crate_size"))


@_registry_errors_are_skips
def resolve_github(http: Http, repo: str, branch: str, override: tuple[str, str] | None = None,
                   include: tuple[str, ...] | None = None) -> Archive:
    data = http.json(f"https://api.github.com/repos/{repo}/license")
    spdx = (data.get("license") or {}).get("spdx_id")
    verify = None
    if override is not None and spdx in (None, "NOASSERTION", "OTHER"):
        spdx, verify = override
    ok, licence = sources.allowed_licence(spdx)
    if not ok:
        raise Skip(f"licence not allowed: {licence or 'missing'}")
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    return Archive("github", repo, branch, licence, url, "zip", None, verify, list(include) if include else None)


def _licence_text_matches(path: Path, fingerprint: str) -> bool:
    """True when the top-level LICENSE of the archive contains the fingerprint text."""
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) == 2 and parts[1] == "LICENSE":
                if info.file_size > 2_000_000:
                    return False
                return fingerprint in archive.read(info).decode("utf-8", errors="replace")
    return False


def _file_stem(name: str, version: str) -> str:
    return f"{name}-{version}".replace("/", "__").replace("@", "").replace(" ", "_")[:180]


def download(http: Http, arc: Archive, raw_dir: Path) -> dict:
    """Stream the archive to disk and hash it as it goes. Anything over the size cap is refused."""
    folder = raw_dir / arc.eco
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{_file_stem(arc.name, arc.version)}.{arc.ext}"
    tmp = dest.with_name(dest.name + ".part")
    digest, size = hashlib.sha256(), 0
    cap = MAX_GITHUB_ARCHIVE_BYTES if arc.eco == "github" else MAX_ARCHIVE_BYTES
    try:
        with http.open(arc.url) as response, tmp.open("wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if size > cap:
                    raise Skip("archive over the size cap")
                digest.update(block)
                out.write(block)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"file": str(dest), "bytes": size, "sha256": digest.hexdigest()}


def _record(arc: Archive, allowed: bool, reason: str, extra: dict | None = None) -> dict:
    entry = {**asdict(arc), "key": arc.key, "allowed": allowed, "reason": reason, "recorded_at": int(time.time())}
    entry.update(extra or {})
    return entry


def load_manifest(path: Path) -> dict[str, dict]:
    """Latest entry per source key. A later line replaces an earlier one, so re-runs repair failures."""
    entries: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries[entry["key"]] = entry
    return entries


def _already_downloaded(previous: dict | None) -> bool:
    if not previous or not previous.get("allowed") or not previous.get("file"):
        return False
    path = Path(previous["file"])
    return path.exists() and path.stat().st_size == previous.get("bytes")


def _fetch_one(http: Http, arc: Archive, raw_dir: Path, dry_run: bool, previous: dict | None) -> dict:
    if _already_downloaded(previous):
        # Keep the file. Take the current licence and filters, which may have changed since it was fetched.
        return _record(arc, True, "already downloaded",
                       {"file": previous["file"], "bytes": previous["bytes"], "sha256": previous["sha256"]})
    if dry_run:
        return _record(arc, True, "dry run", {"bytes": arc.size_hint or 0})
    try:
        extra = download(http, arc, raw_dir)
        if arc.verify and not _licence_text_matches(Path(extra["file"]), arc.verify):
            Path(extra["file"]).unlink(missing_ok=True)
            return _record(arc, False, "LICENSE text did not match the override")
        return _record(arc, True, "downloaded", extra)
    except Skip as skip:
        return _record(arc, False, str(skip))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        return _record(arc, False, f"download failed: {type(err).__name__}")


def _jobs(only: set[str] | None) -> list[tuple[str, str, Callable[[], Archive]]]:
    # One client per registry, so each one is paced to its own limits. crates.io asks for about one request a second.
    npm, pypi, crates, github = Http(0.05), Http(0.2), Http(1.2), Http(1.0)
    jobs: list[tuple[str, str, Callable[[], Archive]]] = []
    if only is None or "npm" in only:
        jobs += [("npm", n, lambda n=n: resolve_npm(npm, n)) for n in dict.fromkeys(sources.NPM)]
    if only is None or "pypi" in only:
        jobs += [("pypi", n, lambda n=n: resolve_pypi(pypi, n)) for n in dict.fromkeys(sources.PYPI)]
    if only is None or "crates" in only:
        jobs += [("crates", n, lambda n=n: resolve_crate(crates, n)) for n in dict.fromkeys(sources.CRATES)]
    if only is None or "github" in only:
        jobs += [("github", repo, lambda r=repo, b=branch, o=override, i=include: resolve_github(github, r, b, o, i))
                 for repo, branch, override, include in sources.GITHUB]
    return jobs


def _resolve(eco: str, name: str, fn: Callable[[], Archive]):
    """Return (archive, None), or (None, skip record) when the source cannot be used."""
    try:
        return fn(), None
    except Skip as skip:
        return None, _record(Archive(eco, name, "", "", "", ""), False, str(skip))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/corpus"))
    p.add_argument("--dry-run", action="store_true", help="resolve and check licences; download nothing")
    p.add_argument("--only", nargs="*", choices=["npm", "pypi", "crates", "github"], default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-stdlib", action="store_true", help="skip the local Python standard library")
    args = p.parse_args(argv)

    raw_dir = args.out / "raw"
    manifest_path = args.out / "manifest.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    previous = load_manifest(manifest_path)
    only = set(args.only) if args.only else None
    data_http = Http(min_interval_s=0.0)
    counts: Counter = Counter()
    volume = 0
    records: list[dict] = []

    def emit(entry: dict) -> None:
        records.append(entry)
        counts["allowed" if entry["allowed"] else "skipped"] += 1
        if not entry["allowed"]:
            counts["skip: " + entry["reason"].split(":")[0]] += 1
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        status = "ok" if entry["allowed"] else "SKIP"
        print(f"{entry['key']:<42} {status:<5} {entry['reason'][:80]}", flush=True)

    def resolve_and_fetch(eco: str, name: str, fn) -> dict:
        # Resolution and download both run in the pool, so large archives download in parallel.
        arc, skipped = _resolve(eco, name, fn)
        if skipped is not None:
            return skipped
        return _fetch_one(data_http, arc, raw_dir, args.dry_run, previous.get(arc.key))

    jobs = _jobs(only)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(resolve_and_fetch, eco, name, fn) for eco, name, fn in jobs]
        for future in as_completed(futures):
            entry = future.result()
            volume += (entry.get("bytes") or 0) if entry["allowed"] else 0
            emit(entry)

    if not args.no_stdlib and only is None:
        stdlib = Path(sysconfig.get_paths()["stdlib"])
        emit(_record(Archive("local", "python-stdlib", sys.version.split()[0], sources.STDLIB_LICENCE,
                             str(stdlib), "dir"), True, "local directory", {"path": str(stdlib)}))

    print("\nsummary:", dict(sorted(counts.items())))
    print(f"volume: {volume / 1e6:.0f} MB {'(estimate, dry run)' if args.dry_run else 'downloaded'}")


if __name__ == "__main__":
    main()
