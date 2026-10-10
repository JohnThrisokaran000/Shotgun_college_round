"""Turn downloaded archives into text chunks.

Selection is conservative on purpose: source-like text only, no minified or generated bundles,
no binaries, no oversized files, and every archive member is checked before it is read. Archive
members are read as data. Links, devices and absolute or ".." paths are dropped.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

from training.corpus import sources

# Source, documentation and shader text. Data files (json, csv, images, fonts) are not included.
EXTENSIONS = frozenset({
    ".py", ".pyi", ".pyx", ".pxd", ".rs", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".md", ".rst",
    ".html", ".htm", ".css", ".scss", ".less", ".glsl", ".vert", ".frag", ".wgsl", ".toml", ".yaml", ".yml",
    ".sh", ".bs", ".adoc",
})
SKIP_NAME_PREFIXES = ("license", "copying", "notice", "authors", "contributors", "changelog.")
SKIP_NAME_SUFFIXES = (".min.js", ".min.css", ".bundle.js", ".umd.js", ".map", ".lock")

MAX_FILE_BYTES = 1_500_000
MAX_LINE_CHARS = 1_500
MAX_MEAN_LINE_CHARS = 160  # minified or generated text has very long lines on average
MAX_PACKAGE_CHARS = 60_000_000
MIN_CHUNK_CHARS = 40


def safe_relative(name: str) -> str | None:
    """Drop the archive's top folder and reject unsafe paths. Returns None when the member must be skipped."""
    if "\\" in name or name.startswith("/"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if len(parts) < 2 or any(p == ".." for p in parts):
        return None
    return "/".join(parts[1:])


def wanted_file(rel: str, data: bytes) -> str | None:
    """Return the decoded text if the file is worth keeping, else None."""
    parts = rel.split("/")
    name = parts[-1]
    lowered = name.lower()
    if lowered.startswith(SKIP_NAME_PREFIXES) or lowered.endswith(SKIP_NAME_SUFFIXES):
        return None
    if os.path.splitext(lowered)[1] not in EXTENSIONS:
        return None
    if sources.is_skipped_path(tuple(parts[:-1])):
        return None
    if len(data) > MAX_FILE_BYTES or b"\x00" in data[:4096]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    if max(len(line) for line in lines) > MAX_LINE_CHARS:
        return None
    if len(text) / len(lines) > MAX_MEAN_LINE_CHARS:
        return None
    return text


def iter_archive(path: Path, ext: str) -> Iterator[tuple[str, bytes]]:
    """Yield (relative path, bytes) for regular files, reading each member as data only."""
    if ext in ("tgz", "crate"):
        with tarfile.open(path, mode="r|gz") as tar:
            for member in tar:
                if not member.isreg() or member.size > MAX_FILE_BYTES:
                    continue
                rel = safe_relative(member.name)
                if rel is None:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                yield rel, handle.read(MAX_FILE_BYTES + 1)
    elif ext == "zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or info.file_size > MAX_FILE_BYTES:
                    continue
                rel = safe_relative(info.filename)
                if rel is None:
                    continue
                yield rel, archive.read(info)
    else:
        raise ValueError(f"unknown archive type: {ext}")


def iter_directory(root: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (relative path, bytes) for files under a local directory, skipping test and vendored folders."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not sources.is_skipped_path((d,)) and d != "test"]
        for filename in filenames:
            full = Path(dirpath) / filename
            try:
                if full.is_symlink() or full.stat().st_size > MAX_FILE_BYTES:
                    continue
                rel = full.relative_to(root).as_posix()
                yield rel, full.read_bytes()
            except OSError:
                continue


def chunk_text(text: str, max_chars: int = 1100) -> list[str]:
    """Split text at line boundaries into pieces of at most max_chars. Joining the pieces gives the input back."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:
            if current:
                chunks.append("".join(current))
                current, size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if current and size + len(line) > max_chars:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


def package_rows(key: str, version: str, licence: str, members: Iterator[tuple[str, bytes]]) -> Iterator[dict]:
    """Yield one staging row per chunk. The body is the chunk text alone, so dedupe compares content, not
    paths. The FILE header is added when the final shard is written. Stops once the package is at its cap."""
    written = 0
    for rel, data in members:
        text = wanted_file(rel, data)
        if text is None:
            continue
        for chunk in chunk_text(text):
            written += len(chunk)
            yield {"body": chunk, "src": key, "version": version, "file": rel, "lic": licence}
            if written > MAX_PACKAGE_CHARS:
                return
