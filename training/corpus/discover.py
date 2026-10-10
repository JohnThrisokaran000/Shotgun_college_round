"""Find more packages to widen the corpus. Writes the lists the fetcher reads, so the build stays reproducible.

    python -m training.corpus.discover --npm 1500 --pypi 400 --crates 400

npm: packages returned by registry search for a set of topics, in the order the registry ranks them.
PyPI: the most-downloaded projects (a public ranking, updated monthly).
crates.io: the most-downloaded crates.

Nothing here decides licences. The fetcher checks every package against its registry metadata.
Existing lists in sources.py are skipped, so the extra lists only add packages.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from training.corpus import sources

USER_AGENT = "patchwork-corpus/0.1 (training data builder)"
HERE = Path(__file__).resolve().parent

TOPICS = [
    "webgl", "three", "3d", "shader", "glsl", "webgpu", "canvas", "animation", "physics", "game", "audio", "chart",
    "d3", "map", "geo", "svg", "pdf", "excel", "csv", "image", "video", "upload", "form", "validation", "router",
    "state", "store", "react", "vue", "angular", "svelte", "web-components", "css", "sass", "tailwind", "ui",
    "components", "design-system", "icons", "typescript", "eslint", "babel", "webpack", "vite", "bundler",
    "testing", "jest", "mocha", "playwright", "puppeteer", "express", "koa", "fastify", "http", "websocket",
    "graphql", "rest", "api", "sql", "orm", "database", "mongodb", "redis", "queue", "cache", "auth", "jwt",
    "crypto", "date", "time", "string", "array", "util", "cli", "terminal", "logger", "parser", "markdown",
    "html", "dom", "template", "i18n", "accessibility", "pwa", "electron", "react-native", "stream", "buffer",
    "file", "zip", "encoding", "regex", "diff", "git", "docker", "aws", "cloud", "serverless", "openai", "ai",
    "vector", "math", "matrix", "statistics", "color", "font", "markdown-it", "yaml", "json", "schema",
    "scheduler", "websocket-server", "email", "payment", "scraping", "crawler", "chatbot", "discord",
]
PYPI_KEEP = 400


def _get(url: str, accept: str | None = None):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def npm_candidates(limit: int) -> list[str]:
    seen: dict[str, None] = {}
    for topic in TOPICS:
        if len(seen) >= limit * 2:
            break
        url = "https://registry.npmjs.org/-/v1/search?" + urllib.parse.urlencode({"text": topic, "size": 100})
        try:
            results = _get(url).get("objects", [])
        except Exception as err:  # noqa: BLE001 - a missing topic should not stop discovery
            print(f"npm search '{topic}' failed: {type(err).__name__}")
            continue
        for item in results:
            name = item.get("package", {}).get("name")
            if name:
                seen.setdefault(name, None)
        time.sleep(0.3)
    existing = set(sources.NPM)
    return [n for n in seen if n not in existing][:limit]


def pypi_candidates(limit: int) -> list[str]:
    url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json"
    rows = _get(url).get("rows", [])
    existing = set(sources.PYPI)
    names = [r["project"] for r in rows if r.get("project") and r["project"] not in existing]
    return names[:limit]


def crates_candidates(limit: int) -> list[str]:
    names: dict[str, None] = {}
    page = 1
    while len(names) < limit and page <= 20:
        url = "https://crates.io/api/v1/crates?" + urllib.parse.urlencode({"page": page, "per_page": 100, "sort": "downloads"})
        for crate in _get(url).get("crates", []):
            names.setdefault(crate["id"], None)
        page += 1
        time.sleep(1.2)
    existing = set(sources.CRATES)
    return [n for n in names if n not in existing][:limit]


def write_list(path: Path, names: list[str], header: str) -> None:
    path.write_text(header + "\n" + "\n".join(names) + "\n", encoding="utf-8")
    print(f"{path.name}: {len(names)} names")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npm", type=int, default=1500)
    p.add_argument("--pypi", type=int, default=PYPI_KEEP)
    p.add_argument("--crates", type=int, default=400)
    args = p.parse_args(argv)
    write_list(HERE / "npm_extra.txt", npm_candidates(args.npm),
               "# npm packages found by registry search (training.corpus.discover). Licences are checked at fetch time.")
    write_list(HERE / "pypi_extra.txt", pypi_candidates(args.pypi),
               "# PyPI projects from the public top-downloads ranking. Licences are checked at fetch time.")
    write_list(HERE / "crates_extra.txt", crates_candidates(args.crates),
               "# crates.io crates by downloads. Licences are checked at fetch time.")


if __name__ == "__main__":
    main()
