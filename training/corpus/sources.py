"""Where the general corpus comes from, and which licences are allowed.

Each source is resolved against its registry before anything is downloaded. The licence comes
from the registry's own metadata, not from this file. A package whose licence is missing, is
unrecognised, or is not on the allow-list is skipped, and the reason goes into the manifest.

Copyleft (GPL, LGPL, AGPL, MPL) and share-alike (CC-BY-SA, so MDN) sources are left out on
purpose. Those licences put obligations on the model weights, and this project has not settled
them.
"""

from __future__ import annotations

import re
from pathlib import Path

# SPDX identifiers that allow training and redistribution with attribution.
PERMISSIVE = frozenset({
    "0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSL-1.0", "BlueOak-1.0.0", "CC0-1.0",
    "CNRI-Python", "HPND", "ISC", "MIT", "MIT-0", "MIT-CMU", "PSF-2.0", "Python-2.0", "Unlicense",
    "W3C", "W3C-20150513", "Zlib",
})

# Free-text licence names that older PyPI projects use, mapped to SPDX identifiers. Keys are lower case.
FREE_TEXT = {
    "mit": "MIT", "mit license": "MIT",
    "apache 2.0": "Apache-2.0", "apache 2.0 license": "Apache-2.0", "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0", "apache software license": "Apache-2.0",
    "bsd": "BSD-3-Clause", "bsd license": "BSD-3-Clause", "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause", "2-clause bsd license": "BSD-2-Clause",
    "simplified bsd license": "BSD-2-Clause",
    "isc": "ISC", "isc license": "ISC", "isc license (iscl)": "ISC",
    "psf": "PSF-2.0", "python software foundation license": "PSF-2.0",
    "zlib": "Zlib", "unlicense": "Unlicense", "cc0": "CC0-1.0",
}

# Trove classifier names (PyPI) mapped to SPDX identifiers.
PYPI_CLASSIFIERS = {
    "MIT License": "MIT",
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
}

# npm packages, grouped by what they teach. A name that does not resolve is recorded and skipped.
NPM = (
    # 3D, WebGL, WebGPU and other graphics
    "three", "@types/three", "@webgpu/types", "three-mesh-bvh", "three-stdlib", "@react-three/fiber",
    "@react-three/drei", "regl", "gl-matrix", "twgl.js", "ogl", "wgpu-matrix", "@babylonjs/core",
    "@babylonjs/gui", "@babylonjs/loaders", "cannon-es", "matter-js", "@tweenjs/tween.js", "lil-gui",
    "stats.js", "pixi.js", "konva", "fabric", "paper", "d3", "d3-geo", "chart.js", "echarts",
    "maplibre-gl", "leaflet", "deck.gl", "@luma.gl/core", "tone", "howler",
    # Web platform typings and specification data (DOM, WebGL, CSS and WebIDL names)
    "typescript", "@types/node", "@types/react", "@types/react-dom", "@types/express", "@types/lodash",
    "@types/jest", "@types/webxr", "@webref/css", "@webref/idl",
    # Frameworks and UI
    "react", "react-dom", "vue", "svelte", "solid-js", "preact", "lit", "@angular/core", "next", "nuxt",
    "astro", "react-router", "@tanstack/react-query", "zustand", "redux", "@reduxjs/toolkit", "jotai",
    "immer", "mobx", "rxjs", "xstate", "alpinejs", "htmx.org", "jquery", "bootstrap", "tailwindcss",
    "@mui/material", "antd", "styled-components", "@emotion/react", "sass", "less", "postcss",
    "autoprefixer", "normalize.css", "animate.css", "swiper", "dompurify", "marked", "markdown-it",
    "highlight.js", "prismjs", "codemirror", "monaco-editor",
    # Build, test and tooling
    "vite", "esbuild", "rollup", "webpack", "parcel", "eslint", "prettier", "jest", "vitest", "mocha",
    "chai", "playwright", "playwright-core", "puppeteer", "cypress",
    # Server, Node and utilities
    "express", "fastify", "koa", "hono", "socket.io", "ws", "graphql", "drizzle-orm", "knex", "sequelize",
    "mongoose", "pg", "mysql2", "redis", "ioredis", "jsonwebtoken", "passport", "dotenv", "commander",
    "yargs", "inquirer", "chalk", "glob", "minimatch", "semver", "uuid", "nanoid", "zod", "yup",
    "date-fns", "dayjs", "axios", "ky", "node-fetch", "lodash", "underscore", "ramda", "cheerio", "jsdom",
    "multer", "cors", "helmet", "morgan", "winston", "pino", "openai", "@anthropic-ai/sdk", "langchain", "ai",
)

# PyPI projects. Source distributions are preferred, because they carry the source and docs.
PYPI = (
    "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "sympy", "pillow", "requests", "urllib3",
    "httpx", "aiohttp", "fastapi", "starlette", "pydantic", "sqlalchemy", "alembic", "django", "flask",
    "werkzeug", "jinja2", "click", "rich", "typer", "pytest", "attrs", "cryptography", "boto3", "botocore",
    "tqdm", "pyyaml", "tomli", "packaging", "networkx", "beautifulsoup4", "lxml", "selenium", "playwright",
    "scrapy", "celery", "redis", "pymongo", "asyncpg", "uvicorn", "gunicorn", "marshmallow", "pluggy",
    "black", "mypy", "flake8", "isort", "sphinx", "docutils", "pygments", "markdown", "mkdocs", "numba",
    "cython", "jax", "transformers", "tokenizers", "datasets", "huggingface-hub", "accelerate",
    "safetensors", "pyglet", "moderngl", "pyopengl", "vispy", "trimesh", "pyvista", "plotly", "bokeh",
    "altair", "seaborn", "statsmodels", "shapely", "geopandas", "pyproj", "pyarrow", "polars", "openai",
    "anthropic", "tenacity", "orjson", "msgpack", "regex", "python-dateutil", "decorator", "more-itertools",
    "toolz", "joblib", "dask", "xarray", "h5py", "imageio", "scikit-image", "sqlparse", "psutil", "fabric",
    "invoke", "docker", "pyjwt", "passlib", "bcrypt", "itsdangerous", "email-validator", "pydantic-core",
    "uvloop", "anyio", "trio", "h11", "httpcore", "websockets", "protobuf", "keras", "pyjwt", "pyinstrument",
)

# crates.io crates. Each version's licence is checked against the API before download.
CRATES = (
    "serde", "serde_json", "serde_yaml", "toml", "tokio", "futures", "async-trait", "axum", "hyper",
    "tower", "tower-http", "reqwest", "http", "url", "clap", "anyhow", "thiserror", "tracing",
    "tracing-subscriber", "log", "env_logger", "rayon", "crossbeam", "parking_lot", "dashmap", "regex",
    "once_cell", "chrono", "time", "uuid", "rand", "bytes", "memchr", "itertools", "indexmap", "hashbrown",
    "smallvec", "bitflags", "byteorder", "nom", "pest", "syn", "quote", "proc-macro2", "bincode", "flate2",
    "zstd", "sha2", "base64", "hex", "rustls", "sqlx", "rusqlite", "redis", "image", "wgpu", "winit",
    "glam", "nalgebra", "ndarray", "candle-core", "burn", "bevy", "macroquad", "three-d", "ash", "glow",
    "egui", "iced", "ratatui", "crossterm", "pulldown-cmark", "handlebars", "tera", "askama", "actix-web",
    "warp", "tonic", "prost", "git2", "gix", "wasm-bindgen", "web-sys", "js-sys", "yew", "leptos", "scraper",
    "html5ever", "quick-xml", "csv", "rust_decimal", "num-bigint", "num-traits", "memmap2", "tempfile",
    "walkdir", "ignore", "globset", "notify", "which", "console", "indicatif", "dialoguer", "pico-args",
    "criterion", "proptest", "mockall", "rstest", "insta", "pyo3", "cxx", "cc", "libc", "cfg-if",
    "lazy_static", "scopeguard", "pin-project-lite", "futures-util", "async-stream", "tokio-util",
    "tokio-stream", "mio", "socket2", "signal-hook", "sysinfo", "nix", "libloading", "jsonwebtoken",
    "argon2", "rand_chacha", "ring", "hmac", "sha1", "md-5", "aes", "argh", "structopt", "derive_more",
    "strum", "enum-map", "bytemuck", "half", "approx", "statrs", "petgraph", "rustyline", "tui",
)

# GitHub archives: (owner/repo, branch, override, include). The licence comes from the GitHub licence
# API. When that API reports NOASSERTION (one LICENSE file that covers several licences), the
# override applies instead: (SPDX, text that must appear in the top-level LICENSE after download).
# The archive is discarded if the text is not there. `include` keeps only these top-level folders,
# because the rest of the repository carries other licences (Node's deps/, CPython's C modules).
GITHUB = (
    ("mrdoob/three.js", "dev", None, None),
    ("BabylonJS/Babylon.js", "master", None, None),
    ("nodejs/node", "main", ("MIT", "Permission is hereby granted, free of charge"), ("lib/", "doc/")),
    ("python/cpython", "main", ("PSF-2.0", "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2"), ("Lib/", "Doc/")),
    ("microsoft/TypeScript", "main", None, ("src/",)),
    ("microsoft/vscode", "main", None, ("src/",)),
    ("facebook/react", "main", None, ("packages/",)),
    ("vuejs/core", "main", None, ("packages/",)),
    ("sveltejs/svelte", "main", None, ("packages/",)),
)

def _extra(name: str) -> tuple[str, ...]:
    """Names from a committed list written by training.corpus.discover (one per line, # for comments)."""
    path = Path(__file__).with_name(name)
    if not path.exists():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#"))


NPM = tuple(dict.fromkeys(NPM + _extra("npm_extra.txt")))
PYPI = tuple(dict.fromkeys(PYPI + _extra("pypi_extra.txt")))
CRATES = tuple(dict.fromkeys(CRATES + _extra("crates_extra.txt")))

# The Python standard library on this machine. Its licence is the PSF licence.
STDLIB_LICENCE = "PSF-2.0"

# Test and vendored directories are dropped from the stdlib and from packages.
SKIP_DIR_PARTS = frozenset({
    "node_modules", ".git", "__pycache__", "site-packages", "vendor", "third_party", "third-party",
    "bower_components", "dist", "bundles", "coverage", "fixtures", "__snapshots__", "testdata",
})

_SPDX_TERM = re.compile(r"^[A-Za-z0-9.+-]+$")


def _terms(expr: str) -> list[list[str]]:
    """Split an SPDX-style expression into alternatives (OR), each a list of terms (AND)."""
    text = expr.replace("(", " ").replace(")", " ").replace("/", " OR ")
    alternatives = []
    for alt in re.split(r"\s+OR\s+", text, flags=re.IGNORECASE):
        terms = [t for t in re.split(r"\s+AND\s+", alt.strip(), flags=re.IGNORECASE) if t.strip()]
        if terms:
            alternatives.append([t.strip() for t in terms])
    return alternatives


def allowed_licence(expr) -> tuple[bool, str]:
    """Return (allowed, normalised). OR takes the permissive choice. AND needs every term allowed.

    A WITH exception, an unknown identifier, or free text returns (False, text).
    """
    if isinstance(expr, dict):
        expr = expr.get("type")
    if not isinstance(expr, str) or not expr.strip():
        return False, ""
    text = " ".join(expr.split())
    text = FREE_TEXT.get(text.lower().strip("() "), text)  # "Apache 2.0", "BSD" and similar free-text names
    alternatives = _terms(text)
    if not alternatives:
        return False, text
    for terms in alternatives:
        if all(_SPDX_TERM.match(t) and t in PERMISSIVE for t in terms):
            return True, " OR ".join(" AND ".join(terms) for terms in alternatives)
    return False, text


def licence_from_pypi(info: dict) -> tuple[bool, str]:
    """PyPI has three licence fields. Newer projects set license_expression (PEP 639, an SPDX
    expression) and nothing else. Older ones set license, often free text, and classifiers. When
    license_expression is present it decides, because it is the project's own statement."""
    expression = (info.get("license_expression") or "").strip()
    if expression:
        return allowed_licence(expression)
    field = (info.get("license") or "").strip()
    if field and len(field) <= 60 and "\n" not in field:
        mapped = FREE_TEXT.get(field.lower())
        ok, norm = allowed_licence(mapped or field)
        if ok:
            return ok, norm
    for classifier in info.get("classifiers") or []:
        if not classifier.startswith("License :: OSI Approved :: "):
            continue
        name = classifier[len("License :: OSI Approved :: "):]
        if name in PYPI_CLASSIFIERS:
            return True, PYPI_CLASSIFIERS[name]
    return False, field


def is_skipped_path(parts: tuple[str, ...]) -> bool:
    return any(p in SKIP_DIR_PARTS for p in parts)
