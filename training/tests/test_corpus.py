"""Corpus builder: licence rules, safe archive reading, text filters, chunking, and the split."""

import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from training.corpus import build, extract, fetch, sources


class LicenceTests(unittest.TestCase):
    def test_permissive_identifiers_are_allowed(self):
        for expr in ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC", "CC0-1.0", "PSF-2.0"]:
            with self.subTest(expr=expr):
                self.assertTrue(sources.allowed_licence(expr)[0])

    def test_or_takes_the_permissive_choice(self):
        self.assertTrue(sources.allowed_licence("(MIT OR Apache-2.0)")[0])
        self.assertTrue(sources.allowed_licence("MIT/Apache-2.0")[0])
        self.assertTrue(sources.allowed_licence("GPL-3.0 OR MIT")[0])

    def test_and_needs_every_term(self):
        self.assertTrue(sources.allowed_licence("BSD-3-Clause AND 0BSD AND MIT")[0])
        self.assertFalse(sources.allowed_licence("MIT AND GPL-2.0")[0])
        self.assertFalse(sources.allowed_licence("MPL-2.0 AND MIT")[0])

    def test_copyleft_exceptions_and_unknowns_are_refused(self):
        for expr in ["GPL-3.0", "LGPL-2.1", "AGPL-3.0", "CC-BY-SA-4.0", "GPL-2.0 WITH Classpath-exception-2.0",
                     "SEE LICENSE IN LICENSE.md", "NOASSERTION", "", None, "Proprietary"]:
            with self.subTest(expr=expr):
                self.assertFalse(sources.allowed_licence(expr)[0])

    def test_npm_legacy_object_form(self):
        self.assertTrue(sources.allowed_licence({"type": "MIT", "url": "x"})[0])

    def test_pypi_expression_is_authoritative(self):
        info = {"license_expression": "GPL-3.0", "classifiers": ["License :: OSI Approved :: MIT License"]}
        self.assertFalse(sources.licence_from_pypi(info)[0])

    def test_pypi_free_text_and_classifiers(self):
        self.assertTrue(sources.licence_from_pypi({"license": "Apache 2.0 License"})[0])
        self.assertTrue(sources.licence_from_pypi({"classifiers": ["License :: OSI Approved :: BSD License"]})[0])
        self.assertFalse(sources.licence_from_pypi({})[0])
        self.assertFalse(sources.licence_from_pypi({"license": "Copyright (c) 2020\nAll rights reserved."})[0])


class SafePathTests(unittest.TestCase):
    def test_top_folder_is_dropped(self):
        self.assertEqual(extract.safe_relative("pkg-1.0/src/a.js"), "src/a.js")

    def test_unsafe_paths_are_refused(self):
        for name in ["pkg/../../etc/passwd", "../evil", "/abs/x.js", "pkg\\..\\evil.js", "toplevel-only.js"]:
            with self.subTest(name=name):
                self.assertIsNone(extract.safe_relative(name))

    def test_tar_with_link_and_traversal_yields_only_the_regular_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pkg.tgz"
            with tarfile.open(path, "w:gz") as tar:
                good = b"def f():\n    return 1\n"
                info = tarfile.TarInfo("pkg-1.0/lib/good.py")
                info.size = len(good)
                tar.addfile(info, io.BytesIO(good))
                link = tarfile.TarInfo("pkg-1.0/lib/link.py")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tar.addfile(link)
                evil = tarfile.TarInfo("pkg-1.0/../../evil.py")
                evil.size = 4
                tar.addfile(evil, io.BytesIO(b"evil"))
            names = [rel for rel, _ in extract.iter_archive(path, "tgz")]
            self.assertEqual(names, ["lib/good.py"])


class TextFilterTests(unittest.TestCase):
    def test_source_text_is_kept(self):
        self.assertIsNotNone(extract.wanted_file("src/a.js", b"const x = 1;\nconst y = 2;\n"))

    def test_minified_and_generated_text_is_dropped(self):
        self.assertIsNone(extract.wanted_file("dist/a.js", b"x" * 5000))
        self.assertIsNone(extract.wanted_file("src/huge.js", b"var a=1;" * 2000))
        self.assertIsNone(extract.wanted_file("src/a.min.js", b"var a=1;\n"))

    def test_binary_invalid_utf8_licence_and_locked_files_are_dropped(self):
        self.assertIsNone(extract.wanted_file("src/a.js", b"abc\x00def\n"))
        self.assertIsNone(extract.wanted_file("src/a.js", b"\xff\xfe bad\n"))
        self.assertIsNone(extract.wanted_file("LICENSE.md", b"MIT License\n"))
        self.assertIsNone(extract.wanted_file("src/data.json", b"{}\n"))
        self.assertIsNone(extract.wanted_file("node_modules/x/a.js", b"ok\n"))
        self.assertIsNone(extract.wanted_file("Cargo.lock", b"ok\n"))

    def test_chunks_join_back_to_the_input(self):
        # Every line is long enough that no chunk falls under MIN_CHUNK_CHARS, so nothing is dropped.
        text = "".join(f"line {i} " + "x" * (60 + i % 37) + "\n" for i in range(900)) + "y" * 3000 + "\n"
        chunks = extract.chunk_text(text, max_chars=1100)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c) <= 1100 for c in chunks))
        self.assertGreater(len(chunks), 3)

    def test_package_rows_hold_the_body_only_and_stop_at_the_cap(self):
        members = [("src/a.js", b"const a = 1;\n" * 20)]
        rows = list(extract.package_rows("npm:demo", "1.2.3", "MIT", iter(members)))
        self.assertTrue(rows)
        self.assertTrue(rows[0]["body"].startswith("const a = 1;"))
        self.assertNotIn("FILE:", rows[0]["body"])
        self.assertEqual((rows[0]["src"], rows[0]["version"], rows[0]["file"]), ("npm:demo", "1.2.3", "src/a.js"))
        original = extract.MAX_PACKAGE_CHARS
        try:
            extract.MAX_PACKAGE_CHARS = 100
            members = [(f"src/f{i}.js", b"q" * 99 + b"\n") for i in range(50)]
            capped = list(extract.package_rows("npm:demo", "1.2.3", "MIT", iter(members)))
            self.assertGreater(len(capped), 0)
            self.assertLess(len(capped), 50)
        finally:
            extract.MAX_PACKAGE_CHARS = original


class SplitTests(unittest.TestCase):
    def test_digest_is_the_documented_blake2b_value(self):
        # Shard assignment depends on this exact function, so pin its definition.
        import hashlib
        expected = int.from_bytes(hashlib.blake2b(b"abc", digest_size=8).digest(), "little")
        self.assertEqual(build.digest("abc"), expected)

    def test_held_out_is_deterministic_and_roughly_one_in_every(self):
        keys = [f"npm:pkg{i}" for i in range(4000)]
        held = [build.is_held_out(k, 40) for k in keys]
        self.assertEqual(held, [build.is_held_out(k, 40) for k in keys])
        self.assertTrue(0.01 < sum(held) / len(keys) < 0.05)
        self.assertFalse(any(build.is_held_out(k, 0) for k in keys))

    def test_end_to_end_dedupes_and_keeps_held_out_packages_out_of_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = "export function shared(a) {\n  return a * 2;\n}\n" * 3
            unique_a = "export const alpha = 1;\n" * 10
            unique_b = "export const beta = 2;\n" * 10
            held_src = "export const heldout = 3;\n" * 10
            manifest = {}
            held_key = next(f"npm:h{i}" for i in range(1000) if build.is_held_out(f"npm:h{i}", 2))
            train_key = next(f"npm:t{i}" for i in range(1000) if not build.is_held_out(f"npm:t{i}", 2))
            second_key = next(f"npm:s{i}" for i in range(1000) if not build.is_held_out(f"npm:s{i}", 2))
            archives = {
                train_key: [("src/shared.js", shared), ("src/alpha.js", unique_a)],
                second_key: [("src/shared.js", shared), ("src/beta.js", unique_b)],
                held_key: [("src/held.js", held_src), ("src/dup.js", shared)],
            }
            for key, files in archives.items():
                path = root / (key.replace(":", "_") + ".tgz")
                with tarfile.open(path, "w:gz") as tar:
                    for rel, text in files:
                        data = text.encode("utf-8")
                        info = tarfile.TarInfo(f"pkg-1.0/{rel}")
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
                manifest[key] = {"key": key, "allowed": True, "version": "1.0", "licence": "MIT", "ext": "tgz",
                                 "file": str(path)}
            bad = root / "corrupt.tgz"
            bad.write_bytes(b"not a tarball at all")
            manifest["npm:corrupt"] = {"key": "npm:corrupt", "allowed": True, "version": "1", "licence": "MIT",
                                       "ext": "tgz", "file": str(bad)}
            jobs = build._jobs(manifest)
            out = root / "out"
            staging = out / "staging"
            report = build.phase_one(jobs, staging, shards=3, heldout_every=2, workers=1)
            self.assertIn("error", report["npm:corrupt"])
            train_hashes, rows_per_shard, per_package = build.phase_two(staging, out, shards=3)
            kept, leaked = build.phase_three(staging, out, train_hashes, cap=100)

            train_texts = []
            for i in range(3):
                for line in (out / f"shard_{i:02d}.jsonl").read_text(encoding="utf-8").splitlines():
                    train_texts.append(json.loads(line)["completion"])
            self.assertEqual(len(train_texts), len(set(train_texts)), "duplicates survived dedupe")
            self.assertFalse(any("heldout = 3" in t for t in train_texts), "held-out package leaked into training")
            self.assertTrue(any("alpha = 1" in t for t in train_texts))
            self.assertTrue(any("beta = 2" in t for t in train_texts))
            self.assertEqual(leaked, 1, "the shared text inside the held-out package should be removed")
            held_rows = (out / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(kept, len(held_rows))
            self.assertTrue(all("shared(a)" not in json.loads(r)["completion"] for r in held_rows))
            self.assertTrue(isinstance(train_hashes, np.ndarray) and train_hashes.dtype == np.uint64)


class FetchTests(unittest.TestCase):
    def test_licence_override_needs_the_fingerprint_in_the_top_level_licence(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "good.zip", Path(tmp) / "bad.zip"
            with zipfile.ZipFile(good, "w") as zf:
                zf.writestr("repo-main/LICENSE", "Permission is hereby granted, free of charge\n")
                zf.writestr("repo-main/deps/other/LICENSE", "Some other licence")
            with zipfile.ZipFile(bad, "w") as zf:
                zf.writestr("repo-main/README.md", "no licence file here")
            self.assertTrue(fetch._licence_text_matches(good, "Permission is hereby granted, free of charge"))
            self.assertFalse(fetch._licence_text_matches(good, "PSF LICENSE AGREEMENT"))
            self.assertFalse(fetch._licence_text_matches(bad, "Permission is hereby granted"))

    def test_registry_failures_become_skips_not_crashes(self):
        class Broken(fetch.Http):
            def json(self, url):
                raise TimeoutError("boom")

        arc, skipped = fetch._resolve("npm", "x", lambda: fetch.resolve_npm(Broken(), "x"))
        self.assertIsNone(arc)
        self.assertFalse(skipped["allowed"])
        self.assertIn("unreachable", skipped["reason"])


if __name__ == "__main__":
    unittest.main()
