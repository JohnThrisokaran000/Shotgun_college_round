"""Pieces of the distributed run that every group depends on: starting weights, the int8 global,
the learning-rate schedule, the dataset split, and the upload inputs."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training import group_worker, initrng
from training import shards


class StartingWeightsTests(unittest.TestCase):
    def test_canonical_hash_matches_the_rust_coordinator(self):
        # The same value is pinned in crates/pw-train/src/coordinator.rs (canonical_hash_matches_the_python_reference).
        state = {"b": np.array([3.0], dtype=np.float32), "a": np.array([1.0, 2.0], dtype=np.float32)}
        self.assertEqual(initrng.canonical_sha256(state),
                         "8e628779e6a74ee0b36991c10158f63cafec7d340ad4e075592502c8708524dd")

    def test_init_is_deterministic_and_seed_dependent(self):
        shapes = {"layers.0.w": (64, 32), "layers.0.norm": (32,), "embed": (100, 16)}
        a = initrng.init_state(shapes, 0)
        b = initrng.init_state(shapes, 0)
        c = initrng.init_state(shapes, 1)
        self.assertEqual(initrng.canonical_sha256(a), initrng.canonical_sha256(b))
        self.assertNotEqual(initrng.canonical_sha256(a), initrng.canonical_sha256(c))

    def test_matrices_are_near_normal_and_one_dimensional_tensors_are_zero(self):
        state = initrng.init_state({"w": (400, 400), "norm": (512,)}, 3)
        self.assertLess(abs(float(state["w"].mean())), 0.001)
        self.assertLess(abs(float(state["w"].std()) - initrng.NORMAL_STD), 0.001)
        self.assertEqual(float(np.abs(state["norm"]).max()), 0.0)
        self.assertEqual(state["w"].dtype, np.float32)


class Int8GlobalTests(unittest.TestCase):
    def test_rows_dequantise_as_quantised_value_times_row_scale(self):
        q = torch.tensor([[127, -127, 0], [1, 2, 3]], dtype=torch.int8)
        scales = torch.tensor([0.5, 0.25], dtype=torch.float16)
        raw = {"m": q, "m.__scale": scales, "n": torch.tensor([1.5, -2.0], dtype=torch.float16)}
        out = group_worker.decode_global(raw, ["m", "n"])
        expected = torch.tensor([[63.5, -63.5, 0.0], [0.25, 0.5, 0.75]])
        self.assertTrue(torch.allclose(out["m"], expected))
        self.assertEqual(out["n"].dtype, torch.float32)
        self.assertTrue(torch.allclose(out["n"], torch.tensor([1.5, -2.0])))

    def test_missing_name_is_an_error_not_a_silent_zero(self):
        with self.assertRaises(KeyError):
            group_worker.decode_global({}, ["absent"])


class LearningRateTests(unittest.TestCase):
    def test_warmup_then_cosine_to_a_tenth_of_the_base_rate(self):
        base = 6e-4
        self.assertAlmostEqual(group_worker.time_lr(0, 0.0, base, 10), base / 10)
        self.assertAlmostEqual(group_worker.time_lr(20, 0.0, base, 10), base)
        self.assertAlmostEqual(group_worker.time_lr(20, 1.0, base, 10), base * 0.1, places=12)
        mid = group_worker.time_lr(20, 0.5, base, 10)
        self.assertAlmostEqual(mid, base * (0.1 + 0.9 * 0.5), places=12)

    def test_progress_is_clamped(self):
        self.assertAlmostEqual(group_worker.time_lr(20, 7.0, 1.0, 1), 0.1, places=12)
        self.assertAlmostEqual(group_worker.time_lr(20, -3.0, 1.0, 1), 1.0, places=12)
        self.assertTrue(math.isfinite(group_worker.time_lr(5, 0.3, 1.0, 1)))


class SplitTests(unittest.TestCase):
    def test_patchwork_rows_are_split_in_one_pass_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"prompt": "p", "completion": f"row {i}"} for i in range(10)]
            (root / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            paths = shards._split_patchwork(root, [3, 0, 4], root / "split")
            got = [[json.loads(l)["completion"] for l in p.read_text(encoding="utf-8").splitlines()] for p in paths]
            self.assertEqual(got, [["row 0", "row 1", "row 2"], [], ["row 3", "row 4", "row 5", "row 6"]])


if __name__ == "__main__":
    unittest.main()
