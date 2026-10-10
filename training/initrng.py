"""Deterministic weight initialisation that gives the same bits on every machine.

The coordinator and every Kaggle group must start from exactly the same weights, and PyTorch's
random generators do not promise that across versions and hardware. This module uses only
integer hashing (splitmix64, wrapping arithmetic) and float64 additions and multiplications,
which IEEE arithmetic defines exactly, in a fixed order. It calls no transcendental functions,
because those differ between math libraries in the last bit.

Weights are approximately normal (a sum of uniforms, rescaled), with standard deviation
NORMAL_STD. One-dimensional tensors are zero. In this architecture they are the RMSNorm scales,
which are applied as (1 + w), so zero is the identity.

canonical_sha256 hashes the float32 values in name order. The coordinator reports the same
hash, so a group can check that its starting weights match the coordinator's before it trains.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

NORMAL_STD = 0.02
DRAWS = 6  # a sum of 6 uniforms, centred and rescaled, is close enough to normal for initialisation

_MASK = (1 << 64) - 1
_K_STREAM = 0xD1B54A32D192ED03
_K_DRAW = 0x8CB92BA72F3D8DD7
_K_INDEX = 0x2545F4914F6CDD1D
_K_GOLDEN = 0x9E3779B97F4A7C15


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser over a uint64 array, with wrapping arithmetic."""
    x = x + np.uint64(_K_GOLDEN)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _uniform(stream: int, n: int, draw: int) -> np.ndarray:
    """n uniforms in [0, 1), with 53 random bits each, for one stream and one draw."""
    base = (stream * _K_STREAM + draw * _K_DRAW) & _MASK
    index = np.arange(n, dtype=np.uint64) * np.uint64(_K_INDEX)
    bits = _mix(index + np.uint64(base))
    return (bits >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


def _approx_normal(stream: int, shape: tuple[int, ...], std: float) -> np.ndarray:
    n = math.prod(shape)
    total = np.zeros(n, dtype=np.float64)
    for draw in range(DRAWS):
        total = total + _uniform(stream, n, draw)  # fixed order of additions
    centred = (total - DRAWS / 2.0) * math.sqrt(12.0 / DRAWS)  # unit variance: each uniform has variance 1/12
    return (centred * std).astype(np.float32).reshape(shape)


def init_state(shapes: dict[str, tuple[int, ...]], seed: int) -> dict[str, np.ndarray]:
    """Starting weights for every named tensor. Tensors are visited in name order, one stream each."""
    state: dict[str, np.ndarray] = {}
    for index, name in enumerate(sorted(shapes)):
        shape = tuple(int(d) for d in shapes[name])
        if len(shape) == 1:
            state[name] = np.zeros(shape, dtype=np.float32)
        else:
            stream = (seed * 1_000_003 + index + 1) & _MASK
            state[name] = _approx_normal(stream, shape, NORMAL_STD)
    return state


def canonical_sha256(state: dict[str, np.ndarray]) -> str:
    """sha256 over the float32 little-endian bytes of every tensor, in name order. Rust computes the same."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(np.ascontiguousarray(state[name], dtype="<f4").tobytes())
    return digest.hexdigest()
