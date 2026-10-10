"""Pre-tokenized, memory-mapped training data.

Tokenizing once, on the machine that builds the dataset, means no Kaggle group spends GPU
time or session time on tokenization. Each example is stored as one run of uint16 ids.
Alongside it we keep the index where the completion starts, so the loss is taken on the
completion only.

Files written to out_dir:
    tokens.u16   concatenated token ids
    offsets.i64  start offset of each example, plus one final end offset
    comp.i32     index of each example's first completion token
    meta.json    vocabulary size, counts, and the settings used
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

IGNORE = -100


def pack(rows: Iterable[dict], tokenizer, out_dir: Path, *, max_len: int, bos: int, eos: int, pad: int) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = array("H")
    offsets = array("q", [0])
    comp = array("i")
    dropped = 0
    vocab = tokenizer.get_vocab_size()
    if vocab > 65535:
        raise ValueError("uint16 storage needs a vocabulary of at most 65535")
    for row in rows:
        prompt = tokenizer.encode(row["prompt"]).ids
        completion = tokenizer.encode(row["completion"]).ids + [eos]
        seq = [bos] + prompt + completion
        if len(seq) > max_len:
            dropped += 1
            continue
        comp.append(1 + len(prompt))
        tokens.extend(seq)
        offsets.append(len(tokens))
    with open(out_dir / "tokens.u16", "wb") as f:
        tokens.tofile(f)
    with open(out_dir / "offsets.i64", "wb") as f:
        offsets.tofile(f)
    with open(out_dir / "comp.i32", "wb") as f:
        comp.tofile(f)
    meta = {
        "examples": len(comp),
        "tokens": len(tokens),
        "dropped_too_long": dropped,
        "max_len": max_len,
        "vocab": vocab,
        "bos": bos,
        "eos": eos,
        "pad": pad,
        "dtype": "uint16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class PackedDataset:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        self.meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokens = np.memmap(data_dir / "tokens.u16", dtype=np.uint16, mode="r")
        self.offsets = np.fromfile(data_dir / "offsets.i64", dtype=np.int64)
        self.comp = np.fromfile(data_dir / "comp.i32", dtype=np.int32)
        if len(self.offsets) != len(self.comp) + 1:
            raise ValueError("offsets and comp files disagree on the number of examples")

    def __len__(self) -> int:
        return len(self.comp)

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def get(self, i: int) -> tuple[np.ndarray, int]:
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[s:e]), int(self.comp[i])


def shard_indices(n: int, rank: int, world: int) -> list[int]:
    """Every example belongs to exactly one rank."""
    return list(range(rank, n, world))


def length_bucketed_batches(ds: PackedDataset, indices: list[int], batch: int, rng, bucket: int = 50):
    """Shuffle, then group similar lengths together so less of each batch is padding."""
    order = list(indices)
    rng.shuffle(order)
    chunk = batch * bucket
    for start in range(0, len(order), chunk):
        part = sorted(order[start : start + chunk], key=ds.length)
        groups = [part[i : i + batch] for i in range(0, len(part), batch)]
        rng.shuffle(groups)
        yield from groups


def collate(ds: PackedDataset, idxs: list[int], pad: int):
    """Return (input_ids, labels, attention_mask) as numpy int64 arrays. Labels cover the completion only."""
    items = [ds.get(i) for i in idxs]
    width = max(len(t) for t, _ in items)
    ids = np.full((len(items), width), pad, dtype=np.int64)
    labels = np.full((len(items), width), IGNORE, dtype=np.int64)
    mask = np.zeros((len(items), width), dtype=np.int64)
    for r, (toks, c) in enumerate(items):
        n = len(toks)
        ids[r, :n] = toks
        mask[r, :n] = 1
        labels[r, c:n] = toks[c:]
    return ids, labels, mask
