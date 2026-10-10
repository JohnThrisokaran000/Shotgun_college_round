"""Data handling for PatchWork training: JSONL rows, the tokenizer, and batches.

Each row is {"task", "split", "site", "kind", "candidates", "prompt", "completion"}.
A training sequence is  <bos> prompt completion <eos>; loss is taken on the completion
and <eos> only, so the model learns to answer and not to repeat the prompt.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
IGNORE = -100


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def train_tokenizer(texts: Iterable[str], out: Path, vocab_size: int = 8192):
    """Byte-level BPE trained from scratch on the given texts. Saves tokenizer.json."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD, BOS, EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    return tok


def ids(tok, name: str) -> int:
    value = tok.token_to_id(name)
    if value is None:
        raise ValueError(f"tokenizer has no {name} token")
    return value


def encode_row(tok, row: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    """Return (input_ids, labels), or None when the row does not fit in max_len."""
    prompt_ids = tok.encode(row["prompt"]).ids
    completion_ids = tok.encode(row["completion"]).ids
    bos, eos = ids(tok, BOS), ids(tok, EOS)
    seq = [bos] + prompt_ids + completion_ids + [eos]
    if len(seq) > max_len:
        return None
    labels = [IGNORE] * (1 + len(prompt_ids)) + completion_ids + [eos]
    return seq, labels


def encode_rows(tok, rows: Iterable[dict], max_len: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    kept, dropped = [], 0
    for row in rows:
        item = encode_row(tok, row, max_len)
        if item is None:
            dropped += 1
        else:
            kept.append(item)
    return kept, dropped


def batches(items: list[tuple[list[int], list[int]]], batch: int, pad_id: int, rng: random.Random, shuffle: bool = True):
    """Yield (input_ids, labels, attention_mask) as lists of equal-length rows."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch):
        chunk = [items[i] for i in order[start : start + batch]]
        width = max(len(seq) for seq, _ in chunk)
        input_ids = [seq + [pad_id] * (width - len(seq)) for seq, _ in chunk]
        labels = [lab + [IGNORE] * (width - len(lab)) for _, lab in chunk]
        mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq, _ in chunk]
        yield input_ids, labels, mask
