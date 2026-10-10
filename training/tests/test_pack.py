import random
import tempfile
import unittest
from pathlib import Path

from training import pack, pwdata


class PackRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        corpus = ["TASK: relocate OLD: button | name=\"Buy now\"", "CANDIDATES: [0] a | [1] button", '{"match": 1}',
                  '{"ops":[{"op":"setText","value":"Place order"}]}'] * 20
        self.tok_path = Path(self.tmp.name) / "tokenizer.json"
        self.tok = pwdata.train_tokenizer(corpus, self.tok_path, vocab_size=300)
        self.bos, self.eos, self.pad = pwdata.ids(self.tok, pwdata.BOS), pwdata.ids(self.tok, pwdata.EOS), pwdata.ids(self.tok, pwdata.PAD)
        self.rows = [
            {"prompt": "TASK: relocate OLD: button", "completion": '{"match": 1}'},
            {"prompt": "TASK: author TARGET: a button", "completion": '{"ops":[{"op":"hide"}]}'},
            {"prompt": "x" * 2000, "completion": "too long to fit"},
        ]
        self.out = Path(self.tmp.name) / "packed"
        self.meta = pack.pack(self.rows, self.tok, self.out, max_len=128, bos=self.bos, eos=self.eos, pad=self.pad)
        self.ds = pack.PackedDataset(self.out)

    def tearDown(self):
        self.tmp.cleanup()

    def test_examples_round_trip_with_completion_offsets(self):
        self.assertEqual(self.meta["examples"], 2)
        self.assertEqual(self.meta["dropped_too_long"], 1)
        ids0, c0 = self.ds.get(0)
        expected = [self.bos] + self.tok.encode(self.rows[0]["prompt"]).ids + self.tok.encode(self.rows[0]["completion"]).ids + [self.eos]
        self.assertEqual(list(ids0), expected)
        self.assertEqual(c0, 1 + len(self.tok.encode(self.rows[0]["prompt"]).ids))

    def test_collate_masks_prompt_and_padding(self):
        ids, labels, mask = pack.collate(self.ds, [0, 1], self.pad)
        self.assertEqual(ids.shape, labels.shape)
        self.assertEqual(int(mask[0].sum()), self.ds.length(0))
        _, c0 = self.ds.get(0)
        self.assertTrue((labels[0, :c0] == pack.IGNORE).all())
        self.assertEqual(list(labels[0, c0 : self.ds.length(0)]), list(ids[0, c0 : self.ds.length(0)]))
        # Padding is never a training target.
        self.assertTrue((labels[mask == 0] == pack.IGNORE).all())


class Sharding(unittest.TestCase):
    def test_every_example_goes_to_exactly_one_rank(self):
        n, world = 101, 28
        seen = []
        for r in range(world):
            seen.extend(pack.shard_indices(n, r, world))
        self.assertEqual(sorted(seen), list(range(n)))

    def test_bucketed_batches_cover_each_index_once(self):
        with tempfile.TemporaryDirectory() as d:
            tokenizer_path = Path(d) / "t.json"
            tok = pwdata.train_tokenizer(["abc def ghi " * 5] * 10, tokenizer_path, vocab_size=120)
            rows = [{"prompt": "p" * (i % 17 + 1), "completion": "c"} for i in range(60)]
            pack.pack(rows, tok, Path(d) / "p", max_len=256, bos=pwdata.ids(tok, pwdata.BOS), eos=pwdata.ids(tok, pwdata.EOS), pad=pwdata.ids(tok, pwdata.PAD))
            ds = pack.PackedDataset(Path(d) / "p")
            batches = list(pack.length_bucketed_batches(ds, list(range(len(ds))), 8, random.Random(0)))
            flat = sorted(i for b in batches for i in b)
            self.assertEqual(flat, list(range(len(ds))))


if __name__ == "__main__":
    unittest.main()
