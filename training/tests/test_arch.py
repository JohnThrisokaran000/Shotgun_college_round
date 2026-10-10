import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import arch


class LayerPattern(unittest.TestCase):
    def test_every_sixth_layer_is_full_attention(self):
        types = arch.layer_types(24)
        full = [i for i, t in enumerate(types) if t == "full_attention"]
        self.assertEqual(full, [5, 11, 17, 23])
        self.assertEqual(types.count("sliding_attention"), 20)


class ParameterCounts(unittest.TestCase):
    def test_reference_vocab_is_about_0_3b_as_the_card_states(self):
        n = arch.count_params(arch.make_config(262_144))
        self.assertTrue(290e6 <= n <= 315e6, f"{n / 1e6:.1f}M")

    def test_patchwork_vocab_gives_a_model_near_125m(self):
        n = arch.count_params(arch.make_config(32_768))
        self.assertTrue(120e6 <= n <= 132e6, f"{n / 1e6:.1f}M")


def tiny_config(**kw):
    dims = dict(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=64,
    )
    dims.update(kw)
    return arch.make_config(50, **dims)


class TinyModel(unittest.TestCase):
    def test_embeddings_are_tied_and_attention_is_causal_by_default(self):
        cfg = tiny_config()
        self.assertFalse(cfg.use_bidirectional_attention)
        model = arch.build_model(cfg)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(0)
        model = arch.build_model(tiny_config()).eval()
        ids = torch.randint(3, 50, (1, 16))
        other = ids.clone()
        other[0, 10:] = torch.randint(3, 50, (6,))
        with torch.no_grad():
            a = model(input_ids=ids).logits
            b = model(input_ids=other).logits
        self.assertTrue(torch.allclose(a[:, :10], b[:, :10], atol=1e-5))
        self.assertFalse(torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5))

    def test_backward_gives_finite_gradients(self):
        torch.manual_seed(0)
        model = arch.build_model(tiny_config())
        ids = torch.randint(3, 50, (2, 12))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))


class ReferenceFile(unittest.TestCase):
    def test_only_architecture_keys_are_read(self):
        raw = {
            "hidden_size": 640,
            "num_hidden_layers": 18,
            "vocab_size": 262144,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
            },
            "architectures": ["Gemma3TextModel"],
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps(raw), encoding="utf-8")
            got = arch.reference_overrides(p)
        self.assertEqual(got["hidden_size"], 640)
        self.assertEqual(got["num_hidden_layers"], 18)
        self.assertEqual(got["global_rope_theta"], 1_000_000.0)
        self.assertNotIn("vocab_size", got)
        self.assertNotIn("architectures", got)

    def test_explicit_overrides_win_over_the_reference_file(self):
        cfg = arch.make_config(100, num_hidden_layers=4, hidden_size=64, intermediate_size=96,
                               num_attention_heads=2, head_dim=32)
        self.assertEqual(len(cfg.layer_types), 4)
        self.assertEqual(cfg.hidden_size, 64)


if __name__ == "__main__":
    unittest.main()
