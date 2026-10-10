"""Gemma-3-style decoder architecture for the PatchWork model.

Scope, stated plainly: the EmbeddingGemma-300M model is used ONLY as an architectural
reference for the layer design (Gemma 3 blocks: grouped-query attention, QK-norm, pre and
post RMSNorm, GeGLU MLP, RoPE with separate local and global bases, and a 5:1
sliding-window to full-attention layer pattern). No weights, tokenizer, or pretrained state
from that model are used anywhere in this package.

The dimensions in TEMPLATE are assumptions. The EmbeddingGemma config.json is gated and was
not read. They are chosen to match the family's published design and the card's stated
"0.3B params" when the 262k reference vocabulary is used. A reference config (architecture
keys only, no weights) can override them with `reference_overrides(path)`.

The model is a causal language model. Training uses the PatchWork vocabulary, not the
262k vocabulary of the reference model.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import Gemma3TextConfig

TEMPLATE = {
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_hidden_layers": 24,
    "num_attention_heads": 3,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "sliding_window": 512,
    "max_position_embeddings": 2048,
    "local_rope_theta": 10_000.0,
    "global_rope_theta": 1_000_000.0,
    "local_to_global": 5,
}

# Keys we accept from a reference config.json. Anything else (tokenizer, weights) is ignored.
_ARCH_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "sliding_window",
    "max_position_embeddings",
)


def layer_types(n_layers: int, local_to_global: int = 5) -> list[str]:
    """Every (local_to_global + 1)-th layer uses full attention, the rest use a sliding window."""
    return [
        "full_attention" if (i + 1) % (local_to_global + 1) == 0 else "sliding_attention"
        for i in range(n_layers)
    ]


def reference_overrides(path: Path) -> dict:
    """Read architecture keys from a reference config.json. Weights and tokenizer are never read."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {k: raw[k] for k in _ARCH_KEYS if k in raw}
    if "rope_parameters" in raw:
        rp = raw["rope_parameters"]
        out["local_rope_theta"] = float(rp["sliding_attention"]["rope_theta"])
        out["global_rope_theta"] = float(rp["full_attention"]["rope_theta"])
    else:
        if "rope_local_base_freq" in raw:
            out["local_rope_theta"] = float(raw["rope_local_base_freq"])
        if "rope_theta" in raw:
            out["global_rope_theta"] = float(raw["rope_theta"])
    return out


def make_config(
    vocab_size: int,
    *,
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    reference: Path | None = None,
    **overrides,
) -> Gemma3TextConfig:
    """Build the decoder config. Explicit `overrides` win over the reference file, which wins over TEMPLATE."""
    dims = dict(TEMPLATE)
    if reference is not None:
        dims.update(reference_overrides(reference))
    dims.update(overrides)
    n_layers = int(dims["num_hidden_layers"])
    return Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=int(dims["hidden_size"]),
        intermediate_size=int(dims["intermediate_size"]),
        num_hidden_layers=n_layers,
        num_attention_heads=int(dims["num_attention_heads"]),
        num_key_value_heads=int(dims["num_key_value_heads"]),
        head_dim=int(dims["head_dim"]),
        sliding_window=int(dims["sliding_window"]),
        max_position_embeddings=int(dims["max_position_embeddings"]),
        layer_types=layer_types(n_layers, int(dims["local_to_global"])),
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": float(dims["local_rope_theta"])},
            "full_attention": {"rope_type": "default", "rope_theta": float(dims["global_rope_theta"])},
        },
        query_pre_attn_scalar=int(dims["head_dim"]),
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_bidirectional_attention=False,
        use_cache=False,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )


def build_model(cfg: Gemma3TextConfig):
    from transformers import Gemma3ForCausalLM

    return Gemma3ForCausalLM(cfg)


def count_params(cfg: Gemma3TextConfig) -> int:
    """Parameter count without allocating the weights (meta device)."""
    import torch

    with torch.device("meta"):
        model = build_model(cfg)
    return sum(p.numel() for p in model.parameters())
