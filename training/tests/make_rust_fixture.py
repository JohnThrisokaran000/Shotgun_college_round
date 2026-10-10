"""Write the fixture for the Rust decoder tests: a tiny Gemma-3-style model with the real tokenizer, and the
Python reference outputs it must reproduce (the greedy continuation, and the logits after the prompt).

    python -m training.tests.make_rust_fixture

Run it again only when the architecture or the tokenizer changes. The fixture is committed, so the Rust tests
do not need Python.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from training import arch, pwdata

OUT = Path("crates/pw-model/tests/fixture")
PROMPT = "TASK: relocate\nTARGET: button \"Place order\" in: main > section\nCANDIDATES:\n[0] a | \"Add to cart\"\n[1] button | \"Place order\"\nANSWER: "


def main() -> None:
    tokenizer = pwdata.load_tokenizer(Path("runs/tokenizer.json"))
    bos, eos, pad = (pwdata.ids(tokenizer, n) for n in (pwdata.BOS, pwdata.EOS, pwdata.PAD))
    cfg = arch.make_config(
        tokenizer.get_vocab_size(), pad_id=pad, bos_id=bos, eos_id=eos,
        hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, sliding_window=512, max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = arch.build_model(cfg).eval()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    model.save_pretrained(OUT, safe_serialization=True)
    shutil.copy(Path("runs/tokenizer.json"), OUT / "tokenizer.json")

    prompt_ids = tokenizer.encode(PROMPT).ids
    input_ids = torch.tensor([[bos] + prompt_ids])
    with torch.no_grad():
        last = model(input_ids=input_ids).logits[0, -1].float()
        generated = model.generate(input_ids=input_ids, max_new_tokens=8, do_sample=False,
                                   eos_token_id=eos, pad_token_id=pad)[0, input_ids.shape[1]:].tolist()
    if eos in generated:
        generated = generated[: generated.index(eos)]
    top = torch.topk(last, 5)
    expected = {
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids) + 1,
        "greedy_text": tokenizer.decode(generated).strip(),
        "greedy_tokens": generated,
        "top5_index": top.indices.tolist(),
        "top5_logit": [round(v, 6) for v in top.values.tolist()],
        "logits_first16": [round(v, 6) for v in last[:16].tolist()],
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(json.dumps({k: expected[k] for k in ("prompt_tokens", "greedy_text", "top5_index")}, indent=1))
    print(f"fixture: {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
