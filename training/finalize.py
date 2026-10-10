"""Turn a coordinator checkpoint into the finished model: evaluate it, export it, and write its model card.

    python -m training.finalize --state runs/state --serve runs/serve --out runs/final \
        --patchwork data/pw-v2/heldout.jsonl --general data/corpus/general/heldout.jsonl

Reads state/global.safetensors (the newest merged global model) and state/status.json, then writes into --out:
    model.safetensors, config.json, generation_config.json, tokenizer.json   a Hugging Face model folder
    eval.json                  held-out loss, perplexity, and PatchWork task scores
    samples.md                 a few real completions, so the result can be judged by eye
    MODEL_CARD.md              what the model is, how it was trained, the numbers, and its limits
Evaluation runs on --device (cpu by default). It scores held-out data only: whole packages and sites that no
group trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from training import eval_checkpoint as ev
from training import pwdata


def samples(model, tok, patchwork_rows: list[dict], device, n: int) -> str:
    lines = ["# Sample completions (greedy, held-out sites)", ""]
    for task in ("relocate", "author"):
        rows = [r for r in patchwork_rows if r["task"] == task][:n]
        for row in rows:
            pred = ev.greedy(model, tok, row["prompt"], device, 16 if task == "relocate" else 128)
            lines += [f"## {task}", "", "```", row["prompt"][-600:], "```", "",
                      f"expected: `{row['completion']}`", f"model:    `{pred}`", ""]
    return "\n".join(lines)


def model_card(report: dict, status: dict, run: dict, extra: dict) -> str:
    def row(label, key, fmt="{}"):
        return f"| {label} | {fmt.format(report[key])} |" if key in report else ""

    lines = [
        "# PatchWork model", "",
        f"A {report['params'] / 1e6:.1f}M-parameter decoder trained from scratch with distributed DiLoCo-style training on",
        "Kaggle T4 GPUs. Architecture: Gemma-3-style (grouped-query attention, QK-norm, pre/post RMSNorm, GeGLU,",
        "RoPE with local and global bases, 5:1 sliding-window to full-attention layers). No pretrained weights,",
        "tokenizer or data from any other model were used. EmbeddingGemma was an architectural reference only.", "",
        f"Checkpoint: global version {status.get('round')} (hash `{str(status.get('global_sha256'))[:16]}`).", "",
        "## Held-out results", "",
        "Held-out means whole packages and synthetic sites that no group trained on.", "",
        "| metric | value |", "|---|---|",
        row("general held-out loss (nats/token)", "general_heldout_loss"),
        row("general held-out perplexity", "general_heldout_ppl"),
        row("PatchWork held-out loss (completion tokens)", "patchwork_heldout_loss"),
        row("relocate exact match", "relocate_exact"),
        row("relocate: picks the right element", "relocate_pick_exact"),
        row("relocate: refuses when it should", "relocate_refusal_exact"),
        row("relocate: wrong-pick rate (harmful errors)", "relocate_wrong_pick_rate"),
        row("author exact ops match", "author_exact"),
        row("author valid DSL JSON", "author_valid_json"), "",
        f"Scored on {report.get('relocate_n', 0)} relocate and {report.get('author_n', 0)} author rows.", "",
        "## Training data", "",
        "- General corpus: permissively licensed code and docs from npm, PyPI, crates.io and GitHub (three.js,",
        "  Babylon.js, TypeScript, VS Code, React, Vue, Svelte, Node, CPython). Sources and licences: see LICENSES.md",
        "  in the dataset. Copyleft and share-alike sources (GPL, LGPL, MPL, CC-BY-SA such as MDN) were excluded.",
        "- Synthetic PatchWork data (about 20% of tokens): element re-anchoring across eight page kinds and sixteen",
        "  mutations, including refusals for deleted targets and for identical twins, plus DSL authoring requests.", "",
        "## Limits", "",
        "- This is a small model with a short training run. It follows the PatchWork formats; its general code",
        "  knowledge is partial and it will make mistakes.",
        "- Context is 512 tokens. The tokenizer has 32,768 tokens and was trained on the same mix.",
        "- Synthetic pages are templated. Scores on real websites have not been measured.",
        f"- {extra.get('note', '')}", "",
    ]
    return "\n".join(l for l in lines if l is not None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", type=Path, default=Path("runs/state"))
    p.add_argument("--weights", type=Path, default=None, help="override the weights file (default state/global.safetensors)")
    p.add_argument("--serve", type=Path, default=Path("runs/serve"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--patchwork", type=Path, required=True)
    p.add_argument("--general", type=Path, required=True)
    p.add_argument("--limit", type=int, default=300, help="rows per PatchWork task for generation scoring")
    p.add_argument("--loss-rows", type=int, default=2000)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--note", default="")
    args = p.parse_args()

    started = time.time()
    weights = args.weights or args.state / "global.safetensors"
    status_path = args.state / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    run = json.loads((args.serve / "run.json").read_text()) if (args.serve / "run.json").exists() else {}
    device = torch.device(args.device)
    tok = pwdata.load_tokenizer(args.serve / "tokenizer.json")
    model, _ = ev.load_model(args.serve, weights, device)

    report: dict = {"weights": str(weights), "params": sum(p.numel() for p in model.parameters()),
                    "global_version": status.get("round"), "global_sha256": status.get("global_sha256")}
    general = list(pwdata.read_jsonl(args.general))[: args.loss_rows]
    loss = ev.teacher_forced_loss(model, tok, general, device)
    report.update(general_heldout_loss=round(loss, 4), general_heldout_ppl=round(math.exp(min(loss, 50)), 2))
    rows = list(pwdata.read_jsonl(args.patchwork))
    report["patchwork_heldout_loss"] = round(ev.teacher_forced_loss(model, tok, rows[: args.loss_rows], device), 4)
    report.update(ev.score_patchwork(model, tok, rows, device, args.limit))
    report["eval_seconds"] = round(time.time() - started, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    ev.export(model, args.serve, args.out, report)
    (args.out / "samples.md").write_text(samples(model, tok, rows, device, args.samples), encoding="utf-8")
    (args.out / "MODEL_CARD.md").write_text(model_card(report, status, run, {"note": args.note}), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nfinished model written to {args.out}")


if __name__ == "__main__":
    main()
