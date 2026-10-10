# PatchWork training

The model is a Gemma-3-style decoder trained from scratch with PyTorch. The coordinator and launcher are Rust (`pw train`). The training code is Python, which the brief allows. Architecture details, the topology, and the Kaggle runbook are in `docs/KAGGLE.md`.

## Files

| File | Role |
|---|---|
| `arch.py` | Gemma-3-style architecture config and model, and parameter counts (EmbeddingGemma is the architectural reference only) |
| `pack.py` | Pre-tokenized, memory-mapped data; completion-only loss masks; uniform sharding; length-bucketed batches |
| `prepare.py` | Builds the folder the coordinator serves: config, tokenizer, starting weights, packed data |
| `group_worker.py` | One DiLoCo replica per group (two GPUs with DDP, or one GPU). Async uploads and merges, and `PW` log lines |
| `train.py` | Single-process training and evaluation (CPU or one GPU), also used by the group worker for its schedule |
| `logparse.py` | Summarises `PW` log lines per group |
| `eval_checkpoint.py` | Held-out evaluation of any checkpoint |
| `tests/` | Unit tests (`python -m unittest discover -s training/tests -t .`) and the notebook rehearsal harness |

## Data

```sh
pw train gen-data --out data/patchwork --sites 20000 --variants 6 --author 3 --seed 1
```

RELOCATE rows use the same fast path, prefilter, and prompt as the CLI. AUTHOR rows are checked against the DSL. Splits are by site, and deleted targets are capped at 15% of RELOCATE rows.

## Prepare, then train

```sh
# Served folder: config, tokenizer, starting weights, packed data
python -m training.prepare --data-dir data/patchwork --tokenizer runs/x/tokenizer.json --out runs/serve \
    --hidden 768 --layers 24 --heads 3 --kv-heads 1 --head-dim 256 --ffn 1152 --sliding-window 512 --ctx 512

# Coordinator (Rust)
pw train serve --init runs/serve/init.safetensors --state runs/state --serve runs/serve \
    --listen 127.0.0.1:8765 --workers 2 --quorum 2

# One group on this machine (CPU here; on Kaggle the notebook runs this with torchrun on two GPUs)
python -m training.group_worker --coordinator http://127.0.0.1:8765 --group g0 --replica 0 --replicas 2 \
    --device cpu --inner-steps 200 --sync-every 10 --micro-batch 8 --grad-accum 2 --total-steps 2000

# Watch the logs
python -m training.logparse runs/*/g0.log runs/*/g1.log
```

## Checks

```sh
python -m unittest discover -s training/tests -t . -v           # architecture and data
python -m training.tests.run_notebook_locally <notebook> <dir> --replace-url <url> --with-url <local url>
```

## Limits

- The GPU paths (fp16 autocast, fused AdamW, NCCL DDP, T4 throughput) have not run in this environment. This machine's NVIDIA driver and NVML library do not match.
- The synthetic data teaches the two PatchWork jobs. It does not teach general developer knowledge.
- No model is wired into the `pw` CLI yet. Inference from Rust (candle) is the next step.
