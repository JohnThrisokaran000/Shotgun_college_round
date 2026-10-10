# Training the PatchWork model

The model is a LLaMA-style decoder trained from scratch with PyTorch. The data, tokenizer, coordinator and launcher are in Rust (`pw` and `pw-train`). Training itself is Python, which the brief allows.

## 1. Data (Rust)

```sh
cargo run -p pw-cli -- train gen-data --out data/patchwork --sites 20000 --variants 6 --author 3 --seed 1
```

Writes `train.jsonl`, `heldout.jsonl` and `meta.json`. RELOCATE rows use the same fast path, prefilter and prompt as the CLI. AUTHOR rows are checked against the DSL. Splits are by site, and deleted targets are capped at 15% of RELOCATE rows.

## 2. Single-process training (CPU or one GPU)

```sh
.venv/bin/python -m training.train --data data/patchwork --out runs/x \
    --hidden 256 --layers 4 --heads 4 --steps 2000 --batch 32 --lr 6e-4 --device auto --amp
```

Outputs `runs/x/model/` (safetensors, config, tokenizer), `runs/x/metrics.json`, and the tokenizer. Metrics are heldout loss and greedy exact match on held-out rows.

## 3. Distributed training (DiLoCo)

```sh
.venv/bin/python -m training.init_model --tokenizer runs/x/tokenizer.json \
    --train-rows data/patchwork/train.jsonl --out runs/dist --hidden 512 --layers 8 --heads 8
./target/debug/pw train serve --init runs/dist/init.safetensors --state runs/dist/state \
    --serve runs/dist/serve --listen 127.0.0.1:8765 --workers 3 --timeout-minutes 10
.venv/bin/python -m training.worker --coordinator http://127.0.0.1:8765 --worker-id w0 \
    --rank 0 --world 3 --inner-steps 500 --rounds 40 --batch 32 --work-dir runs/dist/w0
```

Each worker trains on its shard (`items[rank::world]`) for `--inner-steps`, sends `global - local` as fp16, and resyncs when the coordinator advances. The coordinator averages the deltas and applies a Nesterov outer step (lr 0.7, momentum 0.9 by default).

Evaluate any round's global weights:

```sh
.venv/bin/python -m training.eval_checkpoint --config runs/dist/config.json \
    --weights runs/dist/state/global.safetensors --tokenizer runs/dist/tokenizer.json \
    --data data/patchwork/heldout.jsonl --limit 200
```

## 4. Kaggle

See `docs/KAGGLE.md`. Plans are offline. Nothing is pushed without explicit confirmation.

## Limits

- The synthetic data teaches the two PatchWork jobs. It does not teach general developer knowledge. A general corpus is a separate decision (see `docs/KAGGLE.md`).
- `training/worker.py` uses CUDA and fp16 AMP when a GPU is present. Only CPU was exercised in this environment.
- No model is wired into the `pw` CLI yet. Inference from Rust (candle) is the next step. Until then the CLI uses the deterministic heuristic.
