# PatchWork training

The model is a Gemma-3-style decoder trained from scratch with PyTorch. The coordinator and launcher are Rust (`pw train`). The training code is Python, which the brief allows. Architecture details, the topology, and the Kaggle runbook are in `docs/KAGGLE.md`.

## Files

| File | Role |
|---|---|
| `arch.py` | Gemma-3-style architecture config and model (EmbeddingGemma is the architectural reference only) |
| `initrng.py` | Deterministic starting weights: the same bits on every machine, and the hash every group checks against the coordinator |
| `pack.py` | Pre-tokenized, memory-mapped data; completion-only loss masks; length-bucketed batches |
| `corpus/` | General corpus: `discover`, `fetch` (licence-checked), `extract` (safe archive reading, filters, chunking), `build` (dedupe, held-out packages) |
| `shards.py` | Tokenizer training, and one packed dataset per Kaggle account |
| `prepare.py` | The coordinator's folder: config, tokenizer, starting weights, and their hash |
| `group_worker.py` | One DiLoCo replica per group (two GPUs with DDP). Async uploads and merges, live events to the coordinator |
| `kaggle_upload.py`, `kaggle_ops.py`, `nettest.py`, `keep_alive.py` | Dataset upload, status checks, the internet check, and the kernel watchdog |
| `live.py`, `status.py`, `logparse.py` | Follow the run: event stream, one table, and log summaries |
| `eval_checkpoint.py`, `finalize.py` | Held-out evaluation and export; `finalize` also writes the model card |
| `tests/` | Unit tests (`python -m unittest discover -s training/tests -t .`) and the notebook rehearsal harness |

The commands, in order, are in `docs/KAGGLE.md`.

## Checks

```sh
python -m unittest discover -s training/tests -t . -v     # 40 tests
cargo test --workspace                                     # 121 tests
python -m training.tests.run_notebook_locally <notebook> <dir> --replace-url <url> --with-url <local url> \
    --replace-input <local datasets> --fake-gpus 2         # rehearse a notebook on CPU
```

## Limits

- The corpus is about 700M tokens; a long run repeats it several times.
- The synthetic PatchWork data is templated. Scores on real websites have not been measured.
- The `pw` CLI runs the model with `--model DIR` (crates/pw-model, candle). The heuristic remains the default.
