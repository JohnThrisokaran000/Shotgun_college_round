# Kaggle training: runbook, design, and open decisions

## What the setup is

- **Model.** A Gemma-3-style decoder, built with transformers' `Gemma3ForCausalLM`. Grouped-query attention, QK-norm, pre- and post-RMSNorm, GeGLU, and RoPE with separate local and global bases. Layers follow a 5:1 sliding-window to full-attention pattern. Trained from scratch, with the PatchWork vocabulary.
- **EmbeddingGemma-300M is an architectural reference only.** Its weights, tokenizer, and pretrained state are not used. Its config.json is gated behind Google's usage license, so I could not read it, and the model card does not list the layer dimensions. The dimensions in `training/arch.py` (`TEMPLATE`) are my assumptions, chosen to match the family's design and the card's stated "0.3B params" when the 262k vocabulary is used. To match the real config exactly, pass its architecture keys as a file: `training.prepare --reference path/to/config.json`. The loader reads architecture keys only, and ignores weights and tokenizer settings.
- **Size.** About 126M parameters with a 32k vocabulary (about 302M with the 262k reference vocabulary, consistent with the card's 0.3B). The verified counts are in `training/tests/test_arch.py`.
- **Transfer learning:** not used, per your instruction.

## Topology

- **Groups.** One Kaggle notebook per account, with two T4s. Groups cannot talk to each other directly. They share only through the coordinator.
- **Inside a group (default, `--mode ddp`).** The two GPUs train one replica with DistributedDataParallel. Rank 0 does all network I/O, and every merge decision is broadcast, so both GPUs stay identical. The checksum test confirms this on CPU.
- **Fallback (`--mode independent`).** Each GPU is its own replica, with no communication inside the group. Use this if NCCL fails on a notebook.
- **Across groups (DiLoCo).** Each group sends a fp16 outer delta and receives the merged global back.
- **Maximum usage.** Communication overlaps with compute. A group keeps training while its delta is in flight, and merges when the coordinator's next version includes it: `local ← global + (local − snapshot)`. The progress made during the wait is kept. Deltas may be one global version old (`MAX_STALENESS = 1` in the coordinator). A delta older than that is refused, and the group resyncs.
- **Idempotent retries.** Every delta carries a per-worker sequence number. A repeated upload is acknowledged without being counted twice. This fixed a double-counting bug found during testing.

## Monitoring: logs only

The notebook output is the only feedback channel, so every monitoring line is `PW {json}`. Events:
`start`, `step` (loss, tokens/s, merges, sends, memory), `send`, `merge`, `eval` (held-out loss after each merge), `delta_dropped`, `net_error`, `probe`, `error` (with a trace), and `done`. The coordinator logs one `outer` event per global version.

Summarise any set of logs with:

```sh
python -m training.logparse kaggle-output/*.log
```

Each line gives status, last step, loss, tokens/s, sends, merges, version, the held-out trend, dropped deltas, network errors, and tracebacks. Each notebook also writes `pw_run.log` and prints its own summary at the end.

## Runbook

1. **Data.** Use the synthetic PatchWork data, or any JSONL with `prompt` and `completion` fields. Real general developer data is still open (see "Open decisions").
2. **Prepare the served folder.**
   ```sh
   python -m training.prepare --data-dir data/patchwork --tokenizer runs/x/tokenizer.json --out runs/serve \
       --hidden 768 --layers 24 --heads 3 --kv-heads 1 --head-dim 256 --ffn 1152 --sliding-window 512 --ctx 512
   ```
   Add `--reference config.json` to take the architecture from the real EmbeddingGemma config instead. Use `--limit-train` to test on a subset.
3. **Start the coordinator** at a public HTTPS URL, with one worker per group (14 in DDP mode):
   ```sh
   PW_TOKEN=... pw train serve --init runs/serve/init.safetensors --state runs/state \
       --serve runs/serve --listen 0.0.0.0:8765 --workers 14 --quorum 10 --timeout-minutes 45
   ```
4. **Rehearse one notebook on this machine** before any push. It runs the notebook's own cells against your local coordinator (gloo, CPU):
   ```sh
   python -m training.tests.run_notebook_locally kaggle/plan/k01/pw_worker.ipynb rehearsal/k01 \
       --replace-url https://coord.example.org --with-url http://127.0.0.1:8765
   ```
5. **Plan.** This is offline and writes no credentials:
   ```sh
   pw train kaggle plan --accounts kaggle/accounts.txt --coordinator-url https://YOUR-PUBLIC-URL \
       --out kaggle/plan --gpus 2 --mode ddp --inner-steps 1000 --sync-every 10 \
       --micro-batch 8 --grad-accum 2 --lr 6e-4 --warmup-steps 200 --total-steps 20000 --max-minutes 540
   ```
   Review `kaggle/plan/PLAN.md`. Set `--total-steps` to the number you actually plan to run.
6. **Probe fp16 health first.** Run one notebook with `--probe-only` before a full run. It reports finite losses and maximum activations. Gemma-style activations can overflow in fp16, and T4s have no bf16. If the probe shows overflow, switch to fp32 (much slower on a T4) or reduce the learning rate.
7. **Push.** A dry run prints the commands and sends nothing:
   ```sh
   pw train kaggle push --plan kaggle/plan
   pw train kaggle push --plan kaggle/plan --keys kaggle/secrets/keys.env \
       --execute --confirm-accounts 14 --i-have-checked-kaggle-rules
   ```
8. **Monitor.** Read the notebook output, or fetch the logs and run `training.logparse` on them.

## Behaviour under failure

- A worker that misses a round gets HTTP 409 and resyncs. Any other error stops the worker with its trace, so the failure is visible in the log.
- The coordinator closes a round when `quorum` deltas arrive, or after `--timeout-minutes` once at least one has arrived.
- Each notebook stops itself at `--max-minutes` (plan default 540), after a final send, before a session limit can kill it mid-round.
- Coordinator state persists, so the global model survives restarts. Optimizer moments do not: a restarted worker begins with a fresh AdamW state and a short warmup.
- Workers need only the coordinator URL. Config, tokenizer, and packed data come from `/v1/files/...`, so no Kaggle dataset is needed.

## Verified here (CPU only)

- The architecture: correct layer pattern, causal attention, tied embeddings, and parameter counts.
- Packing: completion-only loss masks, uniform sharding, bucketed batches.
- The coordinator over HTTP: the staleness window, quorum, fp16 downloads, and idempotent retries.
- Two independent groups, two rounds each, with real outer updates and held-out loss falling after each merge. No tracebacks.
- Two DDP ranks on one group: identical parameter checksums after every merge.
- The probe mode runs and reports finite losses.
- The generated notebooks: pinned library cell, package written, launch cell, and the rehearsal harness running the cells end to end.

## Not verified

- **Anything on a GPU.** The AMP path, fused AdamW, NCCL DDP, and T4 throughput have not run. This machine's NVIDIA driver and NVML library do not match, so no CUDA work has run here.
- **fp16 stability** for this architecture on a T4. Only the probe mode can test it, on real Kaggle hardware.
- Kaggle's session limits and startup times, and how they interact with the quorum.
- Coordinator bandwidth at this model size. A 126M model is about 0.75 GB of traffic per group per round in f32, or about 0.5 GB with fp16 downloads.

## Open decisions

1. **Kaggle's rules on multiple accounts.** I could not verify them. Read Kaggle's terms yourself before pushing 14 accounts. Pooling quota across accounts may breach them.
2. **Quotas and session limits.** The figures I found are secondary: about 30 GPU hours per week per account (user-reported), and a 12-hour session limit.
3. **A public coordinator URL.** Kaggle cannot reach private addresses. Use a tunnel or a small VM.
4. **The keys.** The 14 KGAT keys were pasted into this chat. Treat them as exposed and rotate them. Put the new keys in `kaggle/secrets/keys.env` (gitignored). Nothing here reads or writes them except `push`.
5. **General developer data.** The synthetic data is 11.3M tokens (measured). The SmolLM `python-edu` split ships only blob IDs (about 7.7M files). Its contents must be fetched from Software Heritage's S3 bucket, and I have not measured their size or token count. `bigcode/the-stack-smol` is gated behind an accept-terms step on your Hugging Face account, so I did not download it.
6. **The reference architecture file.** Provide the EmbeddingGemma architecture keys if you have access, to replace the assumed dimensions.
