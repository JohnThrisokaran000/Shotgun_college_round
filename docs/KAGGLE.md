# Kaggle training: how it runs, how to operate it, and what was verified

## What the setup is

- **Model.** A Gemma-3-style decoder, 126,702,336 parameters (24 layers, hidden 768, 3 heads, 1 KV head, head dim 256, FFN 1152, 32,768-token vocabulary, context 512). Built with transformers' `Gemma3ForCausalLM`. Layers follow a 5:1 sliding-window to full-attention pattern. Trained from scratch.
- **EmbeddingGemma-300M is an architectural reference only.** No weights, tokenizer or pretrained state are used. The "300M" in its name is mostly its 262k-token embedding table. Its config.json is gated, so the dimensions in `training/arch.py` (`TEMPLATE`) are my assumptions. Pass the real keys with `training.prepare --reference config.json` if you have them.
- **Training is DiLoCo-style.** Each Kaggle account (a "group", two T4s, DDP inside the group) trains on its own data shard and, every 2,000 steps, sends an fp16 delta to a coordinator on this machine. The coordinator averages deltas with an outer Nesterov step and serves the new global model, as int8, back to the groups. Communication overlaps with compute.

## Data flow

```
training.corpus.fetch  ->  training.corpus.build  ->  training.shards  ->  training.kaggle_upload  ->  notebooks
 (licence-checked)        (dedupe, held-out         (tokenizer, packed     (one private dataset      mount it at
                           packages)                 shard per account)     per account)              /kaggle/input/
```

- **Corpus**: 2,383 permissively licensed sources, about 2.0M chunks and 2.4 GB of text (about 700M tokens). Licences come from each registry's own metadata. Copyleft (GPL, LGPL, MPL) and share-alike (CC-BY-SA, so MDN) are excluded. `data/corpus/general/LICENSES.md` lists every source.
- **PatchWork synthetic data** (about 20% of training tokens): `pw train gen-data`. Eight page kinds, sixteen mutations. Deleted targets and identical twins are labelled as refusals.
- **Each account trains on its own shard** (about 52M tokens). Held-out data is the same in every shard: whole held-out packages and held-out synthetic sites.
- **The coordinator sends no bulk data.** It sends only the global model (int8, 126 MB) and receives deltas (fp16, 252 MB).

## Things learned the hard way

1. **Kernel metadata keys are snake_case** (`enable_gpu`, `enable_internet`, `dataset_sources`, `machine_shape`). camelCase keys are silently ignored, which starts a CPU-only kernel.
2. **`machine_shape: NvidiaTeslaT4` gives two T4s.** Verified: workers report `world=2`. The Kaggle SDK lists no separate 2xT4 name.
3. **Internet is silently off on accounts that are not phone-verified**, even when the kernel asks for it. Symptom: every pip retry fails with `NewConnectionError`. Check every account first with `training.nettest`. In this run 8 of 19 accounts failed and were purged.
4. **There is no cancel command.** `kaggle kernels delete -y user/slug` stops a kernel. A fresh `push` creates it again.
5. **Queues vary.** A pushed kernel took 1 to 15 minutes to reach RUNNING. Re-pushing every 30 s did not help. Re-pushing every 5 minutes (`training.keep_alive`) did.
6. **An empty `%%writefile` cell is an error** in Kaggle's runner. `__init__.py` gets a one-line body.
7. **Datasets mount at `/kaggle/input/<dataset-slug>`.** The launch cell also searches recursively and prints the tree if nothing is found.
8. **Weekly GPU quota is 30 h per account.** The `kaggle` CLI cannot report it, so check the account's settings page. A 10-hour run uses a third of it.
9. **The link to this machine is about 2.3 MB/s** (measured through the tunnel, both directions). That sets the sync cadence: 11 deltas of 252 MB need about 20 minutes to arrive, so syncs are 2,000 steps (about an hour) apart. The tunnel accepted a 120 MB request body, so Cloudflare's 100 MB limit does not apply to quick tunnels.
10. **fp16 on T4s was healthy.** The probe on real hardware: finite losses, maximum activation 34 at initialisation and 247 after 2,000 steps (the fp16 limit is 65,504).

## Measured throughput

About 4,500 tokens/s on rank 0 and about 9,000 per two-GPU group, using 11.5 GB of 16 GB per GPU. About 0.57 optimizer steps per second at 16 sequences per GPU per step.

## Runbook

All commands run from the repository root.

```sh
# 0. Data (once). Needs the .venv.
python -m training.corpus.fetch --out data/corpus
python -m training.corpus.build --corpus data/corpus --out data/corpus/general
pw train gen-data --out data/pw-v2 --sites 450000 --variants 6 --author 3 --seed 7
python -m training.shards tokenizer --general data/corpus/general --patchwork data/pw-v2 --out runs/tokenizer.json
python -m training.prepare --tokenizer runs/tokenizer.json --out runs/serve \
    --hidden 768 --layers 24 --heads 3 --kv-heads 1 --head-dim 256 --ffn 1152 --sliding-window 512 --ctx 512
python -m training.shards build --general data/corpus/general --patchwork data/pw-v2 --tokenizer runs/tokenizer.json \
    --serve runs/serve --out data/accounts --accounts 14 --accounts-file kaggle/accounts.txt --pw-share 0.2
# This run built 14 shards (the corpus is split into 14), then copied the shards of accounts that had no internet
# into the folders of the accounts added later (acct15, 17, 18, 19), so no two running groups share data.

# 1. Coordinator and tunnel. The token is secret and goes into every notebook. Do not restart the tunnel mid-run:
#    its address changes and the notebooks cannot find it.
. kaggle/secrets/run.env                      # PW_TOKEN=...
pw train serve --init runs/serve/init.safetensors --state runs/state --serve runs/serve \
    --listen 127.0.0.1:8765 --workers 13 --quorum 10 --timeout-minutes 45
# --workers is the number of accounts, shown in the status. It must be at least --quorum. It does not change how rounds close.
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765      # prints https://....trycloudflare.com

# 2. Which accounts have internet? Purge the ones that do not.
python -m training.nettest --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env --url <tunnel url> --purge

# 3. Datasets, one private dataset per account.
python -m training.kaggle_upload --shards data/accounts --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env \
    --execute --confirm-accounts 11 --i-have-checked-kaggle-rules
python -m training.kaggle_ops datasets --accounts kaggle/accounts.txt --keys kaggle/secrets/keys.env --wait 600

# 4. Notebooks, then push. --max-minutes is the training window per notebook.
pw train kaggle plan --accounts kaggle/accounts.txt --coordinator-url <tunnel url> --out kaggle/plan \
    --inner-steps 2000 --sync-every 10 --micro-batch 8 --grad-accum 2 --lr 6e-4 --warmup-steps 200 --max-minutes 600
pw train kaggle push --plan kaggle/plan --keys kaggle/secrets/keys.env --execute --confirm-accounts 11 --i-have-checked-kaggle-rules
# Run the watchdog as a background process with a log, not inside a monitor: a monitor is killed at its
# timeout (30 minutes at most), and the watchdog dies with it. On 2026-10-10 it stopped for about 40 minutes that way.
nohup .venv/bin/python -u -m training.keep_alive --plan kaggle/plan8 --accounts kaggle/active2.txt \
    --keys kaggle/secrets/keys.env --interval 30 --grace 300 --silent 10 >> runs/keep_alive.log 2>&1 < /dev/null & disown

# 5. Watch.
python -m training.status runs/state/events.jsonl --watch 30     # one table
python -m training.live runs/state/events.jsonl                  # event stream

# 6. Finish: evaluate the newest global model and export it.
python -m training.finalize --out runs/final --patchwork data/pw-v2/heldout.jsonl \
    --general data/corpus/general/heldout.jsonl
```

## Behaviour under failure

- A group that crashes is re-pushed by `keep_alive` and rejoins at the coordinator's current global version (it downloads the int8 global).
- A delta refused as too stale (HTTP 409) is not lost. The group merges the newer global and keeps the progress it made since its snapshot, and its next delta carries that progress. The cost is delay.
- A global download of 126 MB per group, with 13 groups fetching at once, takes about 5 minutes through the tunnel. Each merge therefore lands several minutes after the round closes.
- Every delta carries a per-worker sequence number, so a retried upload is never counted twice.
- A delta based on a global version more than one behind is refused (409). The group resyncs and loses only that stretch.
- A round closes when `--quorum` deltas arrive, or `--timeout-minutes` after the round started once at least one has arrived.
- Coordinator state (global model, momentum, round) is written after every round, so the coordinator can restart. Pending deltas, and the groups' optimizer states, are not saved.
- Each notebook stops itself at `--max-minutes`, after sending its last delta.
- A re-push starts a new session, and nothing stops the old one (there is no cancel command). Both sessions' deltas are accepted, because sequence numbers are millisecond timestamps: a later session always sends a higher number. Within one round, a newer delta from an account replaces its older one, so an account never counts twice.
- `training.status` shows the newest event per group. Two live sessions for one account look like one row. The `start` events in `runs/state/events.jsonl` show sessions: each session logs two, one for the probe and one for training.
- A coordinator restart makes uploads fail with 502 for a minute or two. An upload retries four times, 5 to 20 seconds apart. A delta whose upload still fails after four tries is dropped, so it never reaches the global. The group keeps training.
- The coordinator needs about 4 GB of RAM with 10 pending 252 MB deltas, and this machine must stay on for the whole run.

## Outer momentum: what went wrong on 2026-10-10

- The merge that closed round 3 (22:35, version 4) took held-out loss from about 3.2 (version 2) to 7 to 8 on every group. Training loss went from 1.86 (version 3) to 2.26. Version 3 had one held-out reading of 3.5, and it had the lowest training loss of the run.
- The global's norm doubled (524 to 1,023), and it moved 815 away from version 1. Of the squared change, 99.6% is in the 2-D weights, so the norm layers are not the cause.
- Each group's 2,000-step delta has a norm of about 255, which is half the model's norm. With outer momentum 0.9 the coordinator adds the same progress again on every round. The momentum grew to a norm of 366, and each merge overshot.
- Deltas that arrive one version late are still accepted (MAX_STALENESS = 1). With momentum on, that double-counts progress.
- This diagnosis is a hypothesis from these measurements. No ablation has been run.
- Decision, 22:45: roll forward. The coordinator restarted from the current global with `--outer-momentum 0` (learning rate 0.7). The default is now 0 in `pw train serve`. Check held-out loss at the next two merges. If it stays above about 4, roll back to version 1, which is the only older global on disk (`runs/checkpoints/v1-from-acct09.safetensors`).
- Outcome, 23:07: version 5, the first merge with momentum 0, gave held-out loss of 3.24 to 3.47 across the groups. Version 4 gave 6.9 to 8.2. Training loss is about 1.8. The second merge (version 6) is the confirmation.
- The coordinator keeps only the newest global, so versions 2 and 3 are gone. Save a dated copy of each global before the next run.

## Verified

- Unit and end-to-end tests: 137 Rust (`cargo test --workspace`) and 48 Python (`python -m unittest discover -s training/tests -t .`), including the coordinator's staleness window, quorum, int8 download, token checks, and the cross-language hash of the starting weights.
- The CLI with the model (`pw --model runs/final-v1`): an empty project gives an empty result, a missing model folder fails with a clear error, and `check --write` prints the experimental warning.
- A two-notebook, two-GPU-per-group rehearsal on this machine (CPU, gloo): probe, matching starting-weights hash, sends, merges, held-out evals.
- On real Kaggle T4 pairs: dataset mount, DDP across two GPUs, fp16 training, the first sync round, merge, and held-out evaluation.

## Not verified

- Final model quality. The numbers appear in the model card that `training.finalize` writes.
- Scores on real websites. All PatchWork data is synthetic.
- Kaggle's multi-account terms. I could not read them, and the account owner decided to proceed. Treat the 19 keys that were pasted into chat as exposed and rotate them.
- Relocation quality with the model. The Rust decoder matches Python on the v1 checkpoint, and `pw --model` runs end to end. But the v1 model mis-matches deleted targets far more often than the heuristic (DESIGN.md, section 6.5).
