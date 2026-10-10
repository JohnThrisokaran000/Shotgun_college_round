# Kaggle distributed training: runbook and open decisions

## Status

- Implemented: `pw train kaggle plan` writes one kernel folder per account (kernel-metadata.json and a notebook). Each notebook writes the training package and starts one DiLoCo worker per T4, so two workers per account. The plan is offline and contains no credentials.
- Implemented: `pw train kaggle push` is a dry run unless you pass `--execute --confirm-accounts N --i-have-checked-kaggle-rules`.
- Checked: plans with placeholder accounts validate (notebooks parse, every code cell compiles, GPU and internet flags set, no key material in the output).
- **Not run:** no notebook has been pushed or started on any Kaggle account. Nothing has been sent to Kaggle.

## Decisions needed before any push

1. **Kaggle's rules on multiple accounts.** I could not confirm them. Fetching kaggle.com/terms and the notebook docs returned page titles only, and the search results were secondary sources. Read Kaggle's terms yourself. Running 14 accounts to pool their GPU quota may breach them. If it does, the account risk is yours to weigh before you push.
2. **Quotas and session limits.** The figures I found are secondary: about 30 GPU hours per week per account (user-reported), and a 12-hour session limit. Confirm both in the account settings before planning the run length.
3. **Coordinator URL.** Kaggle notebooks must reach the coordinator over public HTTPS. A coordinator on 127.0.0.1 or a LAN address is rejected by `pw train kaggle plan`. Options: a tunnel from your machine, or a small VM. Pick one.
4. **API keys.** The 14 KGAT keys were pasted into this chat. Treat them as exposed and rotate them in Kaggle's settings. Then put the new keys in `kaggle/secrets/keys.env` (gitignored, one `alias=KEY` per line). I have not written any key to disk.
5. **Training data.** The synthetic PatchWork data (about 95k rows, 11.3M tokens measured with the smoke tokenizer) does not cover general developer knowledge. A general corpus is needed. Options checked: `bigcode/the-stack-smol` is gated behind an accept-terms step on your Hugging Face account, so I did not download it. `HuggingFaceTB/smollm-corpus` is ungated under ODC-BY (attribution required). Its `python-edu` split ships only blob IDs, about 7.7M files across two shards. The file contents must be fetched from Software Heritage's S3 bucket with the script on the dataset card. I have not measured that download's size or its token count, so it is not yet a usable corpus. Choose before the run.

## Runbook (after the decisions above)

```sh
# 1. Accounts file (no keys): one alias,username per line
printf 'k01,your-username-1\nk02,your-username-2\n' > kaggle/accounts.txt

# 2. Shared starting point and served files (on a machine with the data)
python -m training.init_model --tokenizer runs/x/tokenizer.json \
    --train-rows data/patchwork/train.jsonl --out runs/dist --hidden 512 --layers 8 --heads 8

# 3. Coordinator, reachable at the public URL you chose
PW_TOKEN=... ./target/release/pw train serve --init runs/dist/init.safetensors \
    --state runs/dist/state --serve runs/dist/serve --listen 0.0.0.0:8765 \
    --workers 28 --quorum 20 --timeout-minutes 45

# 4. Plan, then review kaggle/plan/PLAN.md
./target/release/pw train kaggle plan --accounts kaggle/accounts.txt \
    --coordinator-url https://YOUR-PUBLIC-URL --out kaggle/plan --inner-steps 500 --rounds 40

# 5. Dry run first (prints the commands, sends nothing)
./target/release/pw train kaggle push --plan kaggle/plan

# 6. Push, only once every decision above is made
./target/release/pw train kaggle push --plan kaggle/plan --keys kaggle/secrets/keys.env \
    --execute --confirm-accounts 14 --i-have-checked-kaggle-rules
```

## How the run behaves

- A round closes when `quorum` deltas arrive, or after `--timeout-minutes` once at least one has arrived. A worker that misses a round gets 409 and resyncs. A worker that gets any other error stops with the error text, so failures are visible in the notebook output.
- Each notebook stops itself after `--max-minutes` (the plan default is 540), before a session limit can kill it mid-round. Coordinator state persists in `--state`, so it survives restarts.
- Workers need only the coordinator URL. Config, tokenizer and training rows come from `/v1/files/...`, so no Kaggle dataset is needed.
- Coordinator endpoints: `GET /v1/status`, `GET /v1/global`, `POST /v1/delta`, `GET /v1/files/<name>`. Set `PW_TOKEN` to require a bearer token. Put it in a Kaggle secret named `PW_TOKEN` so it is never in the notebook.

## What was not verified

- Nothing has run on Kaggle, so GPU speed, session behaviour, and Kaggle's network limits are unmeasured.
- The distributed protocol was tested only locally: one coordinator and three CPU workers, two rounds, with the outer step and the held-out comparison reported in `docs/DESIGN.md`.
