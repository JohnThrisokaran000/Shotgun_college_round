# PatchWork: design, decisions, measured results, limits

Source plan: the "PatchWork — Planning Report" artifact (16-hour plan, four lanes). This document records where the build follows the plan, where it changed, and what was measured. Numbers here come from commands in this repository, and each one can be reproduced.

## 1. What the brief asks for, and what was built

| Brief feature | Built | Verified how |
|---|---|---|
| Visual patching | `pw edit` (headed Chromium overlay, Save writes a validated patch) | Save path not exercised interactively; the overlay and save binding are in `runtime/editor.js` and `crates/pw-cli/src/browser.rs` |
| Smart element tracking | Multi-locator anchors, fast path, prefilter, similarity, model-or-heuristic chooser | Benchmark below; unit tests in `pw-core` |
| Collaborative editing | Patches are YAML files in git-friendly folders; `pw merge` detects hard and soft conflicts | `pw-core` conflict tests (write-write, behaviour, structural, anchor dependency, order dependence) |
| Isolated preview | Proxy channels selected by cookie (`/__pw/channel?name=preview/x`) | Proxy end-to-end tests |
| Automated verification | `pw verify` runs assertions in Chromium through the proxy | Ran against a real page: both assertions passed, one of which proves the runtime applied the patch |
| Versioned deployment and rollback | Immutable content-addressed releases, channel pointers, `pw rollback`, beacon-driven auto-rollback | Proxy e2e test trips the policy and checks the channel and audit entry |
| Audit trail | Hash-chained JSONL with HEAD file; `pw log`, `pw audit-verify` | Tamper and truncation tests |

## 2. Architecture

```
developer machine                                   visitors
-----------------                                   --------
pw edit / pw snapshot ──► Playwright (Chromium)
pw check / pw release ──► pw-core (relocate, DSL, conflicts, releases)
                                  │ release manifests (JSON, immutable)
                                  ▼
                          pw proxy (pw-proxy) ◄──── browser ──► upstream site
                                  │ inject runtime.js, /__pw/manifest, /__pw/beacon
                                  ▼
                          beacon window ──► auto-rollback ──► channel pointer + audit
```

- `pw-core` has no I/O besides the snapshot, patch and release types. It is deterministic. Its only optional model hook is the `Chooser` trait.
- `pw-proxy` is the only component in the request path. It decodes gzip, deflate and brotli, injects one script tag, and adjusts CSP for that script and for beacons. Anything it cannot decode passes through unchanged.
- The runtime (`runtime/runtime.js`, about 7.5 KB against the 10 KB budget) applies patches idempotently, so re-running after an SPA re-render changes nothing. It reports `ok`, `missing` and `error` beacons. Only `error` counts toward rollback.

## 3. The re-anchoring pipeline

1. **Fast path.** Each stored locator (CSS, XPath, testid, role plus name) resolves to a unique element. If at least two agree and the fingerprint similarity is at least 0.9, the patch is `valid`. No scoring is needed.
2. **Prefilter.** Compatible elements (button-like tags, or any element with `role=button`), scored by a Similo-style weighted similarity. The top 8 go forward.
3. **Chooser.** Either the heuristic or the model returns `Pick`, `Unsure` or `Null`. `Pick` is `relocated`. `Unsure` is `ambiguous` (a human picks, and the top three are shown). `Null` is `broken`.

Similarity weights (proposal, see section 6): tag 2, id 2, name 2, class 1, text 2, other attributes 1, XPath 1, ancestors 1 (compared by tag), neighbour text 2, rectangle 1. An id or class the candidate no longer has counts as no evidence, because redesigns often drop them. A different id scores 0. Other stored attributes the candidate lacks score 0.

The heuristic accepts a candidate only if it scores at least 0.6, leads the runner-up by at least 0.15, scores at least 0.5 on the floor, and keeps at least half of its stored surroundings (exact ancestor labels or neighbour texts). The surroundings check exists because tag, text and class alone can match a look-alike elsewhere on the page.

## 4. Safety rules

- **DSL validation** (`pw-core/src/dsl.rs`). Only the declared ops exist. `script` is rejected with an explicit message. Style properties are an allowlist, and values cannot contain `;`, braces, `url(`, `expression(` or `javascript:`. Attributes starting with `on` and `style`/`srcdoc` are refused. URLs must be same-site paths. `setHTML` and `insertBefore`/`insertAfter` content is sanitised with ammonia. Selectors cannot target `html`, `head` or `body`.
- **Identity assertion.** Every patch created by `pw edit` carries `textContains` on the element's text, or on the new text when the patch sets it. A relocation that lands on a look-alike with different text fails this in preview. Section 6 measures how often that catches a wrong match.
- **Release gate.** `pw release` refuses to publish if any patch is broken, ambiguous, or conflicts hard. It also refuses heuristic relocations unless `--accept-relocated` is given after `pw verify` on a preview channel.
- **Auto-rollback.** A release's channel rolls back to its parent when at least 5 error beacons from at least 3 sessions arrive within 60 seconds. Each release trips at most once, and the rollback is written to the audit log.
- **Audit.** Each event hashes its body and the previous hash. Editing a line or truncating the tail is detected. The HEAD file is local, so anyone who can rewrite both files can forge the log. Commit HEAD somewhere else if you need that protected.

## 5. Model and data

- **From scratch, Gemma-3-style architecture.** The plan fine-tuned Qwen2.5-Coder. This build trains from scratch a Gemma-3-style decoder (grouped-query attention, QK-norm, pre- and post-RMSNorm, GeGLU, separate local and global RoPE bases, and a 5:1 sliding-window pattern). EmbeddingGemma-300M is used only as an architectural reference. Its weights, tokenizer, and pretrained state are not used, and transfer learning is not used. Its config is gated, so the dimensions are assumptions (see `docs/KAGGLE.md`).
- **Jobs.** RELOCATE: given the stored fingerprint and up to 8 candidates, answer `{"match": k}` or `{"match": null}`. AUTHOR: given a target and a request, answer DSL ops as JSON.
- **Data.** Synthetic storefront-style pages from four site kinds, with mutations: class renames, id removal, wrapping, moving the target, retext, look-alike decoys, deletion, sibling shuffles, and noise. Splits are by site, so no held-out page appears in training. Deleted targets are capped at 15% of RELOCATE rows. Prompts use the same prefilter, fast path and serialiser as the CLI, and omit element boxes, because training pages have none.
- **PatchWork synthetic data** (`pw train gen-data --sites 450000 --variants 6 --author 3 --seed 7`): eight page kinds (checkout, product, login, dashboard, table, search results, settings form, modal), sixteen mutations, multi-step authoring requests, and refusals for deleted targets and for identical twins. About 20% of the training tokens.
- **General data.** A licence-checked corpus of 2,383 sources (npm, PyPI, crates.io, GitHub), about 700M tokens. Copyleft and share-alike sources are excluded. See `docs/KAGGLE.md`.

## 6. Measured results

All numbers are reproducible from the commands shown. Dates are from this session (2026-10-10).

### 6.1 Re-anchoring benchmark, held-out sites

Command: `pw train bench --sites 200 --variants 6 --seed 7`. These are 200 unseen synthetic sites with 6 mutated variants each. The 961 "present" and 239 "deleted" cases come from 1,200 variants.

| Method | Target present (961) | Target deleted (239) |
|---|---|---|
| Heuristic (`pw check`) | correct 84.6%, wrong match 0.73%, abstain 14.7% | wrong match **17.6%** (42 cases), abstain 82.4% |
| Selector-only baseline | correct 92.9%, wrong match 3.85%, abstain 3.2% | wrong match 29.3% |

A text-identity assertion would fail 26 of the 42 deleted-target wrong matches, and 1 of the 7 present-target ones. Those are the cases it would catch when `pw verify` runs on a preview channel. The other 16 pass the identity check.

What went wrong: the heuristic relocates a deleted "Buy now" onto a sibling action, such as "Continue shopping", in the same container. Structure matches, and the text score is too small to outweigh it. The heuristic cannot tell "deleted" from "re-texted" on structure alone. The RELOCATE model is meant to learn exactly this distinction. This is why heuristic relocations must be verified before release.

I did not tune the thresholds against this benchmark. Doing that on the same held-out set would be leakage. A separate development split is the next step.

### 6.2 Model, single process, CPU

Command: `training.train --hidden 128 --layers 2 --heads 4 --steps 300 --batch 16 --lr 1e-3` on the 20,000-site dataset. Model: 1.43M parameters. Evaluated on the first 100 held-out rows per task.

- Heldout loss on completion tokens: 0.0776.
- RELOCATE exact match: 81%. On non-null answers it is 97.6%. On null answers it is 0%, because the model never predicts null in this short run.
- AUTHOR exact match: 84%.

This is a smoke test, not a quality claim. The null class is the main gap, and it matters most for safety.

### 6.3 Distributed training (DiLoCo), CPU, 3 workers

Command: `pw train serve` with three synchronous workers (the earlier `training/worker.py`, since replaced by `training/group_worker.py`), `--inner-steps 40 --rounds 2 --batch 16`. Each worker trains on one third of the 95,447 training rows.

- Inner loss, round 0: 8.12, 8.11, 8.12 across the three workers (about log of the vocabulary size). Round 1: 5.36, 5.36, 5.35.
- The coordinator received three deltas per round, advanced to round 2, and wrote both global checkpoints.
- Evaluated on the first 200 held-out rows per task (`training.eval_checkpoint`):

| Weights | Heldout loss | RELOCATE exact | RELOCATE, non-null | AUTHOR exact |
|---|---|---|---|---|
| Random initialisation | 9.03 | 0% | 0% | 0% |
| After 2 outer rounds (global model) | 2.87 | 79% | 94% | 0% |

The protocol improves the shared model. The run was too short for AUTHOR, and too short to learn nulls. Comparing it with the single-process run is not a like-for-like test. That run took about 300 batched steps, and this one took about 80 inner steps per worker. A same-compute comparison is still to be done.

### 6.4 Test suite

`cargo test --workspace` runs 137 Rust tests and all pass: 66 `pw-core` unit tests, 1 `pw-cli` unit test, 7 `pw-model` unit tests and 4 model fixture tests, 13 `pw-proxy` unit tests, 7 proxy end-to-end tests, 33 `pw-train` unit tests and 6 coordinator end-to-end tests. The Python training tests (`python -m unittest discover -s training/tests -t .`) run 48 tests, all passing.

### 6.5 Model relocation, v1 checkpoint, 30 held-out sites

Command: `pw train bench --sites 30 --variants 3 --seed 7 --model runs/final-v1`. The model is the version 1 export (126.7M parameters, from the first merged checkpoint of the Kaggle run). There are 72 "present" cases and 18 "deleted" cases.

| Method | Target present (72) | Target deleted (18) |
|---|---|---|
| Heuristic (`pw check`) | correct 69.4%, wrong match 2.8%, abstain 27.8% | wrong match 0%, abstain 100% |
| Selector-only baseline | correct 86.1%, wrong match 2.8%, abstain 11.1% | wrong match 22.2%, abstain 77.8% |
| Model alone (`--model`) | correct 86.1%, wrong match **12.5%**, abstain 1.4% | wrong match **55.6%**, abstain 44.4% |
| Model, then heuristic | same counts as model alone | same counts as model alone |

For deleted targets the right outcome is an abstention, so the "correct" column is always 0% there.

- On present targets the model matches the selector (86.1%) and is about 17 points above the heuristic. It also makes 9 wrong matches where the heuristic makes 2, and it answers almost every case.
- On deleted targets the model mis-matches 10 of 18. This is the failure section 6.1 describes, and this checkpoint does not fix it.
- The model-then-heuristic rows equal the model-alone rows, so the heuristic fallback changed no outcome in this run.
- The sample is small. 72 and 18 cases give wide intervals, so a difference of 2 against 9 wrong matches is a signal, not a settled result. All the data is synthetic. The final model may do better than the v1 checkpoint.

Decision for v1: the heuristic stays the default. `--model` is opt-in. It prints a warning when used with `check --write` or a release that publishes relocations.

#### Choosing the confidence bar

`model_accept` (0.8) was not tuned on the v1 benchmark, because tuning there would leak into the test. The protocol below was written before the development run.

- Development split: seed 101, 300 sites, 3 variants. It is used only to choose the bar.
- Bars tried: 0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999.
- Rule: choose the lowest bar at which model-then-heuristic is no worse than the heuristic on both measures, on the development split. "No worse" means its deleted-target wrong-match rate is at most the heuristic's, and its present-target correct rate is at least the heuristic's. If no bar meets both, the model stays off by default at every bar.
- Tests, each run once at the chosen bar: seed 7 (the v1 set, 30 sites, 3 variants) and seed 202 (300 sites, 3 variants, never used before).
- Adoption, also written before the tests: the model becomes the recommended way to use `--model` only if, on both test runs at the chosen bar, model-then-heuristic has (a) fewer deleted-target wrong matches than the heuristic, (b) no more present-target wrong matches, and (c) no fewer present-target correct answers. Otherwise it stays opt-in and is documented as not recommended. Either way, the chosen bar becomes the `model_accept` default that `--model` uses.

`pw train bench --accept` runs this sweep. The model is asked once per case, and each bar is applied to the same answers. At the default bar, the sweep's rows equal the direct run's rows, which was checked on a small sample.

Development result, seed 101 (693 present and 207 deleted cases). The heuristic on the same cases: present correct 70.7%, present wrong 1.44%, deleted wrong 9.18% (19 cases).

| Bar (model then heuristic) | Present correct | Present wrong | Present abstain | Deleted wrong | Deleted abstain |
|---|---|---|---|---|---|
| 0.00 to 0.85 | 89.8% | 10.10% | 0.1% | 84.1% | 15.9% |
| 0.90 | 89.6% | 10.10% | 0.3% | 74.9% | 25.1% |
| 0.95 | 81.5% | 5.34% | 13.1% | 31.4% | 68.6% |
| 0.98 | 74.3% | 3.46% | 22.2% | 12.6% | 87.4% |
| **0.99** | **70.7%** | **1.44%** | **27.8%** | **7.2%** | **92.8%** |
| 0.995 | 70.6% | 1.44% | 28.0% | 7.2% | 92.8% |
| 0.999 | 70.6% | 1.44% | 28.0% | 7.2% | 92.8% |

Bar 0.99 is the lowest bar that meets the rule. At 0.98 the deleted-target wrong-match rate is above the heuristic's. At 0.99 it is 15 cases instead of 19, and the present-target rates equal the heuristic's. The model alone at 0.99 answers only 47.8% of present targets correctly, against 70.7% for the heuristic. So at this bar the model acts mainly as a filter on deleted targets. The gain is four cases out of 207, which is within noise. That is why the test runs matter.

#### Test results at bar 0.99

Test seed 7 (the v1 set: 30 sites, 3 variants; 72 present and 18 deleted cases). Running the sweep at 0.8 reproduces the v1 model rows exactly, which checks the replay.

| Method | Present correct | Present wrong | Deleted wrong |
|---|---|---|---|
| Heuristic | 69.4% | 2.78% | 0.00% |
| Model then heuristic, bar 0.99 | 69.4% | 2.78% | 0.00% |
| Model alone, bar 0.99 | 43.1% | 0.00% | 0.00% |

Test seed 202 (never used before: 300 sites, 3 variants; 696 present and 204 deleted cases).

| Method | Present correct | Present wrong | Deleted wrong |
|---|---|---|---|
| Heuristic | 73.3% | 1.15% | 6.86% (14 cases) |
| Model then heuristic, bar 0.99 | 73.0% | 1.29% | 6.9% (14 cases) |
| Model alone, bar 0.99 | 48.0% | 0.43% | 3.4% |

Adoption criteria not met. On seed 7 the heuristic makes no deleted-target wrong matches, so the model cannot make fewer. On seed 202 the model makes the same number of deleted-target wrong matches, one more present-target wrong match (9 against 8), and two fewer correct answers (508 against 510). The development gain (19 to 15 deleted wrong matches) did not replicate. The model stays opt-in and is not recommended, and the heuristic stays the default. Bar 0.99 is the default `model_accept`, as the protocol says, because 0.8 mis-matches far more deleted targets. Deleted targets remain an open problem: the heuristic mis-matches 6.9% of them on seed 202, and 17.6% on the earlier 200-site run.

## 7. Deviations from the plan and the brief

1. **Model from scratch, architecture from a reference only.** Per the brief. The plan's Qwen fine-tune and its quality expectations do not apply. EmbeddingGemma-300M supplies the layer design only. Its weights and tokenizer are not used.
2. **Heuristic fallback is not trusted for deleted targets.** Section 6.1 shows why. Relocations are provisional until verified, and the release gate enforces that.
3. **Added identity assertions and a release gate.** Neither is in the plan text. Both follow from the plan's own rule that every relocated patch must pass its assertions in preview.
4. **Browser runtime and editor are JavaScript.** Browsers execute JavaScript or WebAssembly. The runtime is small and is the only code visitors run.
5. **Playwright engine is Microsoft's driver.** Driven through `playwright-rs`. The repository's JavaScript is limited to the in-page snippets.
6. **Rust inference is in place.** `crates/pw-model` runs the exported model with candle (CPU, f32), using candle's Gemma 3 implementation. It is checked against the Python reference: on the trained checkpoint the top five tokens match and the logits differ by at most 3e-6. `pw --model DIR` makes the model the first chooser for relocation. It decides when it is confident (`model_accept`, now 0.99, chosen in section 6.5) and the heuristic decides otherwise. A confident refusal is final. The benchmark (`pw train bench --model DIR`) scores the model against the heuristic on the same held-out cases.
7. **`hk` (the Zig runtime) is not used.** The brief says Rust.
8. **Kaggle training runs on 13 accounts (26 T4s).** Each account is one group: DDP across its two GPUs, its own data shard, DiLoCo sync through a coordinator on this machine behind a Cloudflare tunnel. Details, measured throughput and the problems hit (silently disabled internet, camelCase metadata, no cancel) are in `docs/KAGGLE.md`.

## 8. Known limits

- Heuristic relocation mis-matches about 18% of deleted targets on the synthetic benchmark. Verify before release.
- The model does not yet answer null, so it cannot protect against deleted targets. This is the most important open item.
- The outer momentum of the DiLoCo run (0.9) overshot on 2026-10-10, and held-out loss rose from about 3.2 to 7 to 8. The coordinator now defaults to momentum 0, and the run was restarted from the damaged global to see whether it recovers. See docs/KAGGLE.md, "Outer momentum".
- The model is wired into the CLI behind `--model`. The heuristic remains the default. On deleted targets the v1 checkpoint is much worse than the heuristic (section 6.5), so it should not become the default yet.
- The proxy buffers whole response bodies, so very large pages use memory.
- When a response carries several CSP headers with different nonces, the runtime tag can carry only the first nonce. This is an edge case, and it is not tested.
- Shadow DOM is not handled. The runtime does not look inside shadow roots, so patches cannot target elements there. Cross-origin iframes and canvas are out of scope.
- `clickNavigatesTo` assertions are reported as not checked. Checking them needs a click, which `pw verify` does not yet perform.
- The audit log's HEAD file is local. See section 4.
- Final model quality is not known until the run ends. Scores on real websites have not been measured: all PatchWork data is synthetic.
- The general corpus is about 700M tokens, so a long run repeats it several times.

## 9. Next steps, in order

1. Done (section 6.5): a development split (seed 101) chose the bar, and two test seeds checked it. The model did not beat the heuristic on either. The next step for deleted targets is a model that answers null, or a better heuristic, not a threshold.
2. Done: candle inference, the model chooser behind `--model`, and a confidence gate for both picks and refusals.
3. Train longer on the synthetic data, and measure the null class on the held-out split.
4. After the run: `pw --model runs/final pw train bench --model runs/final`, and read the model-then-heuristic rows against the heuristic rows.
5. Make `pw verify` click `clickNavigatesTo` targets.
6. Run the same-compute comparison, single process against the DiLoCo run.
7. After the run: `python -m training.finalize`, read the model card, and compare against `runs/final-v1` (the first merged checkpoint).
8. Measure the GPU paths: fp16 autocast, fused AdamW, NCCL DDP inside a group, and T4 tokens per second.
