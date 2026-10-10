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

- **From scratch, per the brief.** The plan fine-tuned Qwen2.5-Coder. This build trains a LLaMA-style decoder from scratch with transformers (RMSNorm, RoPE, SwiGLU, tied embeddings). Its safetensors and config follow the layout that Rust's candle loads.
- **Jobs.** RELOCATE: given the stored fingerprint and up to 8 candidates, answer `{"match": k}` or `{"match": null}`. AUTHOR: given a target and a request, answer DSL ops as JSON.
- **Data.** Synthetic storefront-style pages from four site kinds, with mutations: class renames, id removal, wrapping, moving the target, retext, look-alike decoys, deletion, sibling shuffles, and noise. Splits are by site, so no held-out page appears in training. Deleted targets are capped at 15% of RELOCATE rows. Prompts use the same prefilter, fast path and serialiser as the CLI, and omit element boxes, because training pages have none.
- **Dataset for 20,000 sites** (`pw train gen-data --sites 20000 --variants 6 --author 3 --seed 1`): 46,064 RELOCATE rows (15.0% null), 60,000 AUTHOR rows, and a held-out split of 4,617 RELOCATE and 6,000 AUTHOR rows.
- **Limit.** This data teaches the two jobs. It does not teach general developer knowledge. A general corpus has not been built; see `docs/KAGGLE.md` for the options and the licence and gating constraints.

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

Command: `pw train serve` with three `training.worker` processes, `--inner-steps 40 --rounds 2 --batch 16`. Each worker trains on one third of the 95,447 training rows.

- Inner loss, round 0: 8.12, 8.11, 8.12 across the three workers (about log of the vocabulary size). Round 1: 5.36, 5.36, 5.35.
- The coordinator received three deltas per round, advanced to round 2, and wrote both global checkpoints.
- Evaluated on the first 200 held-out rows per task (`training.eval_checkpoint`):

| Weights | Heldout loss | RELOCATE exact | RELOCATE, non-null | AUTHOR exact |
|---|---|---|---|---|
| Random initialisation | 9.03 | 0% | 0% | 0% |
| After 2 outer rounds (global model) | 2.87 | 79% | 94% | 0% |

The protocol improves the shared model. The run was too short for AUTHOR, and too short to learn nulls. Comparing it with the single-process run is not a like-for-like test. That run took about 300 batched steps, and this one took about 80 inner steps per worker. A same-compute comparison is still to be done.

### 6.4 Test suite

`cargo test --workspace`: 66 `pw-core` unit tests, 13 `pw-proxy` unit tests, 7 proxy end-to-end tests, and 23 `pw-train` tests. All pass.

## 7. Deviations from the plan and the brief

1. **Model from scratch.** Per the brief. The plan's Qwen fine-tune and its quality expectations do not apply.
2. **Heuristic fallback is not trusted for deleted targets.** Section 6.1 shows why. Relocations are provisional until verified, and the release gate enforces that.
3. **Added identity assertions and a release gate.** Neither is in the plan text. Both follow from the plan's own rule that every relocated patch must pass its assertions in preview.
4. **Browser runtime and editor are JavaScript.** Browsers execute JavaScript or WebAssembly. The runtime is small and is the only code visitors run.
5. **Playwright engine is Microsoft's driver.** Driven through `playwright-rs`. The repository's JavaScript is limited to the in-page snippets.
6. **No Rust inference yet.** The plan used `llama-server` and the brief asks for Rust. The next step is to serve the exported weights with candle. Until then the CLI uses only the heuristic.
7. **`hk` (the Zig runtime) is not used.** The brief says Rust.
8. **Kaggle orchestration is planned and gated, not run.** See `docs/KAGGLE.md`.

## 8. Known limits

- Heuristic relocation mis-matches about 18% of deleted targets on the synthetic benchmark. Verify before release.
- The model does not yet answer null, so it cannot protect against deleted targets. This is the most important open item.
- No model is wired into the CLI.
- The proxy buffers whole response bodies, so very large pages use memory.
- When a response carries several CSP headers with different nonces, the runtime tag can carry only the first nonce. This is an edge case, and it is not tested.
- Shadow DOM is not handled. The runtime does not look inside shadow roots, so patches cannot target elements there. Cross-origin iframes and canvas are out of scope.
- `clickNavigatesTo` assertions are reported as not checked. Checking them needs a click, which `pw verify` does not yet perform.
- The audit log's HEAD file is local. See section 4.
- Nothing has run on Kaggle. Its GPU speed and session behaviour are unmeasured.
- The synthetic data does not cover general developer knowledge.

## 9. Next steps, in order

1. Build a development split for threshold tuning, keep the test seed untouched, and re-run `pw train bench` on the test seed.
2. Add candle inference for the exported model and wire it into the CLI as the model chooser. Require the model to answer null with a probability threshold (the plan's logprob gate).
3. Train longer on the synthetic data, and measure the null class on the held-out split.
4. Add the general developer corpus once its licence terms are decided (see `docs/KAGGLE.md`).
5. Make `pw verify` click `clickNavigatesTo` targets.
6. Run the same-compute comparison, single process against the DiLoCo run.
7. Decide on Kaggle, and run the plan as described in `docs/KAGGLE.md`.
