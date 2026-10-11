# PatchWork: design, decisions, measured results, limits

Source plan: the "PatchWork — Planning Report" artifact (16-hour plan, four lanes). This document records where the build follows the plan, where it changed, and what was measured. Numbers here come from commands in this repository, and each one can be reproduced.

## 1. What the brief asks for, and what was built

| Brief feature | Built | Verified how |
|---|---|---|
| Visual patching | `pw edit`: a Chromium overlay that changes text, background, text colour, font size, classes, a link target, visibility, and position (move before or after another element). Save writes a draft patch | An automated test runs the real overlay script in Chromium and checks the patch it writes (`cargo test -p pw-cli -- --ignored`). Headed use is not tested |
| Smart element tracking | Multi-locator anchors, fast path, prefilter, similarity, a heuristic chooser with a needs-a-person band, and a matcher experiment | Sections 6.1, 6.6, 6.7 and 6.9 (synthetic benchmark only); unit tests in `pw-core` |
| Collaborative editing | Patches are YAML files. `pw review` moves a patch through draft, proposed, approved and rejected, and each step is audited. `pw merge --write` copies another branch's new patches in | `pw-core` conflict tests; `crates/pw-cli/tests/gate.rs` (self-review refusal, approval withdrawn by an edit, merge conflicts) |
| Isolated preview | Proxy channels selected by cookie (`/__pw/channel?name=preview/x`) | Proxy end-to-end tests |
| Automated verification | `pw verify` runs each assertion in Chromium through the proxy, writes a verification record, and compares the candidate page with production outside the patched elements (visual check). `--interact` runs click checks | Section 6.8: sample page, real browser; the assertion, click, and visual results are reproduced there |
| Versioned deployment and rollback | Immutable content-addressed releases, channel pointers, `pw rollback`, beacon-driven auto-rollback | Proxy e2e test trips the policy and checks the channel and audit entry |
| Audit trail | Hash-chained JSONL with HEAD file, now recording who acted (`--actor`, `PW_ACTOR`, or the login name) for edits, reviews, verifications, merges, relocations, releases, waivers, rollbacks and blocked releases | Tamper and truncation tests; section 6.8 |

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
- **Release gate.** `pw release` refuses to publish if any patch is broken, ambiguous, or conflicts hard. It also refuses heuristic relocations unless `--accept-relocated` is given. For a protected channel (`production` by default, set in `pw.toml`) it also requires two things. Every patch must be approved, and the approval must be in the audit log for the patch's current content (`crates/pw-core/src/review.rs`). A hand-edited `status: approved` does not count. The newest verification record for the same patch behaviour must be complete: every check ran, nothing failed, and the visual check was included. A waiver (`--waive-review`, `--waive-verification`) lets one of these through, with a reason, and the waiver is written to the audit log. `pw release --dry-run` lists every reason a real release would be blocked, and exits non-zero when there is one.
- **Auto-rollback.** A release's channel rolls back to its parent when at least 5 error beacons from at least 3 sessions arrive within 60 seconds. Each release trips at most once, and the rollback is written to the audit log.
- **Audit.** Each event hashes its body and the previous hash. Editing a line or truncating the tail is detected. The HEAD file is local, so anyone who can rewrite both files can forge the log. Commit HEAD somewhere else if you need that protected. The actor on each event is declared by whoever runs the command (`--actor`, `PW_ACTOR`, or the login name). It is not authenticated, so the log records who claimed the action.

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

### 6.6 Tuning the heuristic (the model-free relocation chooser)

The protocol below was written before the tuning run. `pw train bench --set` applies the settings.

- Objective: J = present-target correct rate − 5 × (present-target wrong rate + deleted-target wrong rate). A wrong match silently moves a patch to the wrong element, and an abstention is the safe failure. So one wrong match costs five correct answers.
- Tuning split: seed 101, 300 sites, 3 variants (the same development split as section 6.5). Only this split chooses values.
- Settings searched (`pw_core::relocate::Config`): fast_threshold {0.80, 0.85, 0.90, 0.93, 0.95, 0.97}; prefilter_k {4, 6, 8, 12, 16, 24}; floor {0.3, 0.4, 0.5, 0.6}; heuristic_accept {0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}; heuristic_margin {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3}; context_min {0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.
- Method: coordinate descent from the defaults, two passes. Each setting moves to the value with the highest J on the tuning split. Ties keep the current value.
- Test seeds: 7 (30 sites) and 202 (300 sites). Each is run once with the defaults and once with the tuned settings.
- Adoption: the tuned settings become the defaults only if, on both test seeds, (a) J is higher than with the defaults, (b) the deleted-target wrong-match rate is no higher, and (c) the present-target correct rate is at most 2 points lower. Otherwise the defaults stay.
- Not searched in this pass: the similarity weights (constants in `crates/pw-core/src/similarity.rs`). They are a separate, second pass.

Result of the first pass (run 2026-10-10, 23:33):

| Run | Settings | J | Present correct | Present wrong | Deleted wrong |
|---|---|---|---|---|---|
| Dev, seed 101 | defaults | 0.176 | 70.7% | 1.44% | 9.18% |
| Dev, seed 101 | tuned | 0.370 | 73.9% | 1.59% | 5.80% |
| Test, seed 7 | defaults | 0.556 | 69.4% | 2.78% | 0.00% |
| Test, seed 7 | tuned | 0.486 | 69.4% | 4.17% | 0.00% |
| Test, seed 202 | defaults | 0.332 | 73.3% | 1.15% | 6.86% |
| Test, seed 202 | tuned | 0.469 | 75.4% | 1.29% | 4.41% |

The tuned settings are fast_threshold 0.8, heuristic_accept 0.7 and context_min 0.0. prefilter_k 8, floor 0.5 and heuristic_margin 0.15 did not move. On seed 202 the tuned settings meet all three criteria. On seed 7 they add one wrong match in 72 present targets (3 against 2), which fails criterion (a) by that one case. So the adoption rule is not met, and the defaults stay. The `--set` option stays, so the tuned settings can be reproduced and tested.

Second pass (pre-registered before it ran): the ten similarity weights (`W_TAG` … `W_RECT` in `similarity.rs`), with the thresholds at their defaults. Each weight is searched over {0.5, 1.0, 1.5, 2.0, 3.0}. The objective J, the dev split (seed 101), the two-pass coordinate descent and the adoption rule on seeds 7 and 202 are the same as above. The defaults are the comparison. If adopted, the weights are adopted alone, without the first pass's thresholds.

Result of the second pass (run 2026-10-10, 23:36). Five weights moved, and the other five stayed at their defaults: tag 2 → 0.5, id 2 → 3, class 1 → 1.5, text 2 → 1.5, xpath 1 → 0.5.

| Run | Weights | J | Present correct | Present wrong | Deleted wrong |
|---|---|---|---|---|---|
| Dev, seed 101 | defaults | 0.176 | 70.7% | 1.44% | 9.18% |
| Dev, seed 101 | tuned | 0.383 | 77.6% | 1.59% | 6.28% |
| Test, seed 7 | defaults | 0.556 | 69.4% | 2.78% | 0.00% |
| Test, seed 7 | tuned | 0.583 | 79.2% | 4.17% | 0.00% |
| Test, seed 202 | defaults | 0.332 | 73.3% | 1.15% | 6.86% |
| Test, seed 202 | tuned | 0.391 | 77.4% | 1.29% | 6.37% |

Both test seeds pass all three criteria, so the tuned weights are adopted as the defaults in `similarity.rs`. The gains are mostly more correct answers: about 29 more on seed 202. The costs are small but real. Present wrong matches rise by one case on each test set (3 against 2 on seed 7, 9 against 8 on seed 202). Deleted-target wrong matches fall by one case on seed 202 and are unchanged on seed 7. The safety gain on deleted targets is therefore small, and the claim rests on the correct-answer gain, which the objective counts. Real websites are not measured, and this is synthetic data.

### 6.7 Matcher model: protocol fixed before the test seeds

Status: written 2026-10-11, before any matcher result on a test seed. A first run on leaked data was invalid (6.7a). The corrected run, and its decision, are in 6.7b.

What it is. A cross-encoder of 1.6 million parameters, trained from scratch by `training/match.py` on 24,486 pairs generated from seed 5001 (`pw train match-data`: 2,000 sites, 3 variants per site, 8 prefilter candidates per variant; 22% of pairs are positive, and 10.6% of variants have a deleted target). Its text comes from `pw_core::describe`, the same code that writes the training pairs and that the chooser uses at run time. It is a separate experiment. The language model trained on Kaggle is not changed, and it is not used here.

Why it is worth one more measurement. The heuristic cannot relocate the case the report describes: an element whose id is dropped, whose text changes and which moves. On 900 such cases (seed 303, `--drift compound`) it relocates 33% correctly, makes 3 wrong matches, and sends 66% to a person (section 6.6). A learned match score on structure and text is the one change that could address that case.

The protocol, fixed in advance:

1. The three thresholds (`matcher_accept`, `matcher_margin`, `matcher_review`) are chosen on seed 101 only, with `pw train bench --matcher models/matcher-v2 --matcher-grid --sites 300 --variants 3 --seed 101`. The grid is `matcher_grid()` in `crates/pw-train/src/bench.rs`: accept 0.3 to 0.9, margin 0.0 to 0.4, review 0.1 to 0.4. The setting is chosen by `select_setting()`: the highest objective (correct present minus five times the wrong matches), then the most correct present matches, then the least work for a person, then the earlier grid row.
2. The selected thresholds are then run once, unchanged, on seed 7 (30 sites), seed 202 (300 sites), and on the compound drift with seed 303 (300 sites). Three variants per site throughout.
3. The matcher becomes the default chooser only if all three hold: (a) its objective is at least the heuristic's on seeds 7 and 202; (b) its present wrong matches do not exceed the heuristic's on seeds 7 and 202; (c) on seed 303 its correct rate is at least ten points above the heuristic's.
4. Otherwise the matcher stays opt-in behind `--matcher`, and this section says so.

Reference numbers for the heuristic under the current defaults (review_floor 0.45): seed 7, 57 of 72 present correct, 3 wrong, 0 deleted wrong, objective 42. Seed 202, 539 of 696 present correct, 9 wrong, 13 deleted wrong, objective 429. Seed 303 compound, 301 of 900 present correct (33.4%), 3 wrong.

Data separation. Seed 5001 is used only for the matcher's training data. The language model's data on Kaggle was generated with seed 7 (`pw train gen-data --seed 7`). Its first site is identical to site 0 of the benchmark's seed 7, because both start from the same random state. The later sites differ, since the random draws diverge after the first site, so the overlap is one site in about 450,000. It is recorded here, and seed 7 is kept out of any future evaluation of the language model.

### 6.7a Correction: the first matcher run was invalid (leak found and fixed, 2026-10-11)

The first matcher run described every attribute of each candidate, and the generator marks the gold element with `data-pwgold="1"`. So the candidate text carried the answer whenever the gold element survived. The first run's numbers (for example 97% and 100% correct on the compound drift) came from that marker. They are invalid, and they are not reported as results. The first model is kept as `models/matcher-v1-leaked` and is not used.

What was fixed.

1. Relocation snapshots no longer contain the marker. `synth::strip_gold` removes it after the gold index has been read from the marked page. The benchmark (`bench::cases`) and the matcher data (`match_data`) both use it. The regression test `relocation_never_sees_the_gold_marker` checks it.
2. Candidate descriptions are built from the same fingerprint fields as anchors (`describe::element_text` is `fingerprint_text(Fingerprint::from_element(el))`), so the two sides cannot disagree about which attributes exist.

The heuristic was never affected. Its fingerprint keeps only the attributes in `KEPT_ATTRS` (`pw-core/src/anchor.rs`), and the marker is not one of them. Its numbers are the same before and after the fix: the development sweep at every review floor, and seeds 7, 202 and 303.

The protocol in items 1 to 4 is run again on the corrected data, with the same procedure and the same seeds. The thresholds from the invalid run are not reused. The results are in the next subsection.

### 6.7b Corrected matcher run and decision (2026-10-11)

The protocol of items 1 to 4 was run again on the corrected data, with the model `models/matcher-v2`. It was trained on CPU, for six epochs, on the leak-free pairs, and it has 1,586,305 parameters. Its validation numbers come from the training distribution, so they are not a test result. The final epoch scored 0.90 top-1 on present targets, and it refused 0.68 of absent targets.

Development, seed 101. The grid selected accept 0.6, margin 0.4 and review 0.4, with an objective of 248. The heuristic's objective on the same seed is 418. Every setting near the top of the grid makes 16 wrong matches on present targets, against 11 for the heuristic.

Test seeds, with the selected thresholds, unchanged. The heuristic's numbers are in brackets.

| Slice | Present correct | Present wrong | Deleted wrong | Objective |
|---|---|---|---|---|
| Seed 7 (30 sites) | 48 of 72, 66.7% (57, 79.2%) | 3 (3) | 4 (0) | 13 (42) |
| Seed 202 (300 sites) | 510 of 696, 73.3% (539, 77.4%) | 17 (9) | 33 (13) | 260 (429) |
| Seed 303, compound drift (300 sites) | 814 of 900, 90.4% (301, 33.4%) | 2 (3) | 0 (0) | 804 (286) |

Outcome of the rule fixed in advance. (a) The objective is at least the heuristic's on seeds 7 and 202: false. (b) Present wrong matches do not exceed the heuristic's on seeds 7 and 202: false, because seed 202 has 17 against 9. (c) The compound correct rate is at least ten points above the heuristic's: true, by 57 points. The matcher is therefore not adopted. It stays opt-in, as `pw train bench --matcher DIR`, and `pw check` and `pw release` do not use it.

What the result shows. On compound drift the matcher relocates targets that the heuristic cannot, which is the case the report named. On mixed drift it makes more wrong matches, on present and on deleted targets, and the heuristic is better overall. A combined rule, the heuristic first with the matcher only for the targets the heuristic sends to a person, might keep both gains. That rule was not fixed in advance, so it cannot be claimed from these numbers. Testing it needs a new protocol, with its own seeds, written before any run.

### 6.8 Review, verification and the release gate, sample page (2026-10-11)

Setup: a sample checkout page served locally; one patch (`checkout-button-text`, setText "Checkout now") anchored from that page; the proxy on 127.0.0.1:8787; Chromium 153. Every step below was run with the `pw` binary from this repository. The commands are the ones in section 9 of `docs/ALIGNMENT.md`, updated for the new flags.

- Review. The author could not approve their own patch without `--allow-self-review`. A second actor's approval was recorded in the audit log with `self_review: false`. `pw review list` then showed the patch approved.
- Blocked dry run. Before approval, `pw release --channel production --dry-run` listed two reasons: the patch was not approved, and there was no verification record. It exited non-zero and published nothing.
- Edit after approval. Changing the text withdrew the approval: the list showed `NOT VALID: the patch changed after it was approved`. The production dry run then listed both the review reason and the verification reason. Restoring the file restored the approval.
- Verification. A preview release, verified with assertions only, gave an incomplete record (exit 1). With the visual check, the record was complete: two checks passed, and the visual check reported 0 of 919,479 pixels outside the patched element as different. Production had no release yet, so the baseline was the unpatched page.
- Production release. It passed the gate, and the audit event carries the time of the verification it relied on. `pw audit-verify` passed.
- Layout shift. A patch that hides the first element of the cart moves the content below it. The visual check found 508 differing pixels, 0.057% of the compared area. A 0.1% tolerance would have passed it. The default is now exact, and it fails. A patch that hides the last element moves nothing else, and it passes.
- Waiver and rollback. A production release with `--waive-verification "<reason>"` went through. The audit log holds a `release.waiver` event with the reason and the gate it bypassed. `pw rollback` moved production back, and that is audited too.
- Click checks. Without `--interact`, both click checks were skipped and the record was incomplete. With `--interact`, a click that lands on `/` passed. A click asserted to land on `/order/confirm` failed, and the failure says where the click went: `the click led to /`.
- Editor save path. An automated test runs the real overlay script in Chromium. It selects the button, sets the text, adds a class and a text colour, moves the button before the first paragraph, and saves. The draft patch it writes has anchor `#buy`, the four ops, and status draft. The save is audited under the tester's name. The test needs Chromium, so it is ignored by default: `cargo test -p pw-cli -- --ignored`.

Integration tests that need no browser cover the same gate rules: `cargo test -p pw-cli --test gate` (nine tests). They cover the unapproved dry run, self-review refusal, withdrawal by an edit, the gate with a complete record and with an incomplete one, the waiver, the verification that must match the content, merge conflicts, and merge copies.

### 6.9 Tracking: the needs-a-person band, and the compound drift

The band. A candidate below `floor` (0.5) that still scores at least `review_floor` is now reported as ambiguous, so a person checks it, instead of as broken. Nothing is applied from the band. So the correct and wrong counts cannot change, and only the labels move.

Choosing the floor. Sweep on the development split (seed 101, 300 sites, 3 variants), made before any test seed was used for this setting. Correct and wrong counts are the same at every floor, as the design requires.

| review_floor | present correct | present wrong | present ambiguous | present broken | deleted wrong | deleted ambiguous | deleted broken |
|---|---|---|---|---|---|---|---|
| 0.20 | 538 | 11 | 143 | 1 | 13 | 155 | 39 |
| 0.30 | 538 | 11 | 142 | 2 | 13 | 145 | 49 |
| 0.35 | 538 | 11 | 140 | 4 | 13 | 133 | 61 |
| 0.40 | 538 | 11 | 140 | 4 | 13 | 121 | 73 |
| 0.45 | 538 | 11 | 139 | 5 | 13 | 103 | 91 |
| 0.50 | 538 | 11 | 136 | 8 | 13 | 77 | 117 |

The rule was fixed in advance: the highest value at which present targets reported broken stay under 1% of present cases. That is 5 of 693 at 0.45, against 8 at 0.50, so the default is 0.45.

Test seeds, 0.45 against the old behaviour (0.50):

- Seed 202 (300 sites). Present correct 539 and wrong 9 in both. Present broken fell from 2 to 1. Deleted targets reported ambiguous rose from 80 to 106, and broken fell from 111 to 85. That is 26 more deleted targets for a person to look at, which is the price of the change.
- Seed 7 (30 sites). Present correct 57 and wrong 3 in both. Deleted ambiguous 7 to 9, broken 11 to 9.

The compound drift (`pw train bench --drift compound`), seed 303, 300 sites, 3 variants, 900 present targets. Each target loses its id, its text changes, and it moves; half of the variants also wrap it. The heuristic at 0.50 relocates 301 (33.4%), wrongly matches 3, sends 560 to a person, and calls 36 broken. At 0.45 it relocates the same 301, wrongly matches 3, sends 594 to a person, and calls 2 broken. The live case from the report, an element whose id, text and place all changed, now comes back ambiguous at confidence 0.378 instead of broken.

So the compound case is not solved. Two thirds of such targets still need a person. The matcher in section 6.7 is the only attempt at this case so far.

### 6.10 Tuned tracker: protocol, written before any tuned run (2026-10-11)

Why. The statement asks for smart element tracking that uses structural and semantic context. The heuristic cannot relocate a target whose id, text and place all change (33%, section 6.9). The matcher can (90%), but it loses on mixed drift (section 6.7b). This section tunes the matcher on the GPU and tests a combined rule, with the choices fixed before any run.

Data. Dev: seed 101, mixed drift (300 sites, 3 variants), and seed 102, compound drift (300 sites, 3 variants). Test, fresh and used for nothing else: seed 404 and seed 505, mixed drift (300 sites, 3 variants each), and seed 606, compound drift (300 sites, 3 variants). Seeds 7, 202 and 303 were used for the earlier decision (6.7b) and are not used for any choice here. The matcher's training pairs stay on seed 5001.

Objective on dev: J_dev = J(seed 101) + J(seed 102), with J = correct present minus five times (wrong present plus wrong deleted), as in section 6.6.

Step 1, hyperparameters on the GPU, one factor at a time from the baseline (hidden 128, 4 layers, 4 heads, lr 5e-4, batch 64, 6 epochs, vocabulary 6000, max length 192, pos_weight 1, dropout 0.1). Levels: hidden 192 (6 heads) and 256 (8 heads); 2 and 6 layers; lr 2e-4 and 1e-3; 10 epochs; batch 32 and 128; vocabulary 4000 and 8000; max length 128 and 256; pos_weight 2 and 4; dropout 0.0 and 0.2. Each configuration is scored on dev with the matcher grid of section 6.7 (`--matcher-grid`), and its J_dev is its grid-selected objective. The baseline is also scored. The chosen configuration is the one with the highest J_dev. If none beats the baseline, the baseline stays. There is no second pass.

Step 2, the combined rule. C1: the heuristic decides when it picks a candidate; when it is unsure, the matcher decides, with the thresholds of section 6.7. C2: C1, and also a heuristic pick is turned into a person's decision when the matcher's probability for that pick is below a veto. The veto grid is 0.1, 0.2, 0.3, 0.4, 0.5. The matcher thresholds come from the 140-setting grid. Both are chosen on dev by J_dev, with ties going to fewer ambiguous results.

Step 3, the test, run once with the chosen setting on seeds 404, 505 and 606.

Adoption rule, fixed now. The combined tracker becomes the default chooser only if all of these hold on the test seeds: (a) its objective is at least the heuristic's on 404 and on 505; (b) its present wrong matches do not exceed the heuristic's on 404 and 505; (c) its deleted wrong matches do not exceed the heuristic's on 404 and 505; (d) on compound seed 606 its correct rate is at least ten points above the heuristic's. Otherwise the heuristic stays the default, and the combined tracker is reported as opt-in.

Results (2026-10-11, run in the order above, nothing re-run).

Step 1, hyperparameters. Eighteen configurations, each trained on the laptop GPU and scored with the matcher grid on dev seeds 101 and 102. J_dev is the sum of the two grid-selected objectives. The baseline scores 1146 (seed 101: 322, seed 102: 824). The best change is learning rate 2e-4, at 1176 (352 and 824), so it replaces the baseline. The others, in order: hidden 256 at 1135, vocabulary 4000 at 1134, no dropout at 1116, batch 32 at 1112, learning rate 1e-3 at 1087, hidden 192 at 1083, pos_weight 4 at 1076, dropout 0.2 at 1056, vocabulary 8000 at 1055, max length 128 at 1055, batch 128 at 1047, max length 256 at 1032, 6 layers at 1007, pos_weight 2 at 1005, 10 epochs at 946, 2 layers at 915. The heuristic's J_dev is 679 (418 and 261).

Step 2, the combined rule, on dev with the selected matcher. Of 840 settings, the selected one is accept 0.8, margin 0.2, review 0.4 and veto 0.4, with J_dev 1187. Its objective is 389 on seed 101 (the heuristic: 418) and 798 on seed 102 (the heuristic: 261).

Step 3, the test, run once with that setting on fresh seeds:

| Seed | Combined objective | Heuristic objective | Present wrong (combined, heuristic) | Deleted wrong (combined, heuristic) | Correct present (combined, heuristic) |
|---|---|---|---|---|---|
| 404, mixed | 309 | 398 | 21, 12 | 20, 14 | 514 of 684 (75.1%), 528 of 684 (77.2%) |
| 505, mixed | 368 | 450 | 17, 10 | 23, 14 | 568 of 728 (78.0%), 570 of 728 (78.3%) |
| 606, compound | 799 | 306 | 0, 0 | 0, 0 | 799 of 900 (88.8%), 306 of 900 (34.0%) |

On compound seed 606 the combined tracker relocates 88.8% of targets correctly against the heuristic's 34.0%, a gain of 54.8 points, with no wrong matches on either side.

Adoption rule, applied in code (`/home/harshit/.cache/track/adopt.py`): (a) the objective is at least the heuristic's on 404 and 505: false. (b) present wrong matches do not exceed the heuristic's on 404 and 505: false. (c) deleted wrong matches do not exceed the heuristic's on 404 and 505: false. (d) on 606 the correct rate is at least ten points above the heuristic's: true. The combined tracker is not adopted. The heuristic stays the default. `pw train track` runs the combined tracker as an opt-in.

What this shows. The learned matcher helps on compound drift, where the heuristic has no answer, and the combined rule carries that gain over. On mixed drift the matcher's own wrong matches outweigh its gains, and the hyperparameter sweep moved the mixed-drift objective only a little (322 to 352 on seed 101). The bottleneck is the matcher's errors on mixed targets, not its size or its schedule. Fixing that needs better training pairs or a better features, which is the next step, and it needs a new pre-registered protocol before anyone claims it.

### 6.11 Matcher with a lower positive weight: protocol, written before any test run (2026-10-11)

Why. Section 6.10 found that the combined rule gains on compound drift and loses on mixed drift, because the matcher's own wrong matches on mixed targets outweigh its gains. The sweep in 6.10 tried positive weights of 2 and 4, both above 1. A weight below 1 makes a false pick cost more than a missed pick, so it should make the matcher refuse more often. This is the cheapest single change that targets the errors section 6.10 identified. It is a calibration change, not new training pairs or features.

Candidate, matcher-v3. The matcher-demo recipe, which is `training/match.py` on `data/match` (seed 5001), learning rate 2e-4, batch 64, 6 epochs, vocabulary 6000, hidden 128, 4 layers, 4 heads, dropout 0.1, warm-up 0.05, maximum length 192, training seed 0, with the positive weight changed from 1 to 0.5 and nothing else. Trained on the laptop GPU.

Data. Dev, used only to choose the relocation setting, as in 6.10: seed 101 (mixed) and seed 102 (compound), 300 sites and 3 variants each. Test, fresh and used for nothing else: mixed seeds 808 and 811, and compound seed 909, 300 sites and 3 variants each. Seeds 7, 202, 303, 404, 505, 606 and the training seed 5001 are not test seeds here.

Choice. The relocation setting is the grid's selection on dev by the sum of the two objectives, as in 6.10. The grid writes it to `models/matcher-v3/selection.json`. It is not changed after the test runs.

Objective. J = correct present minus five times (wrong present plus wrong deleted), as in 6.6 and 6.10.

Baseline. The heuristic on the same test seeds, with `pw train bench` at 300 sites and 3 variants. The combined tracker is `pw train track --matcher models/matcher-v3`, which uses the recorded setting.

Adoption rule, fixed now and the same as 6.10 on fresh seeds. The combined tracker with matcher-v3 becomes the default chooser only if all of these hold: (a) its objective is at least the heuristic's on 808 and on 811; (b) its present wrong matches do not exceed the heuristic's on 808 and 811; (c) its deleted wrong matches do not exceed the heuristic's on 808 and 811; (d) on 909 its correct rate is at least ten points above the heuristic's. Otherwise the matcher stays opt-in, and this section says so.

Limits. A pass on synthetic drift does not show performance on real websites (section 8). If the rule fails, the result is reported as it is, and the matcher is not tuned again on these seeds.

Results (2026-10-11, each step run once, nothing re-run; seeds 808, 811 and 909 are spent). Dev: the grid selected accept 0.5, margin 0.2, review 0.4 and veto 0.1 for matcher-v3, with J_dev 1183 (matcher-demo scored 1187 on the same seeds). Raw results: `reports/2026-10-11/results/tracking-v3/`.

| Seed | Drift | J combined | J heuristic | Present wrong (combined, heuristic) | Deleted wrong (combined, heuristic) | Correct present (combined, heuristic) |
|---|---|---|---|---|---|---|
| 808 | mixed | 247 | 433 | 21, 7 | 40, 15 | 552 of 683 (80.8%), 543 of 683 (79.5%) |
| 811 | mixed | 349 | 411 | 18, 10 | 25, 16 | 564 of 708 (79.7%), 541 of 708 (76.4%) |
| 909 | compound | 823 | 272 | 5, 4 | 0, 0 | 848 of 900 (94.2%), 292 of 900 (32.4%) |

Adoption rule, applied as written: (a) false; (b) false; (c) false; (d) true, a gain of 61.8 points. **Not adopted.** The matcher stays opt-in, and the heuristic stays the default.

What this shows. The lower positive weight did not reduce the mixed-drift errors. It increased deleted-target errors on both mixed seeds. The compound-drift gain holds on a new seed (94.2% against 32.4%). So the positive weight is not the lever for the mixed-drift errors, and section 6.10's conclusion stands: the matcher's errors on mixed targets are the bottleneck. A new change needs a new pre-registered protocol, and the test seeds used here cannot be reused.

## 7. Deviations from the plan and the brief

1. **Model from scratch, architecture from a reference only.** Per the brief. The plan's Qwen fine-tune and its quality expectations do not apply. EmbeddingGemma-300M supplies the layer design only. Its weights and tokenizer are not used.
2. **Heuristic fallback is not trusted for deleted targets.** Section 6.1 shows why. Relocations are provisional until verified. For protected channels the release gate now requires a complete passing verification of the same patch behaviour, and approvals in the audit log (section 4). Before 2026-10-11 the gate checked only the relocation flag, and this sentence overstated it.
3. **Added identity assertions and a release gate.** Neither is in the plan text. Both follow from the plan's own rule that every relocated patch must pass its assertions in preview.
4. **Browser runtime and editor are JavaScript.** Browsers execute JavaScript or WebAssembly. The runtime is small and is the only code visitors run.
5. **Playwright engine is Microsoft's driver.** Driven through `playwright-rs`. The repository's JavaScript is limited to the in-page snippets.
6. **Rust inference is in place.** `crates/pw-model` runs the exported model with candle (CPU, f32), using candle's Gemma 3 implementation. It is checked against the Python reference: on the trained checkpoint the top five tokens match and the logits differ by at most 3e-6. `pw --model DIR` makes the model the first chooser for relocation. It decides when it is confident (`model_accept`, now 0.99, chosen in section 6.5) and the heuristic decides otherwise. A confident refusal is final. The benchmark (`pw train bench --model DIR`) scores the model against the heuristic on the same held-out cases.
7. **`hk` (the Zig runtime) is not used.** The brief says Rust.
8. **Kaggle training runs on 13 accounts (26 T4s).** Each account is one group: DDP across its two GPUs, its own data shard, DiLoCo sync through a coordinator on this machine behind a Cloudflare tunnel. Details, measured throughput and the problems hit (silently disabled internet, camelCase metadata, no cancel) are in `docs/KAGGLE.md`.

## 8. Known limits

- On the synthetic benchmark the heuristic mis-matches 6.4% of deleted targets (seed 202, section 6.6) and 0% on seed 7. Its relocations are provisional until verified (section 7, item 2).
- The model does not yet answer null, so it cannot protect against deleted targets. This is the most important open item.
- The outer momentum of the DiLoCo run (0.9) overshot on 2026-10-10, and held-out loss rose from about 3.2 to 7 to 8. The coordinator now defaults to momentum 0, and the run was restarted from the damaged global to see whether it recovers. See docs/KAGGLE.md, "Outer momentum".
- The model is wired into the CLI behind `--model`. The heuristic remains the default. On deleted targets the v1 checkpoint is much worse than the heuristic (section 6.5), so it should not become the default yet.
- The proxy buffers whole response bodies, so very large pages use memory.
- When a response carries several CSP headers with different nonces, the runtime tag can carry only the first nonce. This is an edge case, and it is not tested.
- Shadow DOM is not handled. The runtime does not look inside shadow roots, so patches cannot target elements there. Cross-origin iframes and canvas are out of scope.
- `clickNavigatesTo` assertions are checked only with `pw verify --interact`, because a click can submit a form or trigger a real action. Use that flag on a copy of the site where it is safe. Without it the record is incomplete, and a protected release is refused.
- The synthetic generator gives decoys and twins no id and no test id, while the gold element keeps both. Real look-alikes often keep stable ids, so every result on this benchmark flatters any rule that uses ids. Read section 6.9's compound figures with this in mind.
- The visual check compares the candidate channel with the production channel, pixel by pixel, outside the patched elements. Anything else that changes the page is reported as a failure. That includes a patch that moves or hides content and so shifts the layout below it, and content that changes on each load (animation, clocks, ads). The default tolerance is exact. Raise `visual_tolerance` for such pages, and say why. Visual checks have no stored baseline: production is rendered again on every run.
- Tracking, compound drift: when an element loses its id, changes its text, and moves at once, the heuristic relocates 33% of targets correctly and sends 66% to a person (section 6.9). The matcher experiment (section 6.7) is the only attempt at this case so far.
- Review and the gate depend on the local audit log. Approvals from another branch do not carry over (`pw merge --write` copies patches, and the copies need approval in this log).
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
