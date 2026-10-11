# PATCHWORK: how well the project matches the problem statement

Status: analysis, 2026-10-10. Not reviewed by anyone else.

## 0. Scope and method

- **The statement.** The PATCHWORK section of "CRCE Round Problem Statements" (PDF page 5, Web / App Dev track). The PDF has no judging rubric and no rules on other pages. The rest of the PDF covers other problems, so it was not used as a source. The requirements below are the statement's "Key Features", quoted or paraphrased.
- **The project.** The code in `crates/`, `runtime/`, and `training/`, plus `docs/DESIGN.md`. Where the docs claim something, I say whether I could reproduce it.
- **Live checks.** I ran the product end to end on a sample page, using a local upstream server and Chromium 153 (the Playwright build installed by `pw browser install`). Section 5 has the exact steps. The sample page is the only site tested.
- **Conflict of interest.** Most of this codebase was written by the same assistant that wrote this report. I tried to check the claims rather than trust the docs. Anyone can rerun section 5 to check my results.
- **Not tested:** the visual editor's save path (it needs a person in a headed browser), real websites, the Kaggle run's outcome, and any user study.

## 1. Verdict

The core patch lifecycle is real and mostly works. Preview channels, releases with a gate, rollback, browser-based assertions, conflict detection, and a hash-chained audit log all ran on the sample page. The project is weakest on the statement's first feature, visual patching, and on the breadth of verification and audit. It is also off-scope in a large way: most of the effort went into a distributed model-training pipeline that the statement does not ask for, and whose benefit to the patching features has not been shown.

In one line: the back end of a patch-release system is well built and partly tested; the visual front end and the verification and audit breadth the statement asks for are only partly there.

## 2. Requirement by requirement

| Statement feature | What the project has | How I checked it | Status |
|---|---|---|---|
| Visual patching: modify elements, layouts, content, and supported behaviour through a visual interface | `pw edit` opens a headed overlay. The person selects an element and can change its text, background colour, or visibility. Save writes a patch. The DSL has 18 operations, including moves, inserts, show, hide, and behaviour operations (`onClickNavigate`, `onClickSubmit`, `onClickFetch`, `disable`, `enable`), but the editor only exposes three of them. | Read `runtime/editor.js` (65 lines) and the op list in `crates/pw-core/src/patch.rs`. The interactive save path was not run. | Partial |
| Smart element tracking: find invalidated patches and locate moved elements, using structural and semantic context | Multi-locator anchors (CSS, XPath, test id, role and name) and a fingerprint (tag, text, attributes, ancestors, neighbours, box). A fast path needs two locators to agree. A heuristic then picks or abstains. There is no meaning-level ("semantic") matching. | Live: an element moved into a new wrapper and changed class was relocated with confidence 0.87. It is the only button on the page, so the choice is correct. The same element also renamed and retexted, then wrapped, was reported **broken**, although a person would still call it the same button. | Partial |
| Collaborative editing: create, review and merge patches; detect conflicting changes | Patches are YAML files with an `author` field. `pw merge --with DIR` reports write-write conflicts and anchor-dependency warnings. | Live: a colleague's patch changing the same text gave a hard WriteWrite conflict. There is no review or approval state. `pw merge` writes nothing, so merging is not supported, only checking. | Partial |
| Isolated preview: test patches without affecting the published site | Channels (`production`, `preview/x`), selected by a cookie. Each channel serves its own manifest. | Live: the production manifest was empty while the preview cookie got the patch, with its hash. `pw verify` ran on the preview channel. | Met |
| Automated verification: browser interactions and visual checks to find regressions before deployment | `pw verify` runs assertions in Chromium through the proxy: visible, exists, text contains, attribute equals. It reports pass or FAIL. | Live: a true assertion passed and a deliberately false one was reported FAIL. There are **no visual checks**: no screenshot or pixel comparison (zero matches in the code). `clickNavigatesTo` is reported as "not checked", so click interactions are not run. Verification is **not a release gate** (section 4). | Partial |
| Versioned deployment and rollback: track releases, detect failed changes, restore a working state | Immutable releases, channel pointers, `pw rollback`, and beacon-driven auto-rollback in the proxy. | Live: a release published. The gate refused a broken patch, and blocked a relocated patch unless the accept flag was passed. Rollback and auto-rollback are covered by tests (`cmd_rollback` tests, and a proxy end-to-end test); I did not run them live. | Met, with a weak gate |
| Audit trail: history of modifications, deployments, conflicts, reversions | Hash-chained JSONL with a HEAD file. `pw log` and `pw audit-verify`. | Live: the release was recorded, and `audit-verify` passed. The code records releases and rollbacks only. Patch edits and merge conflicts are not recorded. The actor is the string `cli`, so there is no record of which developer did what. | Partial |
| Work without rebuilding the original application | The proxy injects one script tag and applies patches in the browser. The upstream is not changed. | Live: the upstream page was unchanged, and the in-page runtime applied the patch (the assertion proved it). | Met |

## 3. Strengths

- **The lifecycle hangs together.** Patch, check, release, preview, verify, rollback, and audit all exist, and the pieces connect. The live run went through all of them.
- **The release gate is conservative.** Broken patches are refused. Relocated patches need an explicit flag. A dry run cannot publish. This is the right default for a tool that changes live sites.
- **Verification catches a real failure.** A false assertion produced FAIL, so the signal is not just green checks.
- **Isolation is real.** Production stayed clean while preview carried the patch.
- **Testing is substantial for this stage.** 138 Rust tests and 48 Python tests pass. Conflict semantics, audit tamper and truncation, and proxy behaviour all have tests.
- **The design document is candid about its limits.** Section 8 lists them: shadow DOM, the proxy buffering whole bodies, and the audit HEAD being local. Few projects are this specific.

## 4. Gaps and risks, most important first

1. **The visual interface is the thinnest part of the project.** The statement's first feature is a visual patching interface. The editor is 65 lines and exposes text, background colour, and hide. It cannot move elements, change layout, or change behaviour, even though the DSL can. Its save path has never been run in a browser.
2. **The release gate does not check verification.** `cmd_release` never calls `pw verify`. The gate only checks relocation status and the accept flag. The design document says "the release gate enforces that" (section 7, item 2), which is an overstatement. A relocated patch can ship after a human passes the flag, with no browser run.
3. **The dry run misleads.** `pw release --dry-run` prints "would publish release" even when the real run would be blocked. I confirmed this on the sample page.
4. **Tracking misses renamed and moved elements.** On the sample page, an element that was moved, renamed and retexted was declared broken, not located. "Semantic" context is lexical (text similarity) plus role and name. This matters because the statement names it.
5. **The audit trail is incomplete and anonymous.** Releases and rollbacks are recorded. Patch edits and conflicts are not. Entries have no developer identity. The design document says the HEAD file is local, so anyone who can rewrite both files can forge the log.
6. **Collaboration stops at detection.** There is no review or approval state, and `pw merge` cannot produce a merged set.
7. **Verification breadth is narrow.** There are no visual checks, and click-through assertions are not run.
8. **Most of the effort went off-scope.** The statement asks for a patching platform. The project's Rust and Python lines are split roughly as follows: patching and serving (`pw-core`, `pw-proxy`, `pw-cli`, runtime) about 5,000 (`pw-cli` also holds some training commands); model-training and benchmark code (`pw-train`, `pw-model`, `training/`) about 8,200. The model is not required by the statement. Its benefit is not shown: on the project's own benchmark, the model did not beat the heuristic (docs/DESIGN.md, section 6.5). The distributed training run, on 13 Kaggle accounts (20 registered), took most of the compute and attention in the last day.
9. **Kaggle account use is a terms risk.** The training run uses 13 Kaggle accounts in parallel, from 20 registered. The project's docs say the multi-account terms were not read. The statement does not mention this, but a platform's terms and a judge's view of the approach both apply.

## 5. Documentation claims checked against the code

| Claim | Where | Result |
|---|---|---|
| "Automated verification ... both assertions passed" | DESIGN §1 | Reproduced: a true assertion passed, and the runtime applying the patch is what makes it pass. |
| "the release gate enforces that" (relocations are verified) | DESIGN §7, item 2 | **Not supported.** The gate checks the accept flag, not verification. |
| "Save path not exercised interactively" | DESIGN §1 | Accurate. Still true. |
| "Heuristic relocation mis-matches about 18% of deleted targets" | DESIGN §8 | From an older 200-site run. With the adopted weights the rate is 6.37% on seed 202 (6.86% before tuning) and 0% on seed 7 (section 7), so the figure is stale. |
| Audit tamper and truncation are detected | DESIGN §4 | Accurate, as far as it goes. The HEAD is local, which the design doc says. |

## 6. Effort and scope

- Patching and serving: `pw-core` 2,903 lines, `pw-proxy` 537, `pw-cli` 1,332, runtime JS 239. Total about 5,000.
- Training and benchmark: `pw-train` 3,223, `pw-model` 389, `training/*.py` 4,620. Total about 8,200.

These are counts of lines, not of quality. They show where the time went, not whether any piece is good. The statement does not require the training side, and its benefit to the statement's features is not demonstrated.

## 7. What the measurements say about tracking

On the project's synthetic benchmark (not real websites), with the adopted similarity weights:

- Seed 202 (300 sites, 3 variants; 696 present, 204 deleted): 77.4% of present targets correct, 1.29% wrong, about 21% abstained. Deleted targets wrongly matched: 6.37%. Before tuning: 73.3%, 1.15%, and 6.86%.
- Seed 7 (30 sites): 79.2% correct, 4.17% wrong, 0% of deleted targets wrongly matched.

So about one present target in five is flagged for a person rather than relocated. That is the safe failure, and it costs automation. Wrong matches are rare but not zero, which is why the release gate matters.

## 8. What would bring the project into line with the statement

In order of impact on the statement:

1. Make the visual editor cover the statement's list: layout moves and insertions, behaviour, and content. Test the save path with a scripted browser run.
2. Make `pw release` require passing verification (or an explicit, recorded waiver), and say so in the docs.
3. Extend the audit trail to patch edits, merge conflicts, and verification results, with a developer identity on every entry.
4. Add a review state to patches (proposed, reviewed, approved), so "review" means something.
5. Add visual regression checks: a screenshot per assertion, compared to the last release.
6. Improve tracking for the case that failed here: an element that moved and changed its text. A "needs a person" path already exists; use it instead of "broken".
7. Decide on the model's role explicitly. Either justify it against the patching features with a result, or cut it from the submission's claims. Reduce its share of effort either way.
8. Fix the three documentation overclaims in section 5 and the misleading dry-run message.

## 9. Reproduce the live checks

Commands, from the project root. All files are under a scratch directory (`e2e/`), and nothing here touches the real project.

```sh
# a local site with a checkout button
python -m http.server 8099 --bind 127.0.0.1 --directory e2e/site   # index.html has <button id="buy" class="btn btn-primary">Buy now</button>
pw init --project e2e/proj --name demo --upstream http://127.0.0.1:8099
pw snapshot --project e2e/proj --out e2e/proj/snapshots/index http://127.0.0.1:8099/
# patches/checkout.yaml: css "#buy", role button "Buy now", op setText "Checkout now", assert textContains "Checkout now"
pw check --project e2e/proj --snapshot "/=e2e/proj/snapshots/index"          # valid
pw release --project e2e/proj --channel preview/x --snapshot "/=e2e/proj/snapshots/index"
pw proxy --project e2e/proj --listen 127.0.0.1:8787 &
pw verify --project e2e/proj --base http://127.0.0.1:8787 --channel preview/x  # pass
# change the site (rename the id, retext, wrap in a div) and re-snapshot: check reports broken
# change only the class and move the element: check reports relocated (0.87)
pw release --project e2e/proj --channel production --snapshot "/=..."          # refused (broken) or blocked (relocated)
pw merge --project e2e/proj --with e2e/other/patches --snapshot "/=..."        # hard WriteWrite conflict
pw log --project e2e/proj ; pw audit-verify --project e2e/proj
```

Run `pw browser install` once before `pw verify` and `pw snapshot`.

## 10. Status after the 2026-10-11 changes

This section records what changed after sections 1 to 9 were written, against each recommendation in section 8. The evidence is in `docs/DESIGN.md`, sections 6.7 to 6.9.

| Recommendation (section 8) | Status | Evidence |
|---|---|---|
| 1. Visual editor covers layout, content and behaviour; save path tested | Partly done. The editor now offers text, background, text colour, font size, classes, link target, hide, and move before or after another element. Behaviour operations (onClickNavigate, onClickSubmit, onClickFetch, disable, enable) and inserts are not in the editor yet. The save path is tested in Chromium with the real overlay script, headless only | DESIGN 6.8; `browser.rs` test `the_editor_save_path_writes_a_draft_patch_with_the_chosen_ops` |
| 2. Release requires passing verification, or a recorded waiver | Done for protected channels (`production` by default). The gate needs a complete, passing record for the same patch behaviour, and a waiver is audited. Relocated patches still need `--accept-relocated` | DESIGN 4, 6.8; `crates/pw-cli/tests/gate.rs` |
| 3. Audit extended to edits, conflicts, verification, identity | Done. Every event carries an actor (flag, `PW_ACTOR`, or login name). Edits, reviews, verifications, merges, relocations, releases, waivers, rollbacks and blocked releases are recorded. The identity is declared, not authenticated | DESIGN 4; `audit-verify` passes after each scenario in 6.8 |
| 4. Review state (proposed, reviewed, approved) | Done: draft, proposed, approved, rejected. An approval is tied to the patch's content, so an edit withdraws it. The reviewer is declared, not authenticated | DESIGN 6.8; `pw-core/src/review.rs` |
| 5. Visual regression checks | Done in a limited form. Each patched page is compared pixel by pixel with production outside the patched elements, exact by default. Any other change fails, including layout shifts. There is no per-assertion screenshot | DESIGN 4, 6.8, 8 |
| 6. Better tracking for an element that moved and changed its text | Partly done. The case now reads "needs a person" instead of "broken". On a compound drift (id dropped, text changed, moved) the heuristic relocates only 33% correctly. A matcher trained from scratch reaches 90% on that slice, and a tuned combined tracker reaches 89% on a fresh compound seed. Neither is adopted, because both lose on the mixed-drift protocol slices. DESIGN 6.7b and 6.10 explain the decisions | DESIGN 6.7b, 6.9 |
| 7. Decide the model's role explicitly | Decided for now. The language model was trained from scratch on Kaggle, and it stays an opt-in chooser that the benchmark did not show to help. The matcher is a second from-scratch model. Under the protocol fixed in advance it is not adopted, and it stays opt-in. Neither model is the default; the heuristic is | DESIGN 6.5, 6.7, 6.7b |
| 8. Fix the overclaims and the dry-run message | Done. The design document's "the release gate enforces that" is now accurate for protected channels, the stale 18% figure is replaced, `clickNavigatesTo` is described as run with `--interact`, and the dry run lists every blocker and exits non-zero | DESIGN 4, 7, 8 |

Gaps from section 4, in the same order:

1. Visual editor: partly closed (see recommendation 1).
2. Release gate ignores verification: closed for protected channels.
3. Dry run misleads: closed. It lists the blockers and exits non-zero.
4. Tracking misses renamed and moved elements: partly closed (recommendation 6).
5. Audit incomplete and anonymous: closed for coverage. Identity is declared only.
6. Collaboration stops at detection: review states and `merge --write` are in place. Approvals do not carry across branches (they are per audit log).
7. Verification breadth: click checks (with `--interact`) and the visual check are in place. There are no screenshots per assertion.
8. Effort off-scope: not changed in this round. The language model's training is the largest share of effort, and its benefit to the patching features is still unshown.
9. Kaggle terms risk: not changed.

Update, later on 2026-10-11, after the table above:

- Collaboration: approvals carry over a merge for the same content (`pw merge --with-audit`), each recorded with its source. A review that disagrees across branches is a conflict. Tests in `crates/pw-cli/tests/collab.rs`. This closes most of gap 6 for the single-project workflow. Multi-user identity is still declared, not authenticated.
- Visual patching: the editor lists pending changes and blocks unsafe links before save. Covered by a headless browser test. The headed session is still untested.
- Smart element tracking: a second pre-registered test (DESIGN 6.11, a positive weight of 0.5) was not adopted. Mixed-drift errors did not fall, so the tracker is unchanged. A learned gate on the combined tracker (DESIGN 6.13, trained on dev seeds 101 and 102 only, tested on fresh seeds 1212, 1313 and 1414) cuts wrong matches against the combined rule on both mixed seeds and keeps the compound gain (88.4% against the heuristic's 31.8%). It misses the pre-registered adoption rule narrowly, on seed 1212, so it stays opt-in and the heuristic stays the default.
- Correction to the earlier review: the DSL already restricts navigation and fetch targets to same-site paths (`check_local_path`, `crates/pw-core/src/dsl.rs`), and sanitizes inserted HTML. Section 4 item 1's claim about unchecked targets was wrong. The earlier search covered only `patch.rs`.

One correction to section 5 of this report: the claim "the release gate enforces that" was an overstatement when it was written, and it is accurate now only for protected channels. DESIGN section 7, item 2 says this.
