# PatchWork

Patch live websites without rebuilding them. Patches are small files with a DSL, re-anchored when a site changes, verified in a real browser, released through a reverse proxy, and rolled back when visitors' browsers report errors.

Built for the GDG on Campus CRCE college round, problem 3, "PATCHWORK: Hotfix the Internet". The plan is in `docs/DESIGN.md`. The runbook for distributed training is in `docs/KAGGLE.md`.

## Layout

| Path | What it is |
|---|---|
| `crates/pw-core` | Deterministic core: snapshots, fingerprints, re-anchoring, DSL validation, conflicts, audit chain, releases |
| `crates/pw-proxy` | Reverse proxy: injects the runtime, serves manifests, takes beacons, auto-rolls back |
| `crates/pw-cli` | The `pw` binary: project commands, Playwright management, the training entry points |
| `crates/pw-train` | Synthetic data, held-out benchmark, DiLoCo coordinator, Kaggle launcher |
| `runtime/` | The in-page runtime (`runtime.js`) and visual editor (`editor.js`), the only JavaScript shipped |
| `training/` | Python: tokenizer, from-scratch model, trainer, DiLoCo worker, evaluator |
| `docs/` | `DESIGN.md` (decisions, deviations, measured results, limits), `KAGGLE.md` (runbook) |

## Quick start

```sh
cargo build -p pw-cli                      # builds ./target/debug/pw
./target/debug/pw browser install          # downloads Playwright's Chromium
./target/debug/pw --project demo init --name shop --upstream http://127.0.0.1:8080
./target/debug/pw --project demo snapshot http://127.0.0.1:8080/cart --out demo/snapshots/cart.html
./target/debug/pw --project demo edit http://127.0.0.1:8080/cart      # visual editor; Save writes a patch
./target/debug/pw --project demo check --snapshot /cart=demo/snapshots/cart.html
./target/debug/pw --project demo release --snapshot /cart=demo/snapshots/cart.html
./target/debug/pw --project demo proxy --listen 127.0.0.1:8787        # visitors go here
./target/debug/pw --project demo verify --base http://127.0.0.1:8787 --channel production
./target/debug/pw --project demo rollback --channel production
```

## Commands

- `pw init`, `pw dsl-schema`, `pw log`, `pw audit-verify`
- `pw snapshot URL --out FILE`: captures HTML and element rectangles
- `pw edit URL`: visual editor (headed Chromium); each Save writes a validated patch with an identity assertion
- `pw check --snapshot PAGE=FILE [--write] [--json]`: re-anchors every patch. Exit 2 if any patch is broken or ambiguous
- `pw merge --with DIR --snapshot PAGE=FILE`: conflicts between two patch sets
- `pw release [--dry-run] [--accept-relocated] --snapshot PAGE=FILE`: publishes an immutable release. Refuses heuristic relocations unless `--accept-relocated`
- `pw rollback`, `pw proxy --listen ADDR`, `pw verify --base URL --channel NAME`
- `pw browser install [--with-deps]`, `pw browser status`
- `pw train gen-data | bench | serve | kaggle plan | kaggle push`

## Status

Verified here:

- `cargo test --workspace` passes (unit and end-to-end tests). The end-to-end proxy test runs a real upstream, injects into gzipped HTML under a nonce CSP, and checks the rollback path.
- The CLI ran the full path in Chromium 153 through the proxy: check, release, proxy injection, manifest, and browser assertions, which confirmed that the runtime applied the patch.
- A two-round distributed run with one coordinator and three CPU workers completed.
- A Kaggle plan with placeholder accounts validates. Nothing was pushed.

Not verified or not done:

- **No trained model is wired into the CLI.** The CLI uses the deterministic heuristic. Inference from Rust (candle) is the next step.
- The heuristic is wrong on about 18% of deleted-target cases in the held-out benchmark. See `docs/DESIGN.md`. This is why a relocation needs verification before it is released.
- `pw edit` opens a visible browser. Its save path was not exercised interactively here.
- Nothing ran on Kaggle, and Kaggle's rules on multiple accounts are not confirmed. See `docs/KAGGLE.md`.
- The general developer corpus is not built. See `docs/KAGGLE.md`.

## Language note

Everything in the CLI, proxy, core, data generation, coordinator and launcher is Rust. Two exceptions are required by the platform:

- The in-page runtime and editor are JavaScript, because browsers run JavaScript or WebAssembly. The runtime is small, and it is the only code visitors run.
- Training is Python (PyTorch), which the brief allows. The shared starting point, the tokenizer and the weights are plain files that Rust reads.

Browser automation uses Microsoft's Playwright driver through the `playwright-rs` bindings.
