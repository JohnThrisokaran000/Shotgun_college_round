# Demo runbook: PatchWork for the judges

How to show everything we built, in 5 or 8 minutes, and what to say at each step. Add 3 minutes to either run for an optional live segment on the real binary.

- **The web demo** is `Website/index.html`. It has a presenter mode that walks through the story one step at a time. The step notes in this file match the notes on the page.
- **The live segment** runs `Website/live/demo.sh`: the real `pw` binary, a real proxy, real Chromium, and a local sample shop. It proves the web demo is not a mock-up.

The console on the web page is a simulation, and the page says so. Its commands, flags and messages come from the real CLI, and its outputs were copied from runs of the binary on the sample shop. If a judge asks whether it's real, say exactly that, then run the live segment.

## 1. Pick a format

| Format | Length | When to use it |
|---|---|---|
| Short run | about 5½ min | A tight slot. Covers all seven brief features, the tracking results and the close. |
| Full run | about 8½ min | The default. Adds the brief table, `pw check`, the merge conflict, the patch file, hard tracking cases, the architecture and the models. |
| Either run + live | +3 min | A judge asks "is it real?", or you have the time. Run it after the Audit step or after the close. |

In presenter mode the **Full/Short** button on the bar switches between the two. To open the page already presenting, add `#present` or `#present-short` to its address.

## 2. Before the judges arrive (T−30 min)

1. **Serve the page.** Run `cd Website && python3 -m http.server 8000`, then open `http://127.0.0.1:8000/`. Opening `index.html` directly as a file also works, including the presenter-view window.
2. **Internet.** The 3D globe, the training figure, the fonts and the scroll animations load from CDNs (cdnjs, unpkg, Google Fonts). Open the page once on the venue Wi-Fi so the browser caches them. Without internet the page still works: the console, the presenter mode, the tracking view and every number. Only the two 3D figures stay blank and the fonts fall back.
3. **Display.** Use 100% zoom in a full-screen window (F11). The console is designed for 1280 px width or more. On a projector at 1024 px it stacks the terminal above the shop, which still works but scrolls more.
4. **Second screen.** If the laptop drives a projector as a separate display, press **P**, then **Presenter view ⧉**, and drag that window to the laptop screen. It shows the script, the next step and a timer. If the screens are mirrored, leave the notes closed: the judges would see them.
5. **Rehearse once, then reload.** A reload resets everything: the console, the R1–R7 chips and the timer. The console's **Reset** button resets only the console.
6. **Live segment (optional).** In a terminal, run `Website/live/demo.sh up`. It starts the shop on :8080 and the proxy on :8787, and it needs `cargo` and `python3`. Leave that terminal open beside the browser, with a large font. Open `http://127.0.0.1:8787/` in a second browser tab.
7. **Roles, for two people.** One person drives (keyboard, clicks) and the other speaks. Swap at the Results step if you like.

## 3. Controls

| Key | Does |
|---|---|
| **P** | Start or stop presenter mode (the **▶ Present** button on the cover does the same) |
| **→**, **Space**, **Page Down** | Next step. Most clickers send Page Down |
| **←**, **Page Up** | Previous step |
| **N** | Show or hide the speaker notes on the page |
| **Home** | First step |
| **T** | Reset the timer |
| **Esc** | Leave presenter mode |
| **J** / **K** | Next or previous sheet, outside presenter mode |

When you move to a console step, the page types that chapter's commands for you. It also types any earlier commands that haven't run yet. If you press → while commands are still running, the page scrolls on and the commands finish in the background. Wait for each chapter to finish before you move on.

The bar at the bottom shows the R1–R7 chips. Each chip fills in once its feature has been demonstrated on screen, and the brief table on sheet 02 ticks the matching row. The close step returns to that table, so the judges see all seven ticked.

The timer shows the elapsed time and the target for the current step. It is underlined in a wavy line once you are more than 15 seconds behind.

## 4. Run of show

R1 to R7 are the brief's features: R1 visual patching, R2 smart element tracking, R3 collaborative editing, R4 isolated preview, R5 automated verification, R6 versioned deployment and rollback, R7 audit trail.

| # | Step | Sheet | Shows | Time | Short run |
|---|---|---|---|---|---|
| 1 | Cover | 00 | what it is | 0:20 | yes |
| 2 | The problem | 01 | why | 0:30 | yes |
| 3 | The brief | 02 | the agenda | 0:20 | |
| 4 | Console · Edit | 03 | R1 | 0:35 | yes |
| 5 | Console · Check | 03 | R2 | 0:15 | |
| 6 | Console · Preview | 03 | R4 | 0:30 | yes |
| 7 | Console · Verify | 03 | R5 | 0:30 | yes |
| 8 | Console · Review | 03 | R3, R6 | 0:35 | yes |
| 9 | Console · Merge | 03 | R3 | 0:30 | |
| 10 | Console · Auto-rollback | 03 | R6 | 0:40 | yes |
| 11 | Console · Audit | 03 | R7 | 0:30 | yes |
| — | *Live segment (optional)* | terminal | R1–R7, for real | 3:00 | |
| 12 | Anatomy of a patch | 04 | safety | 0:30 | |
| 13 | Tracking · Redesign | 05 | R2 | 0:25 | yes |
| 14 | Tracking · Hard cases | 05 | R2 | 0:25 | |
| 15 | System | 06 | architecture | 0:20 | |
| 16 | Models | 07 | what we trained | 0:30 | |
| 17 | Results | 08 | evidence | 0:40 | yes |
| 18 | Close | 02 | recap | 0:15 | yes |

## 5. The script

Each step has three parts. **Say** is the talk track, and it is the same text as the on-page notes. **Do** is what to click or point at. **If asked** gives short, accurate answers to the questions that step tends to raise.

### 1 · Cover (0:20)

**Say.** "PatchWork fixes live websites without rebuilding them. A fix is a small, safe patch file. A reverse proxy delivers it to visitors' browsers, and every change is previewed, verified in a real browser, reviewed, released, and rolled back by itself if it breaks."

**Do.** Let the globe turn. The bright squares are live patches.

### 2 · The problem (0:30)

**Say.** "Three problems. A one-word typo still goes through commit, build, test, review and deploy. Tools that patch pages with CSS selectors break as soon as a redesign renames a class. And when a fix breaks in visitors' browsers, nobody on the team finds out."

**Do.** Point at the three cells, left to right.

### 3 · The brief (0:20, full run)

**Say.** "The problem statement lists seven features. Each row maps one to our code, the command, and how we tested it. Every Show button runs that feature on this page. I will now run all seven as one real workflow."

**Do.** Point at the Evidence column, then at the R1–R7 chips in the bar.

**If asked "why is R1 half filled?"** "The editor's save path is tested in headless Chromium with the real overlay script. We haven't run an automated test of a headed session, so we mark it honestly."

### 4 · Console · Edit (0:35) — R1

**Say.** "Left, a terminal; right, the live shop. The commands and messages are the real CLI's. We create a project, snapshot the cart, and start the proxy. Visitors see no change yet. Then the visual editor: click the button, fix 'Byu now', point it at checkout. A link to another site is blocked before Save. Save writes a patch file. No code, no rebuild."

**Do.** Wait for the editor panel. It shows the red error on the outside link, and then the save.

**If asked "what else can the editor change?"** "Text, background and text colour, font size, classes, link targets, visibility, moving an element before or after another, click behaviour (navigate, submit, fetch, disable, enable), and inserting or replacing HTML. All of it goes through the same DSL validation."

### 5 · Console · Check (0:15, full run) — R2

**Say.** "pw check finds the element again in the snapshot. Its locators agree, so it is valid on the fast path. What happens after a redesign comes in a minute."

### 6 · Console · Preview (0:30) — R4

**Say.** "Can I push straight to production? The dry run says no: nobody has approved it, and nothing has verified it. So I release to a preview channel. Same URL, two versions. Production still shows the typo; with the preview cookie I see the fix."

**Do.** Click the **production** and **preview/fix** tabs above the shop.

**If asked "how is preview isolated?"** "A cookie selects the channel (`/__pw/channel?name=preview/fix`). The proxy serves each channel its own release manifest. Production visitors never get the preview's patches."

### 7 · Console · Verify (0:30) — R5

**Say.** "pw verify drives real Chromium through the proxy. It checks the patch's assertions, then compares the page with production pixel by pixel, outside the patched element: 0 of about 917,000 pixels changed. A patch that shifts the layout fails here. The result is a record tied to this exact content."

**If asked "what fails it?"** "A failed assertion, a click that lands somewhere else (with `--interact`), or any pixel change outside the patch. A patch that hid the first cart row shifted 508 pixels, and it failed. The default tolerance is exact."

### 8 · Console · Review (0:35) — R3, R6

**Say.** "Review. I propose the patch. I cannot approve my own work, so that is refused. A reviewer approves, and the approval is tied to the content hash: edit the patch and the approval is withdrawn. Now the production release passes the gate. Releases are immutable; a channel is just a pointer."

**Do.** Point at the Releases strip under the stats.

**If asked "is the identity authenticated?"** "No. `--actor` is declared, not checked, and the docs say so. Signing actions is the next step."

### 9 · Console · Merge (0:30, full run) — R3

**Say.** "A teammate worked on another branch. pw merge finds a hard conflict: we both changed the same button's text. Nothing is written until it is resolved. They drop their version, and the merge brings in their promo banner."

**If asked "what about approvals across branches?"** "`pw merge --with-audit` carries an approval over when the content is identical, and records where it came from. If one branch approved a patch and the other rejected the same content, that is a conflict."

### 10 · Console · Auto-rollback (0:40) — R6

**Say.** "The teammate's promo banner is merged, and they are in a hurry. They waive review and verification, and both waivers go into the audit log with the reason. The banner breaks the shop's own JavaScript in visitors' browsers. The runtime sends error beacons; at 5 errors from 3 sessions within a minute, the proxy rolls production back by itself. We ran this on the real binary, and that error text is what real Chromium reported."

**Do.** Watch the error and session meters fill, then the production pointer step back.

**If asked "why 5 and 3?"** "So that one broken browser can't trigger a rollback. The policy is in `pw.toml`. Only error beacons count; 'ok' and 'missing' don't. Each release rolls back at most once."

### 11 · Console · Audit (0:30) — R7

**Say.** "Every step is in a hash-chained log with who did it: the edit, the review, the waivers, the automatic rollback. Someone edits the waiver line to cover it up, and audit-verify names the exact line that changed."

**If asked "can someone rewrite the whole file?"** "Yes, if they can also rewrite the HEAD file next to it, because HEAD is local. Pinning HEAD somewhere else, such as a git commit or a server, closes that gap. DESIGN §4 says so."

### Live segment (optional, 3:00)

Use it here, or after the close. See section 6.

### 12 · Anatomy of a patch (0:30, full run)

**Say.** "This is the patch file. Locators and a fingerprint find the element again; operations change it; assertions check it. The DSL has 18 operations and deliberately no script op. Styles are allowlisted, links must stay on the same site, and inserted HTML is sanitised. A patch cannot run code or send data off-site."

**Do.** Hover A to D on the right, then the struck-out `script` op.

### 13 · Tracking · Redesign (0:25) — R2

**Say.** "Smart tracking. A redesign renames the class, drops the id and wraps the button. The selectors miss, but the fingerprint still matches: tag, text, neighbours, ancestors, position. So the patch is relocated, and the gate wants it verified on preview before it ships."

**If asked "are those scores real?"** "The weights and thresholds are the real defaults. The scores on this sheet are illustrative. The measured results are on sheet 08, and the live segment shows a real relocation at confidence 0.767."

### 14 · Tracking · Hard cases (0:25, full run) — R2

**Say.** "When the id, the text and the position all change, the heuristic does not guess. It asks a person and shows the top three. If the button is deleted, it reports the patch broken and the release is refused. Our learned tracker handles most of these hard cases; the numbers are coming up."

**Do.** Then click **Deleted**.

### 15 · System (0:20, full run)

**Say.** "The architecture. Developer tools and a deterministic Rust core on the left. The proxy is the only part in the request path: it injects one 7.5 KB script and serves each channel its own release. Beacons from visitors drive rollback. The upstream site is never touched."

**If asked about SPAs, CSP or compression.** "The runtime re-applies patches when the page re-renders, and every op is idempotent. The proxy decodes gzip, deflate and brotli, and adjusts the page's Content-Security-Policy so its one script and the beacons are allowed. If it can't decode a response, it passes it through unchanged."

### 16 · Models (0:30, full run)

**Say.** "We also asked whether learning helps tracking. We trained a 126.7-million-parameter language model from scratch with DiLoCo across 13 Kaggle groups, 26 T4 GPUs, syncing over a 2.3 MB/s tunnel: about 2.2 billion tokens. Rust runs it with candle and matches PyTorch to within 3e-6. And a 1.6-million-parameter matcher for tracking."

**If asked "why from scratch?" or "what architecture?"** "The brief asked for it. The layer design follows Gemma 3, as a reference only: we didn't use its weights or its tokenizer."

### 17 · Results (0:40)

**Say.** "We wrote each test rule down before running it, and used each test seed once. When an element loses its id, changes its text and moves, the heuristic relocates about a third. The learned tracker relocates 88 to 94 percent, with almost no wrong matches. On mixed changes it still makes more wrong matches, so by our own rule it is not the default; it ships opt-in. 192 Rust tests and 51 Python tests pass."

**Do.** Point at the hard-drift bars, then at the decision text.

**If asked "so the ML didn't work?"** "On the case the heuristic can't handle, it works: 34% becomes 89% on a fresh seed, with no wrong matches. We didn't make it the default because our pre-registered rule also requires it to be no worse on ordinary changes. The learned gate came within 2 points of that. We'd rather show you the rule we kept than a number we tuned on the test set."

### 18 · Close (0:15)

**Say.** "That is all seven features, end to end, without rebuilding the site. Every number comes from our design doc and model card, with the command that reproduces it. We can run any part of this live on the real binary."

**Do.** All seven rows should be ticked. Take questions.

## 6. Live segment on the real binary (3 min)

Everything here was run on this machine with the release build. The outputs below are the real ones, shortened. `demo.sh` prints each command before it runs it, from inside the project folder, so the judges see exactly what is typed.

Prepare it before the judges arrive: run `Website/live/demo.sh up`. That builds `pw` if needed, starts the shop on :8080 and the proxy on :8787, and snapshots the cart. To start over at any point, run `up` again.

| Step | Command | What the judges see |
|---|---|---|
| R1 | `demo.sh edit` | A headed Chromium window with the editor. Click "Byu now", type "Buy now", **Save draft patch**, then press Ctrl+C in the terminal. *Rehearse this first.* If anything goes wrong, run `demo.sh fix`, which writes the same patch. |
| R2 | `demo.sh check` | `fix-buy-label  /  valid  conf 1.000  locators agree` |
| R4 | `demo.sh preview` | The production dry run is **BLOCKED** by 2 reasons (not approved, no verification). Then `released 323ccee7… to preview/fix`. Open `http://127.0.0.1:8787/__pw/channel?name=preview/fix` and the button says "Buy now"; `…?name=production` still says "Byu now". |
| R5 | `demo.sh verify` | `pass visible=true`, `pass textContains Buy now`, `pass visual (0 of 916806 pixels … differ)` |
| R3 | `demo.sh review` | Self-approval: `Error: … was written by <you>. Another reviewer should approve it`. Then `approved by reviewer`. |
| R6 | `demo.sh ship` | `released 51b15cb4… to production (1 patches)` |
| R3 | `demo.sh merge` | `CONFLICT WriteWrite on /: buy-label-caps vs fix-buy-label`. The teammate drops theirs, then `added promo-banner`. |
| R6 | `demo.sh waive` | `released … to production (2 patches)`, with both waivers noted in the audit log |
| R6 | `demo.sh visitors` | Three headless visitors load the page. By the third, `production ->` has moved back to the previous release. Instead, you can reload `http://127.0.0.1:8787/` three times in a browser, waiting about 3 seconds each time. |
| R7 | `demo.sh audit` | `proxy auto-rollback … "evidence":"error beacons (patch -): Uncaught TypeError: Cannot read properties of null (reading 'textContent')"`, then `audit ok: 11 events` (one more if you used the editor) |
| R7 | `demo.sh tamper` | `Error: audit chain broken at line 9: hash does not match the event body (edited)`: the waiver line. Run `demo.sh restore` afterwards. |
| R2 | `demo.sh redesign` | The shop ships new markup: `fix-buy-label  /  relocated  conf 0.767  picked candidate 0` |

For the shortest convincing version (90 seconds), run `preview`, `verify` and then `waive` + `visitors`. The judges see isolation, a real browser check and a real automatic rollback. `demo.sh all` runs every step and pauses for Enter between them. Run `demo.sh down` when you finish.

Two things to know:

- In the live run the patch's page is `/`, because the sample shop is served at the root. On the web page it is `/cart`.
- Release ids are hashes of the release. They match the ones above only when the patch is byte-for-byte the same (the `fix` path).

## 7. If something goes wrong

| Problem | Do this |
|---|---|
| The globe or the training figure is blank | No WebGL or no internet. Carry on: nothing else depends on them. |
| A console chapter stops halfway | Click the same chapter button again. It continues from the first step that isn't done. **Reset** starts the console over. |
| Presenter keys do nothing | Click on an empty part of the page; the terminal input may have focus. Esc leaves the input. |
| The Presenter view window doesn't open | Allow pop-ups for the page, or use **N** for on-page notes. |
| `demo.sh up` says a port is in use | Run `demo.sh down`, or pick other ports: `SHOP_PORT=8090 PROXY_PORT=8797 demo.sh up`. Use the same variables on every later step. |
| `demo.sh edit` doesn't open a window | Run `demo.sh fix` and carry on. Say: "the editor writes this same file". |
| `demo.sh visitors` finds no headless Chromium | Run `./target/release/pw browser install`, or reload the proxied page three times in a browser. |
| The live run misbehaves in any other way | Return to the web console and say what happened. Every step there mirrors the live one. |

## 8. Questions to prepare for

Answer plainly. Say what is measured and what is not.

- **"Does it work on real websites?"** "The runtime and proxy work on any site we can put the proxy in front of, and we ran the full path on a sample shop. Our tracking numbers are on synthetic sites only. We haven't benchmarked tracking on real sites yet, and that's the next measurement."
- **"What can't it patch?"** "Inside shadow DOM, cross-origin iframes, and canvas. The proxy also buffers whole responses, so very large pages use memory."
- **"What stops a malicious patch?"** "The DSL has no script operation. Attributes starting with `on`, and `style`, `srcdoc` and `srcset`, are refused. Links and fetches must be same-site paths. Inserted HTML goes through ammonia. Production also needs a review by someone else and a passing verification, and anything waived is logged."
- **"What does it cost visitors?"** "One script tag and a 7.5 KB runtime, about 2.7 KB gzipped, plus one manifest request and one small beacon per page load."
- **"How do you know a relocation is right?"** "We don't trust it blindly. Every patch the editor makes carries an identity assertion, and the gate refuses heuristic relocations unless you pass `--accept-relocated`. Then production needs a passing verification of that exact content."
- **"How long did training take, and on what?"** "13 Kaggle groups of 2 T4s each, syncing through a coordinator on a laptop over a Cloudflare tunnel. The shipped weights are round 16: 138 merged deltas, about 2.2 billion tokens, over about 6.6 hours of logged events."
- **"Did you use 13 Kaggle accounts?"** "Yes, one account per group. We haven't confirmed that Kaggle's terms allow running accounts in parallel like this; `docs/ALIGNMENT.md` flags it as a risk. We stopped the run on 11 October." Don't defend it beyond that.
- **"Why is most of the code ML if the brief is a patching tool?"** "Smart element tracking is the hardest feature in the brief, and selectors alone fail on redesigns. We measured whether learning helps, kept the safe heuristic as the default, and published the evidence either way."
- **"Can I try it?"** "Yes. Type in the console, or press any Show button. The repository README has the quick start for the real CLI."

## 9. Where the numbers come from

| Number | Source |
|---|---|
| 192 Rust tests pass (4 browser tests run separately with `--ignored`) | `cargo test --workspace`, 2026-10-11 |
| 51 Python tests pass | `python -m unittest discover -s training/tests -t .`, 2026-10-11 |
| 7.5 KB runtime | `runtime/runtime.js` is 7,466 bytes (2,698 gzipped) |
| 0 of 916,806 pixels; relocated at 0.767; the auto-rollback error text | Live run of `Website/live/demo.sh`, 2026-10-11 |
| 508 pixels for a layout shift | `docs/DESIGN.md` §6.8 |
| Hard drift 34.0 → 88.8 (606), 32.4 → 94.2 (909), 31.8 → 88.4 (1414) | `docs/DESIGN.md` §6.10, §6.11, §6.13 |
| Mixed-drift objectives (404, 505, 808, 811, 1212, 1313) | `docs/DESIGN.md` §6.10, §6.11, §6.13 |
| LM confidence bar and test seeds 7 and 202 | `docs/DESIGN.md` §6.5 |
| 126.7M parameters, 2.19B tokens (estimate), 16 rounds, 138 deltas, perplexity 10.1 | `docs/MODELS.md`, `docs/REPORT-2026-10-11.md` |
| 2.3 MB/s link, 2,000-step syncs, 252 MB deltas | `docs/KAGGLE.md` |
| Similarity weights and thresholds on sheet 05 | `crates/pw-core/src/similarity.rs`, `crates/pw-core/src/relocate.rs` |
