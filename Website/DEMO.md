# PatchWork demo · Team Shotgun

Five minutes, four speakers, one story. A junior developer inherits a live website from a senior who has left the company, finds bugs in it, and has to fix them without rebuilding code nobody understands. Every speaker carries the story one step further.

- **The presentation** is `Website/index.html` in presenter mode. Press **P**. Its default run is the script below, and the bar and notes show who is speaking.
- **The live minute** runs `Website/live/demo.sh`: the real `pw` binary, a real proxy and real Chromium in front of a sample shop on the laptop.
- **Speaker 3's full live run-through** (section 5) shows every feature on the real binary in about seven minutes. Use it when the judges want to see more, or when you have a longer slot.

The console on the page is a simulation, and the page says so. Its commands and messages come from the real CLI, and its outputs were copied from runs of the binary. The live parts are real.

## 0. Quick start for the team

- **Team hub:** https://claude.ai/artifact/LiG5oifBLKudenYrtPYctA has a shared show clock with cues, each speaker's lines, done-marks, readiness checks and every command with a Copy button. Give each person their own link by adding `#john`, `#s2`, `#s3` or `#s4` to the address. To let teammates tick shared marks, invite them by email as Editors from its Share menu, with the public link off. Otherwise their ticks save only on their own phone.
- **On a new laptop:** install Rust (`cargo`) and `python3`, then run `git pull`, `cargo build --release -p pw-cli`, `./target/release/pw browser install`, and `Website/live/demo.sh doctor`. `doctor` checks the tools and ports, rehearses the live minute out of sight, and leaves it staged. Every line should show ✓.
- **For screen sharing:** present `http://127.0.0.1:8000/#share`. Screen-share mode hides the presenter bar, notes and pop-up messages; the clicker still works, **C** toggles the mode, and **V** opens the notes window privately. Use `Website/live/demo.sh shell` for the terminal: it shows a plain `shop $` prompt with no history. Share only the browser window or tab, never the whole screen.

## 1. Who does what

| Speaker | Role | Speaks | Drives | Takes questions on |
|---|---|---|---|---|
| **John Thrisokaran** · team leader | Tells the story, opens and closes | 0:00–0:56, 4:19–4:39 | The clicker for the whole page, and → after the live minute | The problem statement, scope, Kaggle, anything unassigned |
| **Speaker 2** · product | Plays the junior developer in the web console | 0:56–2:12 | Nothing (John clicks) | The editor, patch safety, review and the release gate |
| **Speaker 3** · engineering | The live minute, and the full live run-through if asked | 2:12–3:22 | The terminal and the shop tab | The proxy, the runtime, rollback, the audit log, performance |
| **Speaker 4** · research | The redesign, the models, the evidence | 3:22–4:19 | Clicks "Compound drift" | Tracking, the models, the evaluation |

Fill in the other three names before rehearsing. In the script, "[Speaker 2]" means "say their name".

## 2. Run of show

| Time | Speaker | On screen | Step | Shows |
|---|---|---|---|---|
| 0:00 | John | Page: cover | The story | |
| 0:35 | John | Page: "A typo should not need a redeploy" | Why PatchWork | |
| 0:56 | Speaker 2 | Page: console | Edit | R1 |
| 1:24 | Speaker 2 | Page: console | Preview | R4 (R2 runs on the way) |
| 1:40 | Speaker 2 | Page: console | Verify | R5 |
| 1:52 | Speaker 2 | Page: console | Review and release | R3 |
| 2:12 | Speaker 3 | **Terminal + shop tab** | Live on the real binary | R6, R7 |
| 3:22 | Speaker 4 | Page: tracking | The redesign | R2 |
| 3:47 | Speaker 4 | Page: models | Models | |
| 3:57 | Speaker 4 | Page: results | Results | |
| 4:19 | John | Page: the brief | Close | all seven ticked |
| 4:39 | | | About 20 seconds spare for hand-offs and applause | |

R1 to R7 are the problem statement's features: R1 visual patching, R2 smart element tracking, R3 collaborative editing, R4 isolated preview, R5 automated verification, R6 versioned deployment and rollback, R7 audit trail. The chips on the presenter bar fill in as each one is shown. The live step ticks R6 and R7 when John moves past it, so the close lands on the brief table with all seven ticked.

## 3. Setup

**T−30 minutes**

1. **Serve the page.** Run `cd Website && python3 -m http.server 8000`, then open `http://127.0.0.1:8000/`. Opening `index.html` as a file also works.
2. **Internet.** Open the page once on the venue Wi-Fi so the CDN files are cached. Without internet the script still works; only the two 3D figures stay blank.
3. **Two windows**, so Alt+Tab switches between them. Practise the switch.
   - Window A: the page, full screen (F11), 100% zoom.
   - Window B: a terminal in `Website/live` with a large font, beside a browser tab for the shop.
4. **Second screen, if you have one.** Press **P**, then **Presenter view ⧉**, and drag that window to the laptop. It shows each speaker's lines, the next step and a timer. With mirrored screens, keep the notes closed, or the judges will see them.

**T−10 minutes**

5. In the terminal, run `./demo.sh stage` (about 30 seconds; or `./demo.sh doctor`, which also checks the laptop). It starts the shop and the proxy and runs everything before the live minute: the fix is released to production, and the teammate's banner is merged but not released. It prints a three-line summary. Then run `./demo.sh shell` for the clean terminal you will share.
6. In the shop tab, open `http://127.0.0.1:8787/__pw/channel?name=production`. This puts the tab on production; a tab left on the preview channel from a rehearsal would hide the banner. It should say **Buy now**.

**T−2 minutes**

7. Open `http://127.0.0.1:8000/#share` (or reload it), which starts the presentation in screen-share mode. One teammate taps **Start** in the hub when John begins.

**Between runs:** run `./demo.sh stage` again, then reload the page. Rehearse at least twice with a stopwatch. The bar shows each step's target time and puts a wavy underline under the timer once you are more than 15 seconds behind.

## 4. The 5-minute script

The **Say** lines are the same text as the notes on the page. Speak them like a story, not a spec sheet.

### 1 · The story · John · 0:00–0:35

**Say.** "Good morning, everyone. I'm John Thrisokaran, team leader of Team Shotgun, and we chose the PatchWork problem statement. Let me start with a story. Imagine I'm a junior developer. I've inherited a website built by a senior developer who has since left the company, and it's already live. Now I find bugs. The 'Buy now' button says 'Byu now' and goes nowhere. A help link points to a page that moved. Small bugs, but they cost money every day, and I can't safely rebuild code I don't understand."

**Do.** John holds the clicker for the whole page.

### 2 · Why PatchWork · John · 0:35–0:56

**Say.** "That's what PatchWork is for. It lets me fix a live website through small, independent patches, without rebuilding the original application. It tracks the elements it changes, catches conflicting or broken patches, tests every change in isolation, and deploys safely, rolling back by itself if anything goes wrong. [Speaker 2] will show you how."

### 3 · Edit · Speaker 2 · 0:56–1:24 · R1

**Say.** "Let me be that junior developer for a minute. On the right is the live shop; on the left, our tool. I put PatchWork in front of the site, and visitors notice nothing. Then I open the visual editor, click the broken button, fix the text and point it at checkout. Watch: when I try a link to another website, it refuses. Save, and that's a patch file. No code, no rebuild."

**Do.** Talk over the animation; it ends when the editor saves.

### 4 · Preview · Speaker 2 · 1:24–1:40 · R4

**Say.** "Can I push it straight to customers? PatchWork says not yet: nobody has reviewed it or tested it. So it goes to a private preview. Same website, two versions: customers still see the typo, and I see the fix."

**Do.** The shop flips from preview to production and back by itself.

### 5 · Verify · Speaker 2 · 1:40–1:52 · R5

**Say.** "Now it tests itself. A real Chrome browser checks that the fix is there, then compares the page pixel by pixel with what customers see. Nothing else moved."

### 6 · Review and release · Speaker 2 · 1:52–2:12 · R3

**Say.** "Then a second pair of eyes. I'm not allowed to approve my own fix. My reviewer approves it, and if anyone edits the patch afterwards, that approval disappears. Now it goes live, as a version we can always return to. [Speaker 3] will show you what happens when someone skips all this."

### 7 · Live on the real binary · Speaker 3 · 2:12–3:22 · R6, R7

Speaker 3 drives; John leaves the page alone until the hand-off. The commands finish in about a second, so all the time is talking.

| Do | Say |
|---|---|
| Alt+Tab to the terminal and shop tab. Point at **Buy now**. | "This isn't a simulation any more. This is our actual Rust program in front of a real shop on this laptop, and the fix you just saw is live." |
| `./demo.sh waive` | "Now meet a teammate in a hurry. There's a weekend sale, so they push a promo banner and skip the review and the testing. PatchWork lets them, but it writes down who skipped what, and why." |
| Reload the shop tab **once**. | "Here's the banner. Looks fine, but it has quietly broken the shop's own JavaScript." |
| `./demo.sh visitors` | "Now three real visitors arrive. Their browsers hit the error and report it back. Five errors from three visitors, and PatchWork rolls the site back on its own." |
| Reload the shop tab. | "Banner gone, our fix still there." |
| `./demo.sh tamper` | "And everything is in a log that can't be edited quietly. Here it checks out. Now someone edits the log to hide the skipped review, and PatchWork points to the exact line that changed." |
| Alt+Tab back. John presses →. | "[Speaker 4], what happens when the website itself changes?" |

What the terminal shows, from the rehearsal:

- `waive`: `released … to production (2 patches)`, with both waivers noted.
- `visitors`: `production ->` steps back to the earlier release by the second visitor, then the logged event: `proxy auto-rollback … Uncaught TypeError: Cannot read properties of null (reading 'textContent')`.
- `tamper`: `audit ok: 11 events`, then `Error: audit chain broken at line 9: hash does not match the event body (edited)`.

Reload the shop tab only once after `waive`. Every page load is another visitor, and three visitors with errors trigger the rollback.

**If the live minute fails:** say "Let me show you the same thing in our console", Alt+Tab back, and click **Rollback (7)**, then **Audit (8)**, above the console. The same lines fit.

### 8 · The redesign · Speaker 4 · 3:22–3:47 · R2

**Say.** "Six months later, the agency redesigns the website. New class names, no IDs, everything wrapped in new boxes. A normal CSS selector breaks right here. PatchWork remembers each element the way you would: its text, its neighbours, where it sits on the page. So it finds the button again. And when too much has changed at once, it doesn't guess; it asks a person."

**Do.** Click **Compound drift** on "too much has changed".

### 9 · Models · Speaker 4 · 3:47–3:57

**Say.** "To handle those hard cases, we built our own models from scratch, including a 126-million-parameter language model trained across 26 Kaggle GPUs."

### 10 · Results · Speaker 4 · 3:57–4:19

**Say.** "And we tested them honestly: rules written before the tests, each test set used once. When an element changes everything at once, the simple method finds a third; our model finds about nine in ten. It's a little worse on everyday changes, so we kept it optional. Safety first. Back to John."

**Do.** Point at the hard-drift bars, then at the decision.

### 11 · Close · John · 4:19–4:39

**Say.** "So, back to our junior developer. The typo is fixed, the bad release rolled itself back, and every change is on record, all without touching the senior developer's code or rebuilding the site. That's all seven features from the problem statement. Thank you; we'd love your questions."

**Do.** All seven rows are ticked. Pass each question to its owner (section 1).

## 5. Speaker 3's live run-through: everything on the real binary

About seven minutes. It is the same story, told for real in the terminal, from an empty project to a redesign. Use it when a judge says "can we see that for real?", at a demo table, or as the main demo in a longer slot.

**Before you start:**
- Run `./demo.sh up` in `Website/live`. It wipes the demo and starts fresh, so the 5-minute run needs `./demo.sh stage` again afterwards.
- Open two browser tabs: the shop directly (`http://127.0.0.1:8080/`) and through PatchWork (`http://127.0.0.1:8787/`).
- Each command prints the exact `pw` command it runs.

| # | Feature | Run | Show in the browser | Say | You should see |
|---|---|---|---|---|---|
| 0 | Setup | `./demo.sh up` | Both tabs, side by side: identical | "This is a real shop on this laptop. Here it is directly, and here it is through PatchWork. Identical: nothing is patched, and nobody rebuilt anything." | `created PatchWork project`, `captured 32 elements`, `PatchWork proxy on …` |
| 1 | R1 visual patching | `./demo.sh edit` | A Chromium window opens with the PatchWork panel. Click **Byu now**, type **Buy now** in Text. Optionally set Click behaviour to "Navigate to a path on click" with `https://example.com`: a red error appears and Save is disabled. Change it to `/checkout/`. Click **Save draft patch**, then press Ctrl+C in the terminal. | "This is the editor our junior developer uses. Click the element, change it, save. If I try to send the button to another website, it refuses before I can even save." | `saved patches/edit-….yaml (draft …)`, then the patch file |
| 2 | R2 tracking | `./demo.sh check` | — | "Before anything ships, PatchWork makes sure the patch can still find its button. Every way of finding it agrees." | `valid  conf 1.000  locators agree` |
| 3 | R4 preview | `./demo.sh preview` | Open `http://127.0.0.1:8787/__pw/channel?name=preview/fix`: **Buy now**. Then `…?name=production`: **Byu now**. Leave the tab on production. | "Can I ship it to customers? Not yet: nobody has reviewed it and nothing has tested it. So it goes to a private preview. Same address; my browser sees the fix, and customers still see the typo." | the dry run `BLOCKED by 2 reason(s)`, then `released … to preview/fix` |
| 4 | R5 verification | `./demo.sh verify` | — | "A real Chrome browser, running in the background, checks the fix is there and compares every pixel outside the button with what customers see. Zero differences." | `pass visible=true`, `pass textContains Buy now`, `pass visual (0 of 916806 pixels … differ)` |
| 5 | R3 review | `./demo.sh review` | — | "I try to approve my own fix, and it refuses. A reviewer approves it, and that approval is tied to this exact version of the patch." | `Error: … was written by <you>`, then `approved by reviewer` |
| 6 | R6 release | `./demo.sh ship` | Reload the PatchWork tab: **Buy now**. Click it: the checkout page. Go back. | "Now it's live for everyone, as a release we can always go back to. And the button finally does its job." | `released … to production (1 patches)` |
| 7 | R3 merge | `./demo.sh merge` | — | "Meanwhile a teammate fixed the same button on their own branch, in capitals. PatchWork catches the clash and won't merge until we agree. They drop theirs, and their promo banner comes in." | `CONFLICT WriteWrite … buy-label-caps vs …`, then `added promo-banner` |
| 8 | R6 skipping the gate | `./demo.sh waive` | Reload the PatchWork tab **once**: the banner. | "The teammate is in a hurry for a weekend sale, so they skip review and testing. PatchWork allows it, but records who skipped what and why. The banner is up, and it has quietly broken the shop's own JavaScript." | `released … (2 patches)`, `waived the review gate`, `waived the verification gate` |
| 9 | R6 rollback | `./demo.sh visitors` | Reload: banner gone, **Buy now** still there. | "Three real visitors arrive. Their browsers hit the error and report it. Five errors from three visitors, and PatchWork rolls production back by itself." | `production ->` steps back; `proxy auto-rollback … Uncaught TypeError …` |
| 10 | R7 audit trail | `./demo.sh audit` | — | "Here's the whole story in order, with who did what: the release, the test, the review, the merge, the skipped checks, and the automatic rollback, with the browser's actual error." | the log, then `audit ok: 11 events` (12 if you used the editor) |
| 11 | R7 tamper check | `./demo.sh tamper`, then `./demo.sh restore` | — | "Each line is chained to the one before it. Someone edits the log to hide the skipped review, and PatchWork names the exact line." | `audit ok`, then `audit chain broken at line 9` (line 10 if you used the editor) |
| 12 | R2 redesign | `./demo.sh redesign` | Reload: still **Buy now**. | "Finally, the agency ships a redesign: new class names, no ID, a new wrapper. Our selector no longer matches, but PatchWork finds the button again from what it remembers, and the live fix never dropped." | `relocated  conf 0.767  picked candidate 0` |
| — | Finish | `./demo.sh down` | — | | `stopped` |

Things to know:
- Each step needs the ones before it, so go in order. To show one feature again, start from `./demo.sh up`.
- `./demo.sh all` runs the whole list and waits for Enter before each step. It uses `fix` instead of the editor.
- The editor opens a visible browser window. We've tested it only in headless runs, so rehearse it on the demo laptop. If it misbehaves, `./demo.sh fix` writes the same patch: "The editor writes exactly this file."
- Only `./demo.sh visitors` and the shop tab's reloads count as visitors. Reload just once between `waive` and `visitors`.

## 6. Questions, and who takes them

Keep answers short and honest. Say what you measured and what you didn't.

- **Speaker 3 · "Does it work on real websites?"** "The proxy and the runtime work on any site you can put the proxy in front of; you just watched it on a real shop. Our tracking accuracy numbers are from synthetic sites, though. Measuring on real sites is our next step."
- **Speaker 3 · "What can't it patch?"** "Anything inside shadow DOM, cross-origin iframes and canvas. And the proxy holds each page in memory, so very large pages cost memory."
- **Speaker 3 · "What does it cost visitors?"** "One extra script tag: a 7.5 KB runtime, under 3 KB compressed, plus one small request and one beacon per page."
- **Speaker 3 · "Why five errors from three visitors?"** "So one broken browser can't roll back the whole site. It's a setting in `pw.toml`, and each release can roll back only once."
- **Speaker 3 · "Can someone rewrite the whole log?"** "Only if they can also rewrite the small HEAD file next to it, which lives on the same machine. Storing that HEAD somewhere else, like a git commit or a server, closes the gap."
- **Speaker 2 · "Could a patch be used to attack the site?"** "We designed against that. There's no way to write JavaScript in a patch, no event-handler attributes, no links to other sites, and any HTML is cleaned first. Production also needs someone else's approval and a passing test, and any skip is logged."
- **Speaker 2 · "What else can the editor change?"** "Text, colours, font size, classes, links, visibility, moving elements, what a click does, and inserting or replacing HTML. All of it goes through the same safety rules."
- **Speaker 2 · "Who is 'reviewer'? Is that a real login?"** "Not yet. Names are declared, not authenticated, and we say so in our docs. Signing each action is the next step."
- **Speaker 4 · "How do you know a relocation is right?"** "We don't just trust it. Every patch from the editor carries a check of its own text, heuristic relocations are refused unless you explicitly allow them, and production needs a passing test of that exact version."
- **Speaker 4 · "So the ML didn't work?"** "It did on the case that matters most: when everything about an element changes, it goes from a third to about nine in ten, with no wrong matches. We kept it optional because our own rule, written before the test, says it must also be no worse on everyday changes, and it came within two points. We'd rather keep the rule than bend it."
- **Speaker 4 · "Why train from scratch? What architecture?"** "The problem statement asked for it. The layer design follows Gemma 3 as a reference; we didn't use its weights or its tokenizer."
- **John · "Did you use 13 Kaggle accounts?"** "Yes, one per training group. We haven't confirmed whether Kaggle's terms allow that, and our own docs flag it as a risk. We stopped the run on 11 October." Leave it there.
- **John · "Why so much ML in a patching tool?"** "Tracking elements through a redesign is the hardest part of the problem, and selectors alone fail. We measured whether learning helps, kept the safe method as the default, and published the results either way."

## 7. If something goes wrong

| Problem | Do this |
|---|---|
| The live minute fails | Back to the page; click **Rollback (7)** and **Audit (8)** above the console. |
| `./demo.sh stage` says a port is in use | Run `./demo.sh down`, or use other ports: `SHOP_PORT=8090 PROXY_PORT=8797 ./demo.sh stage`. Use the same variables on every later command, and point the shop tab at the new proxy port. |
| No banner after `waive` | The tab is on the preview channel. Open `http://127.0.0.1:8787/__pw/channel?name=production`. |
| The rollback happened before `visitors` | The tab was reloaded too often. Say "the visitors beat us to it", run `./demo.sh visitors` anyway, then reload. |
| `visitors` finds no headless Chromium | Run `./target/release/pw browser install` beforehand. On stage, reload the shop tab two more times instead, about 3 seconds apart. |
| A console chapter stops halfway | Click the same chapter button again; it carries on from where it stopped. |
| Presenter keys do nothing | Click an empty part of the page; the terminal input may have focus. |
| The globe or the training figure is blank | No WebGL or no internet. Nothing in the script depends on them. |

## 8. The 8-minute run

Press **5 min** on the presenter bar to switch to **Full**: 18 steps, about 7:45, the same story and the same speakers by section. It adds the problem statement's checklist, the patch check, the merge conflict, what a patch file looks like, the hard tracking cases and the architecture, and it shows the rollback and the audit in the web console. Speaker 3 can still do the live minute after its Audit step. Every line is in the page's notes (N) and the presenter view. To open the page already presenting, add `#present` (or `#present-full`) to its address.

## 9. Controls

| Key | Does |
|---|---|
| **P** | Start or stop presenter mode |
| **→**, **Space**, **Page Down** | Next step. Most clickers send Page Down. On a console step, let the typing finish first |
| **←**, **Page Up** | Previous step |
| **N** · **T** · **Esc** | Speaker notes · reset the timer · leave presenter mode |

## 10. Where the numbers come from

| Number | Source |
|---|---|
| 192 Rust tests pass (4 browser tests run separately with `--ignored`) | `cargo test --workspace`, 2026-10-11 |
| 51 Python tests pass | `python -m unittest discover -s training/tests -t .`, 2026-10-11 |
| 7.5 KB runtime | `runtime/runtime.js` is 7,466 bytes (2,698 gzipped) |
| 0 of 916,806 pixels; relocated at 0.767; the rollback error text; the tampered line | Rehearsals of `Website/live/demo.sh`, 2026-10-11 |
| "A third" to "about nine in ten": 34.0 → 88.8 (seed 606), 32.4 → 94.2 (909), 31.8 → 88.4 (1414) | `docs/DESIGN.md` §6.10, §6.11, §6.13 |
| "Within two points": 425 against 427 on seed 1212 | `docs/DESIGN.md` §6.13 |
| 126.7M parameters, 26 T4 GPUs in 13 groups, 2.3 MB/s link | `docs/MODELS.md`, `docs/KAGGLE.md` |
| Rust matches Python to within 3e-6 | `docs/DESIGN.md` §7, item 6 |
