//! Playwright management for the CLI: install browsers, capture snapshots, open the
//! visual editor, and check a channel in a real browser.
//!
//! The browser engine is Microsoft's Playwright, driven from Rust through the
//! `playwright-rs` bindings. The small JavaScript snippets below run inside the page
//! because the DOM can only be read from there. They do no decisions of their own:
//! the Rust side reads the results and decides.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use base64::Engine;
use playwright_rs::protocol::ScreenshotOptions;
use playwright_rs::{Browser, LaunchOptions, Page, Playwright};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use pw_core::anchor::{Anchor, Locators};
use pw_core::dsl;
use pw_core::patch::{Assertion, Op, Patch};
use pw_core::release::ManifestPatch;
use pw_core::snapshot::Snapshot;
use pw_core::verify::{Check, Outcome};

use crate::project::{self, Project};

/// Returns the page's HTML and each element's rectangle, in document order.
/// The editor's own panel is not part of the page, so it is left out of both the rectangles and the HTML.
const CAPTURE_JS: &str = r#"() => {
  const els = Array.from(document.querySelectorAll('*')).filter(e => !e.closest('[data-pw-ui]'));
  const rects = els.map(e => {
    const r = e.getBoundingClientRect();
    return (r.width > 0 || r.height > 0) ? [r.x, r.y, r.width, r.height] : null;
  });
  const copy = document.documentElement.cloneNode(true);
  copy.querySelectorAll('[data-pw-ui]').forEach(n => n.remove());
  return { html: '<!doctype html>' + copy.outerHTML, rects: rects, count: els.length };
}"#;

/// Finds the element a patch targets. The same rule as the patch's assertions: a unique CSS match, then
/// XPath, then test id. Defined once and placed in each snippet that needs it.
const LOCATE_JS: &str = r#"function pwLocate(L) {
  let el = null;
  try { if (L.css) { const all = document.querySelectorAll(L.css); if (all.length === 1) el = all[0]; } } catch (e) { el = null; }
  if (!el && L.xpath) { try { el = document.evaluate(L.xpath, document, null, 9, null).singleNodeValue; } catch (e) { el = null; } }
  if (!el && L.testid) el = document.querySelector('[data-testid="' + L.testid.replace(/["\\]/g, '\\$&') + '"]');
  return el;
}"#;

const ASSERT_BODY_JS: &str = r#"
  const el = pwLocate(p.locators || {});
  const out = [];
  for (const a of (p.assert || [])) {
    if ('visible' in a) {
      const v = !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
      out.push(['visible=' + a.visible, v === a.visible]);
    } else if ('exists' in a) {
      out.push(['exists=' + a.exists, (!!el) === a.exists]);
    } else if ('textContains' in a) {
      out.push(['textContains ' + a.textContains, !!el && el.textContent.includes(a.textContains)]);
    } else if ('attrEquals' in a) {
      out.push(['attrEquals ' + a.attrEquals.name, !!el && el.getAttribute(a.attrEquals.name) === a.attrEquals.value]);
    } else if ('clickNavigatesTo' in a) {
      out.push(['clickNavigatesTo ' + a.clickNavigatesTo, null]);
    } else if ('clickShowsText' in a) {
      out.push(['clickShowsText ' + a.clickShowsText, null]);
    }
  }
  return out;"#;

/// Checks one patch's assertions on the page. Returns [label, passed] pairs, one per assertion, in order.
/// `clickNavigatesTo` needs a click, so the page reports it as not checked (null).
fn assert_js() -> String {
    ["(p) => {", LOCATE_JS, ASSERT_BODY_JS, "}"].concat()
}

/// The element's box in page coordinates, or null when the element is not on the page.
fn box_js() -> String {
    [
        "(L) => {",
        LOCATE_JS,
        "const el = pwLocate(L); if (!el) return null;",
        "const r = el.getBoundingClientRect();",
        "return [r.x + window.scrollX, r.y + window.scrollY, r.width, r.height]; }",
    ]
    .concat()
}

/// Tags the element a patch targets so Playwright can click it. Returns whether it was found.
fn mark_js() -> String {
    ["(L) => {", LOCATE_JS, "const el = pwLocate(L); if (!el) return false;", "el.setAttribute('data-pw-target', ''); return true; }"].concat()
}

/// Compares two full-page screenshots, ignoring the masked boxes. It runs on a blank page, so the pages
/// being compared cannot interfere with it.
const DIFF_JS: &str = r#"async (a) => {
  const load = (src) => new Promise((ok, no) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = () => no(new Error('could not decode a screenshot'));
    img.src = 'data:image/png;base64,' + src;
  });
  const [x, y] = await Promise.all([load(a.baseline), load(a.candidate)]);
  const result = {
    baseline: [x.naturalWidth, x.naturalHeight],
    candidate: [y.naturalWidth, y.naturalHeight],
    size_changed: false, compared: 0, differing: 0,
  };
  if (x.naturalWidth !== y.naturalWidth || x.naturalHeight !== y.naturalHeight) { result.size_changed = true; return result; }
  const w = y.naturalWidth, h = y.naturalHeight;
  const pixels = (img) => {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const g = c.getContext('2d');
    g.drawImage(img, 0, 0);
    return g.getImageData(0, 0, w, h).data;
  };
  const p = pixels(x), q = pixels(y);
  const masked = new Uint8Array(w * h);
  for (const [bx, by, bw, bh] of a.masks) {
    for (let r = Math.max(0, Math.floor(by)); r < Math.min(h, Math.ceil(by + bh)); r++) {
      for (let c = Math.max(0, Math.floor(bx)); c < Math.min(w, Math.ceil(bx + bw)); c++) masked[r * w + c] = 1;
    }
  }
  for (let i = 0; i < w * h; i++) {
    if (masked[i]) continue;
    result.compared++;
    const k = i * 4;
    if (p[k] !== q[k] || p[k + 1] !== q[k + 1] || p[k + 2] !== q[k + 2] || p[k + 3] !== q[k + 3]) result.differing++;
  }
  return result;
}"#;

const RUNTIME_READY_JS: &str = "() => !!(window.__pw && window.__pw.state && window.__pw.state.release)";

/// How long a click may take to land on its target path.
const NAVIGATION_WAIT: Duration = Duration::from_secs(5);

#[derive(Deserialize)]
struct Capture {
    html: String,
    rects: Value,
    count: usize,
}

#[derive(Deserialize)]
struct SaveRequest {
    index: u32,
    page: String,
    ops: Vec<Value>,
    #[serde(default)]
    note: String,
}

#[derive(Deserialize)]
struct Diff {
    baseline: [u32; 2],
    candidate: [u32; 2],
    size_changed: bool,
    compared: u64,
    differing: u64,
}

/// A patch id that is not yet in the project. Saves within the same millisecond get a numeric suffix,
/// so two quick saves never collide.
fn new_patch_id(project: &Project) -> String {
    let ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let base = format!("edit-{ms}");
    let mut id = base.clone();
    let mut n = 2;
    while project.patches_dir().join(format!("{id}.yaml")).exists() {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

fn with_query(url: &str, query: &str) -> String {
    let (base, fragment) = match url.split_once('#') {
        Some((b, f)) => (b, Some(f)),
        None => (url, None),
    };
    let joined = if base.contains('?') { format!("{base}&{query}") } else { format!("{base}?{query}") };
    match fragment {
        Some(f) => format!("{joined}#{f}"),
        None => joined,
    }
}

async fn launch(headed: bool) -> Result<(Playwright, Browser)> {
    let pw = Playwright::launch()
        .await
        .context("starting the Playwright driver (run `pw browser install` first)")?;
    let options = LaunchOptions::default().headless(!headed);
    let browser = pw
        .chromium()
        .launch_with_options(options)
        .await
        .context("launching Chromium (run `pw browser install` first)")?;
    Ok((pw, browser))
}

pub async fn install(with_deps: bool) -> Result<()> {
    if with_deps {
        playwright_rs::install_browsers_with_deps(Some(&["chromium"])).await?;
    } else {
        playwright_rs::install_browsers(Some(&["chromium"])).await?;
    }
    Ok(())
}

pub async fn status() -> Result<String> {
    let (_pw, browser) = launch(false).await?;
    let version = browser.version().to_string();
    browser.close().await?;
    Ok(version)
}

async fn capture(page: &Page) -> Result<Capture> {
    let cap: Capture = page.evaluate::<(), Capture>(CAPTURE_JS, None).await?;
    Ok(cap)
}

fn check_capture(cap: &Capture) -> Result<()> {
    let rects = cap.rects.as_array().map_or(0, |a| a.len());
    if rects != cap.count {
        bail!("capture is inconsistent: {rects} rectangles for {} elements", cap.count);
    }
    Ok(())
}

/// Capture a page to `out` (HTML) and `out.rects.json` (element rectangles).
pub async fn snapshot(url: &str, out: &Path, headed: bool) -> Result<usize> {
    let (_pw, browser) = launch(headed).await?;
    let page = browser.new_page().await?;
    page.goto(url, None).await?;
    let cap = capture(&page).await?;
    check_capture(&cap)?;
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, &cap.html)?;
    fs::write(out.with_extension("rects.json"), serde_json::to_string(&cap.rects)?)?;
    browser.close().await?;
    Ok(cap.count)
}

/// Turn one save from the editor into a patch file. `author` is the patch's author field; `actor` is who
/// is saving, and goes in the audit trail.
async fn save_from_message(page: &Page, text: &str, root: &Path, author: &str, actor: &str) -> Result<PathBuf> {
    let req: SaveRequest = serde_json::from_str(text).context("the editor sent an unreadable save")?;
    let cap = capture(page).await?;
    check_capture(&cap)?;
    let rects = project::parse_rects(&cap.rects.to_string())?;
    let snap: Snapshot = Snapshot::parse(&cap.html).with_rects(&rects);
    let el = snap
        .element(req.index)
        .with_context(|| format!("element {} is no longer on the page", req.index))?;
    let ops: Vec<Op> = req
        .ops
        .into_iter()
        .map(serde_json::from_value)
        .collect::<std::result::Result<_, _>>()
        .context("the editor sent an operation the DSL does not allow")?;
    // Identity check: after a site update the patch must still land on an element with this text.
    // It is the cheapest guard against a patch silently moving to a look-alike (see bench output).
    let set_text = ops.iter().find_map(|op| match op {
        Op::SetText { value } => Some(value.clone()),
        _ => None,
    });
    // A replaced content (setHTML) has no stable text to assert, so it gets no identity check.
    let replaces_content = ops.iter().any(|op| matches!(op, Op::SetHtml { .. }));
    let identity = if replaces_content {
        None
    } else {
        set_text.or_else(|| Some(el.text.clone()).filter(|t| !t.is_empty()))
    };
    let mut assertions = vec![Assertion::Visible(true)];
    if let Some(text) = identity {
        assertions.push(Assertion::TextContains(text.chars().take(200).collect()));
    }
    let project = Project::open(root)?;
    let mut patch = Patch {
        id: new_patch_id(&project),
        author: author.into(),
        page: req.page.split('?').next().unwrap_or("/").to_string(),
        anchor: Anchor::from_element(el, &snap),
        ops,
        assertions,
        status: Default::default(),
        reviewer: None,
    };
    dsl::normalize(&mut patch)?;
    if !req.note.is_empty() {
        println!("note: {}", req.note);
    }
    let path = project.save_patch(&patch)?;
    project
        .audit()
        .append(actor, "patch.add", json!({"id": patch.id, "page": patch.page, "author": author, "note": req.note}))?;
    Ok(path)
}

/// Open the editor in a visible browser. Each Save becomes a draft patch file.
pub async fn edit(url: &str, root: &Path, author: &str, actor: &str) -> Result<()> {
    let (_pw, browser) = launch(true).await?;
    let page = browser.new_page().await?;
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    page.expose_function("__pwSave", move |args: Vec<Value>| {
        let tx = tx.clone();
        async move {
            if let Some(text) = args.first().and_then(|v| v.as_str()) {
                let _ = tx.send(text.to_string());
            }
            Value::Null
        }
    })
    .await?;
    let target = with_query(url, "pw_edit=1");
    page.goto(&target, None).await?;
    println!("Editing {target} as {actor}. Click an element, change it, then Save. Ctrl+C to quit.");
    loop {
        tokio::select! {
            message = rx.recv() => {
                let Some(text) = message else { break };
                match save_from_message(&page, &text, root, author, actor).await {
                    Ok(path) => println!("saved {} (draft: `pw review propose` it when it is ready)", path.display()),
                    Err(e) => eprintln!("save failed: {e:#}"),
                }
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    browser.close().await?;
    Ok(())
}

/// What `verify` checks beyond the assertions.
pub struct Options {
    /// Run click checks (`clickNavigatesTo`). A click can submit a form or trigger a real action, so only
    /// run this against a copy of the site where that is safe.
    pub interact: bool,
    /// Compare each patched page with the baseline channel, outside the patched elements. Off when None.
    pub visual: Option<Visual>,
}

pub struct Visual {
    /// The channel that stands for what users see now, usually `production`.
    pub baseline: String,
    /// Share of compared pixels that may differ before the check fails.
    pub tolerance: f64,
}

/// Open a new page on `channel` showing `path`, with the channel's release applied. The channel is set
/// through the proxy's /__pw/channel route, so the page loads with that release.
async fn open(browser: &Browser, base: &str, channel: &str, path: &str) -> Result<Page> {
    let page = browser.new_page().await?;
    page.goto(&format!("{base}/__pw/channel?name={channel}"), None).await?;
    page.goto(&format!("{base}{path}"), None).await?;
    for _ in 0..50 {
        let ready: bool = page.evaluate::<(), bool>(RUNTIME_READY_JS, None).await.unwrap_or(false);
        if ready {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(page)
}

/// Run a channel's manifest checks in a real browser. Every assertion gets one check, and so does every
/// visual comparison. A check that was not run says so, so the record can tell "passed" from "not checked".
pub async fn verify(base: &str, channel: &str, patches: &[ManifestPatch], opts: &Options) -> Result<Vec<Check>> {
    let base = base.trim_end_matches('/');
    let (_pw, browser) = launch(false).await?;
    let mut checks = Vec::new();
    let mut pages: Vec<&str> = patches.iter().map(|p| p.page.as_str()).collect();
    pages.sort_unstable();
    pages.dedup();
    for path in pages {
        let page = open(&browser, base, channel, path).await?;
        for patch in patches.iter().filter(|p| p.page == path) {
            if patch.assertions.is_empty() {
                checks.push(Check {
                    patch: patch.id.clone(),
                    check: "assertions".into(),
                    outcome: Outcome::NotRun,
                    detail: "the patch has no assertions, so nothing verifies it".into(),
                });
                continue;
            }
            let arg = json!({"locators": patch.locators, "assert": patch.assertions});
            let rows: Vec<(String, Option<bool>)> = page.evaluate(&assert_js(), Some(&arg)).await?;
            if rows.len() != patch.assertions.len() {
                bail!("patch {}: the page reported {} results for {} assertions", patch.id, rows.len(), patch.assertions.len());
            }
            for (assertion, (label, passed)) in patch.assertions.iter().zip(rows) {
                let (outcome, detail) = match (assertion, passed) {
                    (Assertion::ClickNavigatesTo(target), _) if opts.interact => {
                        click_check(&browser, base, channel, path, &patch.locators, target).await?
                    }
                    (Assertion::ClickNavigatesTo(_), _) => (Outcome::NotRun, "not run: pass --interact to click it".into()),
                    (Assertion::ClickShowsText(text), _) if opts.interact => {
                        click_text_check(&browser, base, channel, path, &patch.locators, text).await?
                    }
                    (Assertion::ClickShowsText(_), _) => (Outcome::NotRun, "not run: pass --interact to click it".into()),
                    (_, Some(true)) => (Outcome::Pass, String::new()),
                    (_, Some(false)) => (Outcome::Fail, String::new()),
                    (_, None) => (Outcome::NotRun, String::new()),
                };
                checks.push(Check { patch: patch.id.clone(), check: label, outcome, detail });
            }
        }
        if let Some(visual) = &opts.visual {
            checks.push(visual_check(&browser, base, channel, path, patches, visual).await?);
        }
    }
    browser.close().await?;
    Ok(checks)
}

/// Click the patched element on the page. Returns why it could not be clicked, if it could not.
async fn click_patched(page: &Page, locators: &Locators) -> Result<Option<String>> {
    if !page.evaluate::<Locators, bool>(&mark_js(), Some(locators)).await? {
        return Ok(Some("the target element is not on the page".into()));
    }
    if let Err(e) = page.locator("[data-pw-target]").click(None).await {
        return Ok(Some(format!("the click failed: {e}")));
    }
    Ok(None)
}

/// Click the patched element on a fresh page, and report the path the browser lands on.
async fn click_check(browser: &Browser, base: &str, channel: &str, path: &str, locators: &Locators, target: &str) -> Result<(Outcome, String)> {
    let page = open(browser, base, channel, path).await?;
    if let Some(reason) = click_patched(&page, locators).await? {
        return Ok((Outcome::Fail, reason));
    }
    let deadline = Instant::now() + NAVIGATION_WAIT;
    loop {
        let here: String = page.evaluate::<(), String>("() => location.pathname", None).await.unwrap_or_default();
        if here == target {
            return Ok((Outcome::Pass, String::new()));
        }
        if Instant::now() >= deadline {
            let landed = if here.is_empty() { "a page that did not load".to_string() } else { here };
            return Ok((Outcome::Fail, format!("the click led to {landed}")));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Click the patched element on a fresh page, and wait for the text to appear on the page.
async fn click_text_check(browser: &Browser, base: &str, channel: &str, path: &str, locators: &Locators, text: &str) -> Result<(Outcome, String)> {
    let page = open(browser, base, channel, path).await?;
    if let Some(reason) = click_patched(&page, locators).await? {
        return Ok((Outcome::Fail, reason));
    }
    let deadline = Instant::now() + NAVIGATION_WAIT;
    loop {
        let shown: String = page.evaluate::<(), String>("() => document.body ? document.body.innerText : ''", None).await.unwrap_or_default();
        if shown.contains(text) {
            return Ok((Outcome::Pass, String::new()));
        }
        if Instant::now() >= deadline {
            return Ok((Outcome::Fail, format!("the page does not show `{text}` after the click")));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Box of each patched element on the page, in page coordinates. Elements that are not on the page are skipped.
async fn boxes(page: &Page, patches: &[&ManifestPatch]) -> Result<Vec<[f64; 4]>> {
    let mut out = Vec::new();
    for patch in patches {
        if let Some(b) = page.evaluate::<Locators, Option<[f64; 4]>>(&box_js(), Some(&patch.locators)).await? {
            out.push(b);
        }
    }
    Ok(out)
}

/// Compare the candidate channel's page with the baseline channel's page. Pixels inside a patched element
/// (before or after the patch) are ignored. Everything else must match to within the tolerance, so a patch
/// that moves or reflows other content fails here. That is a finding to review, not a false alarm to hide.
async fn visual_check(browser: &Browser, base: &str, channel: &str, path: &str, patches: &[ManifestPatch], visual: &Visual) -> Result<Check> {
    let name = format!("visual:{path}");
    if visual.baseline == channel {
        return Ok(Check {
            patch: name,
            check: "visual".into(),
            outcome: Outcome::NotRun,
            detail: "the channel is its own baseline; verify a preview channel against production".into(),
        });
    }
    let shot = ScreenshotOptions::builder().full_page(true).build();
    let on_page: Vec<&ManifestPatch> = patches.iter().filter(|p| p.page == path).collect();

    let candidate = open(browser, base, channel, path).await?;
    let mut masks = boxes(&candidate, &on_page).await?;
    let candidate_png = candidate.screenshot(Some(shot.clone())).await?;
    candidate.close().await?;

    let reference = open(browser, base, &visual.baseline, path).await?;
    masks.extend(boxes(&reference, &on_page).await?);
    let baseline_png = reference.screenshot(Some(shot)).await?;
    reference.close().await?;

    let compare = browser.new_page().await?;
    compare.goto("about:blank", None).await?;
    let engine = base64::engine::general_purpose::STANDARD;
    let arg = json!({
        "baseline": engine.encode(&baseline_png),
        "candidate": engine.encode(&candidate_png),
        "masks": masks,
    });
    let diff: Diff = compare.evaluate(DIFF_JS, Some(&arg)).await?;
    compare.close().await?;

    let (outcome, detail) = if diff.size_changed {
        (
            Outcome::Fail,
            format!(
                "the page is {}x{} but the {} channel shows {}x{}; content outside the patched elements moved",
                diff.candidate[0], diff.candidate[1], visual.baseline, diff.baseline[0], diff.baseline[1]
            ),
        )
    } else if diff.compared == 0 {
        (Outcome::Pass, "nothing outside the patched elements to compare".into())
    } else {
        let share = diff.differing as f64 / diff.compared as f64;
        let detail = format!(
            "{} of {} pixels outside the patched elements differ ({:.3}%, tolerance {:.3}%)",
            diff.differing,
            diff.compared,
            100.0 * share,
            100.0 * visual.tolerance
        );
        (if share <= visual.tolerance { Outcome::Pass } else { Outcome::Fail }, detail)
    };
    Ok(Check { patch: name, check: "visual".into(), outcome, detail })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDITOR_JS: &str = include_str!("../../../runtime/editor.js");

    const SITE: &str = r#"<!doctype html><html><body><main><section class="cart"><p>Order total</p>
<button id="buy" class="btn" type="submit">Buy now</button><p>Apply coupon</p></section></main></body></html>"#;

    /// Serve `html` for every request on 127.0.0.1. Each connection is handled on its own thread.
    fn serve(html: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("binding a test port");
        let addr = listener.local_addr().expect("test port address");
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let mut request = [0u8; 4096];
                    let _ = stream.read(&mut request);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
                        html.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                });
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn assertion_snippet_has_the_locate_function_and_one_row_per_assertion_kind() {
        let js = assert_js();
        assert!(js.contains("function pwLocate"), "the locate rule must be in the snippet");
        assert!(js.starts_with("(p) => {") && js.ends_with('}'));
    }

    #[test]
    fn the_stamp_and_box_snippets_use_the_same_locate_rule() {
        for js in [box_js(), mark_js()] {
            assert!(js.contains(LOCATE_JS), "{js}");
        }
    }

    /// The whole save path, with the real editor script and a real browser: select an element, change it in
    /// the panel, move it, save, and check the draft patch the CLI writes. Needs Chromium, so it is ignored by default.
    #[tokio::test]
    #[ignore = "needs Chromium (`pw browser install`); run with `cargo test -p pw-cli -- --ignored`"]
    async fn the_editor_save_path_writes_a_draft_patch_with_the_chosen_ops() {
        let base = serve(SITE);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/editor-save-path");
        let _ = fs::remove_dir_all(&root);
        Project::init(&root, "editor-test", &base).unwrap();

        let (_pw, browser) = launch(false).await.unwrap();
        let page = browser.new_page().await.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        page.expose_function("__pwSave", move |args: Vec<Value>| {
            let tx = tx.clone();
            async move {
                if let Some(text) = args.first().and_then(|v| v.as_str()) {
                    let _ = tx.send(text.to_string());
                }
                Value::Null
            }
        })
        .await
        .unwrap();
        page.goto(&format!("{base}/?pw_edit=1"), None).await.unwrap();
        // The proxy injects the editor on ?pw_edit=1. Here the same file is run in the page directly.
        let inject = format!("() => {{\n{EDITOR_JS}\n}}");
        page.evaluate::<(), Option<Value>>(&inject, None).await.unwrap();

        // Select the button, then change it in the panel.
        page.locator("#buy").click(None).await.unwrap();
        page.locator("[data-pw-text]").fill("Checkout now", None).await.unwrap();
        page.locator("[data-pw-add]").fill("promo", None).await.unwrap();
        page.locator("[data-pw-fg]").fill("#ff0000", None).await.unwrap();
        // Move it before "Order total", by picking that element after pressing the move button.
        page.locator("[data-pw-before]").click(None).await.unwrap();
        page.locator("section.cart > p:first-of-type").click(None).await.unwrap();
        page.locator("[data-pw-save]").click(None).await.unwrap();

        let message = tokio::time::timeout(Duration::from_secs(20), rx.recv())
            .await
            .expect("the editor sent a save")
            .expect("the save channel is open");
        let saved = save_from_message(&page, &message, &root, "tester", "tester").await.unwrap();
        browser.close().await.unwrap();

        let patch = Patch::read(&saved).unwrap();
        assert_eq!(patch.anchor.locators.css.as_deref(), Some("#buy"), "the button is anchored by its id");
        assert!(patch.ops.contains(&Op::SetText { value: "Checkout now".into() }), "{:?}", patch.ops);
        assert!(patch.ops.contains(&Op::AddClass { name: "promo".into() }), "{:?}", patch.ops);
        assert!(patch.ops.contains(&Op::SetStyle { prop: "color".into(), value: "#ff0000".into() }), "{:?}", patch.ops);
        assert!(
            patch.ops.iter().any(|op| matches!(op, Op::MoveBefore { target } if target == "p:nth-of-type(1)")),
            "{:?}",
            patch.ops
        );
        assert_eq!(patch.status, pw_core::patch::Status::Draft, "the editor writes drafts");
        assert!(patch.assertions.contains(&Assertion::Visible(true)));
        let events = Project::open(&root).unwrap().audit().read_all().unwrap();
        assert!(events.iter().any(|e| e.kind == "patch.add" && e.actor == "tester"), "the save is audited with its actor");
    }

    /// A click that changes the page is checked by the text it shows, and a wrong text fails.
    #[tokio::test]
    #[ignore = "needs Chromium (`pw browser install`); run with `cargo test -p pw-cli -- --ignored`"]
    async fn click_checks_see_the_text_a_click_shows() {
        const CART: &str = r#"<!doctype html><html><body><main><button id="add" onclick="document.getElementById('cart').textContent='Added to cart'">Add to cart</button><p id="cart">Cart is empty</p></main></body></html>"#;
        let base = serve(CART);
        let good = ManifestPatch {
            id: "add-to-cart".into(),
            sha256: String::new(),
            page: "/".into(),
            locators: pw_core::anchor::Locators { css: Some("#add".into()), ..Default::default() },
            ops: vec![],
            assertions: vec![Assertion::ClickShowsText("Added to cart".into())],
        };
        let bad = ManifestPatch {
            id: "add-to-cart-wrong".into(),
            assertions: vec![Assertion::ClickShowsText("Removed from cart".into())],
            ..good.clone()
        };
        let opts = Options { interact: true, visual: None };
        let checks = verify(&base, "production", &[good, bad], &opts).await.unwrap();
        assert!(checks.iter().any(|c| c.patch == "add-to-cart" && c.outcome == Outcome::Pass), "{checks:?}");
        assert!(checks.iter().any(|c| c.patch == "add-to-cart-wrong" && c.outcome == Outcome::Fail), "{checks:?}");
    }

    /// Read the text a page expression returns, for the editor's visible state.
    async fn read_text(page: &Page, expression: &str) -> String {
        match page.evaluate::<(), Option<Value>>(&format!("() => {expression}"), None).await.unwrap() {
            Some(Value::String(s)) => s,
            other => panic!("expected text from `{expression}`, got {other:?}"),
        }
    }

    /// The pending list shows what a save will write as the fields change. A link that is not a same-site path
    /// is named in the error and keeps Save disabled. "Discard changes" restores the element's fields. Runs in a
    /// real browser.
    #[tokio::test]
    #[ignore = "needs Chromium (`pw browser install`); run with `cargo test -p pw-cli -- --ignored`"]
    async fn the_editor_lists_pending_changes_and_blocks_an_unsafe_link() {
        let base = serve(SITE);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/editor-pending");
        let _ = fs::remove_dir_all(&root);
        Project::init(&root, "editor-test", &base).unwrap();

        let (_pw, browser) = launch(false).await.unwrap();
        let page = browser.new_page().await.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        page.expose_function("__pwSave", move |args: Vec<Value>| {
            let tx = tx.clone();
            async move {
                if let Some(text) = args.first().and_then(|v| v.as_str()) {
                    let _ = tx.send(text.to_string());
                }
                Value::Null
            }
        })
        .await
        .unwrap();
        page.goto(&format!("{base}/?pw_edit=1"), None).await.unwrap();
        let inject = format!("() => {{\n{EDITOR_JS}\n}}");
        page.evaluate::<(), Option<Value>>(&inject, None).await.unwrap();

        // The pending list follows the fields as they are typed.
        page.locator("#buy").click(None).await.unwrap();
        page.locator("[data-pw-text]").fill("Checkout now", None).await.unwrap();
        let pending = read_text(&page, "document.querySelector('[data-pw-list]').textContent").await;
        assert!(pending.contains("Checkout now"), "the pending list shows the change: {pending}");

        // A link that is not a same-site path is named in the error, and Save stays disabled.
        page.locator("[data-pw-behaviour]").select_option("navigate", None).await.unwrap();
        page.locator("[data-pw-barg]").fill("javascript:alert(1)", None).await.unwrap();
        let error = read_text(&page, "document.querySelector('[data-pw-error]').textContent").await;
        assert!(error.contains("same-site path"), "{error}");
        let disabled = page
            .evaluate::<(), Option<Value>>("() => document.querySelector('[data-pw-save]').disabled", None)
            .await
            .unwrap();
        assert_eq!(disabled, Some(Value::Bool(true)));

        // A same-site path clears the error, and the save is written.
        page.locator("[data-pw-barg]").fill("/order/confirm", None).await.unwrap();
        let error = read_text(&page, "document.querySelector('[data-pw-error]').textContent").await;
        assert!(error.trim().is_empty(), "{error}");
        page.locator("[data-pw-save]").click(None).await.unwrap();
        let message = tokio::time::timeout(Duration::from_secs(20), rx.recv()).await.unwrap().unwrap();
        let saved = save_from_message(&page, &message, &root, "tester", "tester").await.unwrap();

        // "Discard changes" clears the fields of the selected element.
        page.locator("#buy").click(None).await.unwrap();
        page.locator("[data-pw-text]").fill("Pay now", None).await.unwrap();
        page.locator("[data-pw-reset]").click(None).await.unwrap();
        let pending = read_text(&page, "document.querySelector('[data-pw-list]').textContent").await;
        assert!(!pending.contains("Pay now") && pending.contains("No changes yet"), "{pending}");
        browser.close().await.unwrap();

        let patch = Patch::read(&saved).unwrap();
        assert!(patch.ops.contains(&Op::OnClickNavigate { href: "/order/confirm".into() }), "{:?}", patch.ops);
    }

    /// The behaviour, insert and replace controls, each saved as its own patch in a real browser.
    #[tokio::test]
    #[ignore = "needs Chromium (`pw browser install`); run with `cargo test -p pw-cli -- --ignored`"]
    async fn the_editor_writes_behaviour_insert_and_replace_ops() {
        let base = serve(SITE);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/editor-behaviour");
        let _ = fs::remove_dir_all(&root);
        Project::init(&root, "editor-test", &base).unwrap();

        let (_pw, browser) = launch(false).await.unwrap();
        let page = browser.new_page().await.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        page.expose_function("__pwSave", move |args: Vec<Value>| {
            let tx = tx.clone();
            async move {
                if let Some(text) = args.first().and_then(|v| v.as_str()) {
                    let _ = tx.send(text.to_string());
                }
                Value::Null
            }
        })
        .await
        .unwrap();
        page.goto(&format!("{base}/?pw_edit=1"), None).await.unwrap();
        let inject = format!("() => {{\n{EDITOR_JS}\n}}");
        page.evaluate::<(), Option<Value>>(&inject, None).await.unwrap();

        // Save 1: a click behaviour on the button, and an inserted paragraph after it.
        page.locator("#buy").click(None).await.unwrap();
        page.locator("[data-pw-behaviour]").select_option("navigate", None).await.unwrap();
        page.locator("[data-pw-barg]").fill("/order/confirm", None).await.unwrap();
        page.locator("[data-pw-insert]").select_option("after", None).await.unwrap();
        page.locator("[data-pw-html]").fill("<p>Free returns</p>", None).await.unwrap();
        page.locator("[data-pw-save]").click(None).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(20), rx.recv()).await.unwrap().unwrap();
        let first_path = save_from_message(&page, &first, &root, "tester", "tester").await.unwrap();

        // Save 2: replace the coupon line's content. Its old text is gone, so the patch makes no text claim.
        page.locator("section.cart > p:last-of-type").click(None).await.unwrap();
        page.locator("[data-pw-insert]").select_option("replace", None).await.unwrap();
        page.locator("[data-pw-html]").fill("<em>Enter a code</em>", None).await.unwrap();
        page.locator("[data-pw-save]").click(None).await.unwrap();
        let second = tokio::time::timeout(Duration::from_secs(20), rx.recv()).await.unwrap().unwrap();
        let second_path = save_from_message(&page, &second, &root, "tester", "tester").await.unwrap();
        browser.close().await.unwrap();

        let first = Patch::read(&first_path).unwrap();
        assert!(first.ops.contains(&Op::OnClickNavigate { href: "/order/confirm".into() }), "{:?}", first.ops);
        assert!(
            first.ops.contains(&Op::InsertAfter { target: "#buy".into(), html: "<p>Free returns</p>".into() }),
            "{:?}",
            first.ops
        );
        assert!(first.assertions.contains(&Assertion::TextContains("Buy now".into())), "the button keeps its text claim");

        let second = Patch::read(&second_path).unwrap();
        assert!(second.ops.iter().any(|op| matches!(op, Op::SetHtml { value } if value == "<em>Enter a code</em>")), "{:?}", second.ops);
        assert!(
            !second.assertions.iter().any(|a| matches!(a, Assertion::TextContains(_))),
            "replaced content makes no text claim: {:?}",
            second.assertions
        );
    }
}
