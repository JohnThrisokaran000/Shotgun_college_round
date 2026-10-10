//! Playwright management for the CLI: install browsers, capture snapshots, open the
//! visual editor, and check assertions in a real browser.
//!
//! The browser engine is Microsoft's Playwright, driven from Rust through the
//! `playwright-rs` bindings. The small JavaScript snippets below run inside the page
//! because the DOM can only be read from there. They do no decisions of their own:
//! the Rust side reads the results and decides.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use playwright_rs::{LaunchOptions, Page, Playwright};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use pw_core::anchor::Anchor;
use pw_core::dsl;
use pw_core::patch::{Assertion, Op, Patch};
use pw_core::release::ManifestPatch;
use pw_core::snapshot::Snapshot;

use crate::project::{self, Project};

/// Returns the page's HTML and each element's rectangle, in document order.
const CAPTURE_JS: &str = r#"() => {
  const els = Array.from(document.querySelectorAll('*'));
  const rects = els.map(e => {
    const r = e.getBoundingClientRect();
    return (r.width > 0 || r.height > 0) ? [r.x, r.y, r.width, r.height] : null;
  });
  return { html: '<!doctype html>' + document.documentElement.outerHTML, rects: rects, count: els.length };
}"#;

/// Checks one patch's assertions on the page. Returns [label, passed] pairs.
/// `clickNavigatesTo` needs a click, so it is reported as not checked (null).
const ASSERT_JS: &str = r#"(p) => {
  const L = p.locators || {};
  let el = null;
  try { if (L.css) { const all = document.querySelectorAll(L.css); if (all.length === 1) el = all[0]; } } catch (e) { el = null; }
  if (!el && L.xpath) { try { el = document.evaluate(L.xpath, document, null, 9, null).singleNodeValue; } catch (e) { el = null; } }
  if (!el && L.testid) el = document.querySelector('[data-testid="' + L.testid.replace(/["\\]/g, '\\$&') + '"]');
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
    }
  }
  return out;
}"#;

const RUNTIME_READY_JS: &str = "() => !!(window.__pw && window.__pw.state && window.__pw.state.release)";

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

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
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

async fn launch(headed: bool) -> Result<(Playwright, playwright_rs::Browser)> {
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

async fn save_from_message(page: &Page, text: &str, root: &Path, author: &str) -> Result<PathBuf> {
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
    let mut patch = Patch {
        id: format!("edit-{}", now_secs()),
        author: author.into(),
        page: req.page.split('?').next().unwrap_or("/").to_string(),
        anchor: Anchor::from_element(el, &snap),
        ops,
        assertions: vec![Assertion::Visible(true)],
    };
    dsl::normalize(&mut patch)?;
    if !req.note.is_empty() {
        println!("note: {}", req.note);
    }
    let project = Project::open(root)?;
    let path = project.save_patch(&patch)?;
    project
        .audit()
        .append(author, "patch.add", json!({"id": patch.id, "page": patch.page, "note": req.note}))?;
    Ok(path)
}

/// Open the editor in a visible browser. Each Save becomes a patch file.
pub async fn edit(url: &str, root: &Path, author: &str) -> Result<()> {
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
    println!("Editing {target}. Click an element, change it, then Save. Ctrl+C to quit.");
    loop {
        tokio::select! {
            message = rx.recv() => {
                let Some(text) = message else { break };
                match save_from_message(&page, &text, root, author).await {
                    Ok(path) => println!("saved {}", path.display()),
                    Err(e) => eprintln!("save failed: {e:#}"),
                }
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    browser.close().await?;
    Ok(())
}

#[derive(Debug)]
pub struct AssertResult {
    pub patch: String,
    pub label: String,
    pub passed: Option<bool>,
}

/// Run a channel's manifest assertions in a real browser.
/// The channel is selected through the proxy's /__pw/channel route, so the page loads with
/// that release applied.
pub async fn verify(base: &str, channel: &str, patches: &[ManifestPatch]) -> Result<Vec<AssertResult>> {
    let base = base.trim_end_matches('/');
    let (_pw, browser) = launch(false).await?;
    let page = browser.new_page().await?;
    let mut results = Vec::new();
    let mut pages: Vec<&str> = patches.iter().map(|p| p.page.as_str()).collect();
    pages.sort_unstable();
    pages.dedup();
    for path in pages {
        page.goto(&format!("{base}/__pw/channel?name={channel}"), None).await?;
        page.goto(&format!("{base}{path}"), None).await?;
        for _ in 0..50 {
            let ready: bool = page.evaluate::<(), bool>(RUNTIME_READY_JS, None).await.unwrap_or(false);
            if ready {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        for patch in patches.iter().filter(|p| p.page == path) {
            let arg = json!({"locators": patch.locators, "assert": patch.assertions});
            let rows: Vec<(String, Option<bool>)> = page.evaluate(ASSERT_JS, Some(&arg)).await?;
            if rows.is_empty() {
                results.push(AssertResult { patch: patch.id.clone(), label: "(no assertions)".into(), passed: None });
            }
            for (label, passed) in rows {
                results.push(AssertResult { patch: patch.id.clone(), label, passed });
            }
        }
    }
    browser.close().await?;
    Ok(results)
}
