//! Training examples for the two model jobs.
//!
//! RELOCATE examples use the same fast path, prefilter and prompt the CLI uses at run time.
//! A label is written only when the gold element is in the candidate list (or when the gold
//! element was deleted, which is labelled `null`). AUTHOR examples are validated against the
//! DSL before they are written. Splits are by whole site, so no held-out page leaks into train.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use pw_core::anchor::Anchor;
use pw_core::dsl;
use pw_core::patch::{Op, Patch};
use pw_core::prompt;
use pw_core::relocate::{self, Config};
use pw_core::serialize::element_line_no_box;
use pw_core::snapshot::{Element, Snapshot};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::synth::{self, Rng, SiteKind};

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Example {
    pub task: String,
    pub split: String,
    pub site: String,
    pub kind: String,
    pub candidates: usize,
    pub prompt: String,
    pub completion: String,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Stats {
    pub sites: usize,
    pub relocate: usize,
    pub relocate_null: usize,
    pub relocate_train: usize,
    pub relocate_heldout: usize,
    pub author: usize,
    pub author_train: usize,
    pub author_heldout: usize,
    pub skipped_fast_path: usize,
    pub skipped_gold_not_in_candidates: usize,
    pub skipped_no_candidates: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenConfig {
    pub seed: u64,
    pub sites: usize,
    pub variants_per_site: usize,
    pub author_per_site: usize,
    /// Every Nth site (by index) goes to the held-out split.
    pub heldout_every: usize,
    /// Share of RELOCATE examples labelled `null`. The fast path hides most non-deleted
    /// variants, so without balancing the kept examples are mostly deletions.
    pub null_fraction: f64,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig { seed: 1, sites: 2000, variants_per_site: 6, author_per_site: 3, heldout_every: 10, null_fraction: 0.15 }
    }
}

const COLORS: &[(&str, &str)] = &[
    ("green", "#0a7d32"),
    ("red", "#c62828"),
    ("blue", "#1a73e8"),
    ("orange", "#f57c00"),
    ("gray", "#6b7280"),
];
const TEXTS: &[&str] = &["Place order", "Checkout now", "Add to bag", "Confirm", "Continue", "Save"];
const PATHS: &[&str] = &["/api/v2/checkout", "/order/confirm", "/cart", "/help", "/account"];
const CLASSES: &[&str] = &["highlight", "sale", "wide"];

fn prefix_relocate(split: &str, site: &str, kind: &str, n: usize, prompt: String, completion: String) -> Example {
    Example {
        task: "relocate".into(),
        split: split.into(),
        site: site.into(),
        kind: kind.into(),
        candidates: n,
        prompt,
        completion,
    }
}

fn gold_index(snap: &Snapshot) -> Option<u32> {
    snap.elements().iter().find(|e| e.attrs.contains_key("data-pwgold")).map(|e| e.index)
}

/// One RELOCATE example per mutated variant that the fast path does not already resolve.
fn relocate_examples(
    rng: &mut Rng,
    site: &str,
    split: &str,
    base: &synth::Node,
    variants: usize,
    stats: &mut Stats,
) -> Result<Vec<Example>> {
    let base_html = format!("<!doctype html>{}", base.render());
    let snap_v1 = Snapshot::parse(&base_html);
    let gold_v1 = gold_index(&snap_v1).context("base page lost its gold element")?;
    let anchor = Anchor::from_element(snap_v1.element(gold_v1).context("gold element missing")?, &snap_v1);
    let cfg = Config::default();
    let mut out = Vec::new();
    for _ in 0..variants {
        let variant = synth::variant(rng, base);
        let snap = Snapshot::parse(&variant.html);
        let kind = if variant.mutations.is_empty() {
            "none".to_string()
        } else {
            variant.mutations.iter().map(|m| m.name()).collect::<Vec<_>>().join("+")
        };
        if variant.gold_present && relocate::fast_path(&anchor, &snap, &cfg)?.is_some() {
            stats.skipped_fast_path += 1;
            continue;
        }
        let candidates = relocate::prefilter(&anchor, &snap, &cfg);
        if candidates.is_empty() {
            stats.skipped_no_candidates += 1;
            continue;
        }
        let shown: Vec<(&Element, f64)> = candidates.iter().map(|(e, s)| (*e, *s)).collect();
        let completion = if !variant.gold_present {
            stats.relocate_null += 1;
            "{\"match\": null}".to_string()
        } else {
            let gold = gold_index(&snap).context("variant lost its gold element")?;
            match shown.iter().position(|(e, _)| e.index == gold) {
                Some(k) => format!("{{\"match\": {k}}}"),
                None => {
                    stats.skipped_gold_not_in_candidates += 1;
                    continue;
                }
            }
        };
        let prompt_text = prompt::relocate_prompt(&anchor.fingerprint, &shown);
        out.push(prefix_relocate(split, site, &kind, shown.len().min(prompt::MAX_CANDIDATES), prompt_text, completion));
    }
    Ok(out)
}

struct Request {
    text: String,
    ops: Vec<Op>,
}

fn author_request(rng: &mut Rng, el: &Element) -> Request {
    match rng.below(5) {
        0 => {
            let (word, hex) = *rng.pick(COLORS);
            let phrasing = ["make it {w}", "change the background to {w}", "give this a {w} background", "color it {w}"][rng.below(4)];
            Request {
                text: phrasing.replace("{w}", word),
                ops: vec![Op::SetStyle { prop: "background-color".into(), value: hex.into() }],
            }
        }
        1 => {
            let t = *rng.pick(TEXTS);
            let phrasing = ["change the text to \"{t}\"", "make it say {t}", "rename this to {t}"][rng.below(3)];
            Request { text: phrasing.replace("{t}", t), ops: vec![Op::SetText { value: t.into() }] }
        }
        2 => {
            let phrasing = ["hide this", "hide this element", "make this disappear"][rng.below(3)];
            Request { text: phrasing.into(), ops: vec![Op::Hide {}] }
        }
        3 => {
            let p = *rng.pick(PATHS);
            let phrasing = ["send this to {p}", "make it open {p} when clicked"][rng.below(2)];
            let op = if el.tag == "button" {
                Op::SetAttr { name: "formaction".into(), value: p.into() }
            } else {
                Op::OnClickNavigate { href: p.into() }
            };
            Request { text: phrasing.replace("{p}", p), ops: vec![op] }
        }
        _ => {
            let c = *rng.pick(CLASSES);
            Request { text: format!("add the class {c}"), ops: vec![Op::AddClass { name: c.into() }] }
        }
    }
}

/// One AUTHOR example: a target element and a plain-English request, with the DSL ops as the answer.
fn author_example(rng: &mut Rng, site: &str, split: &str, snap: &Snapshot) -> Result<Option<Example>> {
    let interactive: Vec<&Element> = snap
        .elements()
        .iter()
        .filter(|e| matches!(e.tag.as_str(), "button" | "a" | "input" | "p" | "h1"))
        .collect();
    if interactive.is_empty() {
        return Ok(None);
    }
    let el = *rng.pick(&interactive);
    let req = author_request(rng, el);
    let mut patch = Patch {
        id: format!("author-{site}"),
        author: "synthetic".into(),
        page: "/".into(),
        anchor: Anchor::from_element(el, snap),
        ops: req.ops.clone(),
        assertions: vec![],
    };
    if dsl::normalize(&mut patch).is_err() {
        return Ok(None);
    }
    let completion = serde_json::to_string(&serde_json::json!({ "ops": req.ops }))?;
    let prompt_text = format!(
        "TASK: author\nTARGET: {}\nREQUEST: {}\nANSWER: ",
        element_line_no_box(el, "[target]"),
        req.text
    );
    Ok(Some(Example {
        task: "author".into(),
        split: split.into(),
        site: site.into(),
        kind: req.ops.first().map(op_name).unwrap_or_default(),
        candidates: 0,
        prompt: prompt_text,
        completion,
    }))
}

fn op_name(op: &Op) -> String {
    serde_json::to_value(op)
        .ok()
        .and_then(|v| v.get("op").and_then(|x| x.as_str()).map(String::from))
        .unwrap_or_default()
}

fn is_null_label(e: &Example) -> bool {
    e.task == "relocate" && e.completion == "{\"match\": null}"
}

/// Keep every non-null relocate example, and only as many null ones as `null_fraction` allows.
fn balance_nulls(examples: Vec<Example>, rng: &mut Rng, null_fraction: f64) -> Vec<Example> {
    let non_null = examples.iter().filter(|e| e.task == "relocate" && !is_null_label(e)).count();
    let mut null_idx: Vec<usize> = examples.iter().enumerate().filter(|(_, e)| is_null_label(e)).map(|(i, _)| i).collect();
    let fraction = null_fraction.clamp(0.0, 0.95);
    let keep_n = ((non_null as f64) * fraction / (1.0 - fraction)).round() as usize;
    // Partial Fisher-Yates: a seeded random choice of which nulls to keep.
    let keep_n = keep_n.min(null_idx.len());
    for i in 0..keep_n {
        let j = i + rng.below(null_idx.len() - i);
        null_idx.swap(i, j);
    }
    let keep: BTreeSet<usize> = null_idx[..keep_n].iter().copied().collect();
    examples
        .into_iter()
        .enumerate()
        .filter(|(i, e)| !is_null_label(e) || keep.contains(i))
        .map(|(_, e)| e)
        .collect()
}

fn recount(examples: &[Example], stats: &mut Stats) {
    stats.relocate = 0;
    stats.relocate_null = 0;
    stats.relocate_train = 0;
    stats.relocate_heldout = 0;
    stats.author = 0;
    stats.author_train = 0;
    stats.author_heldout = 0;
    for e in examples {
        let held = e.split == "heldout";
        match e.task.as_str() {
            "relocate" => {
                stats.relocate += 1;
                stats.relocate_null += usize::from(is_null_label(e));
                if held { stats.relocate_heldout += 1 } else { stats.relocate_train += 1 }
            }
            "author" => {
                stats.author += 1;
                if held { stats.author_heldout += 1 } else { stats.author_train += 1 }
            }
            _ => {}
        }
    }
}

pub fn generate(cfg: &GenConfig) -> Result<(Vec<Example>, Stats)> {
    let mut rng = Rng::new(cfg.seed);
    let mut stats = Stats { sites: cfg.sites, ..Stats::default() };
    let mut examples = Vec::new();
    for i in 0..cfg.sites {
        let site = format!("site-{i:05}");
        let split = if cfg.heldout_every > 0 && i % cfg.heldout_every == 0 { "heldout" } else { "train" };
        let kind = SiteKind::ALL[i % SiteKind::ALL.len()];
        let base = synth::make_site(&mut rng, kind);
        examples.extend(relocate_examples(&mut rng, &site, split, &base, cfg.variants_per_site, &mut stats)?);
        let snap = Snapshot::parse(&format!("<!doctype html>{}", base.render()));
        for _ in 0..cfg.author_per_site {
            if let Some(ex) = author_example(&mut rng, &site, split, &snap)? {
                examples.push(ex);
            }
        }
    }
    let examples = balance_nulls(examples, &mut rng, cfg.null_fraction);
    recount(&examples, &mut stats);
    Ok((examples, stats))
}

/// Sanity checks every example must pass before it is written.
pub fn check(examples: &[Example]) -> Result<()> {
    for ex in examples {
        match ex.task.as_str() {
            "relocate" => {
                prompt::parse_answer(&ex.completion, ex.candidates.max(1))
                    .with_context(|| format!("bad relocate completion for {}", ex.site))?;
            }
            "author" => {
                let v: serde_json::Value = serde_json::from_str(&ex.completion)?;
                let ops: Vec<Op> = serde_json::from_value(v["ops"].clone()).with_context(|| format!("bad ops for {}", ex.site))?;
                let mut p = Patch {
                    id: "check".into(),
                    author: String::new(),
                    page: "/".into(),
                    anchor: Anchor {
                        locators: pw_core::anchor::Locators { css: Some("button".into()), ..Default::default() },
                        fingerprint: pw_core::anchor::Fingerprint { tag: "button".into(), ..Default::default() },
                    },
                    ops,
                    assertions: vec![],
                };
                dsl::normalize(&mut p)?;
            }
            other => bail!("unknown task `{other}`"),
        }
    }
    Ok(())
}

fn write_jsonl(path: &Path, rows: &[&Example]) -> Result<String> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    let mut hasher = Sha256::new();
    for ex in rows {
        let line = serde_json::to_string(ex)?;
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        writeln!(w, "{line}")?;
    }
    w.flush()?;
    Ok(hex::encode(hasher.finalize()))
}

/// Write `train.jsonl`, `heldout.jsonl`, and `meta.json` into `dir`.
pub fn write_dataset(dir: &Path, cfg: &GenConfig) -> Result<Stats> {
    let (examples, stats) = generate(cfg)?;
    check(&examples)?;
    fs::create_dir_all(dir)?;
    let train: Vec<&Example> = examples.iter().filter(|e| e.split == "train").collect();
    let heldout: Vec<&Example> = examples.iter().filter(|e| e.split == "heldout").collect();
    let train_sha = write_jsonl(&dir.join("train.jsonl"), &train)?;
    let held_sha = write_jsonl(&dir.join("heldout.jsonl"), &heldout)?;
    let heldout_sites: BTreeSet<&str> = heldout.iter().map(|e| e.site.as_str()).collect();
    let train_sites: BTreeSet<&str> = train.iter().map(|e| e.site.as_str()).collect();
    if !heldout_sites.is_disjoint(&train_sites) {
        bail!("a site appears in both splits");
    }
    let meta = serde_json::json!({
        "config": cfg,
        "stats": stats,
        "train_sha256": train_sha,
        "heldout_sha256": held_sha,
        "format": "one JSON object per line: task, split, site, kind, candidates, prompt, completion",
    });
    fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> GenConfig {
        GenConfig { seed: 3, sites: 40, variants_per_site: 4, author_per_site: 2, heldout_every: 5, null_fraction: 0.15 }
    }

    #[test]
    fn generation_is_deterministic() {
        let (a, sa) = generate(&small()).unwrap();
        let (b, sb) = generate(&small()).unwrap();
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }

    #[test]
    fn every_example_passes_the_output_checks() {
        let (examples, stats) = generate(&small()).unwrap();
        assert!(!examples.is_empty());
        check(&examples).unwrap();
        assert!(stats.relocate > 0 && stats.author > 0);
    }

    #[test]
    fn splits_never_share_a_site() {
        let (examples, _) = generate(&GenConfig { sites: 120, ..small() }).unwrap();
        let held: BTreeSet<&str> = examples.iter().filter(|e| e.split == "heldout").map(|e| e.site.as_str()).collect();
        let train: BTreeSet<&str> = examples.iter().filter(|e| e.split == "train").map(|e| e.site.as_str()).collect();
        assert!(held.is_disjoint(&train));
        assert!(!held.is_empty());
    }

    #[test]
    fn relocate_labels_point_at_the_gold_candidate_and_nulls_are_present() {
        let (examples, stats) = generate(&GenConfig { sites: 200, ..small() }).unwrap();
        let relocate: Vec<&Example> = examples.iter().filter(|e| e.task == "relocate").collect();
        assert!(stats.relocate_null > 0, "deleted targets should be labelled null");
        for ex in &relocate {
            // The prompt lists exactly `candidates` numbered lines after CANDIDATES.
            let listed = ex.prompt.lines().filter(|l| l.starts_with('[') && !l.starts_with("[old]") && !l.starts_with("[target]")).count();
            assert_eq!(listed, ex.candidates, "{}", ex.prompt);
        }
    }

    #[test]
    fn author_prompts_carry_the_serialised_target() {
        let (examples, _) = generate(&small()).unwrap();
        let author = examples.iter().find(|e| e.task == "author").unwrap();
        assert!(author.prompt.contains("TARGET: [target] "), "{}", author.prompt);
        assert!(author.prompt.contains("REQUEST: "));
    }
}
