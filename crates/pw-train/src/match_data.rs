//! Training pairs for the matcher: an anchor's description and a candidate's description, labelled 1 when the
//! candidate is the element the anchor was made from, and 0 otherwise.
//!
//! The candidates are the ones the heuristic already sees (`relocate::prefilter`), so the model learns the
//! choice it will be asked to make. When the target was deleted, every candidate is a negative, which teaches
//! the model to refuse look-alikes. The texts come from `pw_core::describe`, the same code inference uses.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use pw_core::anchor::Anchor;
use pw_core::describe;
use pw_core::relocate::{self, Config};
use pw_core::snapshot::Snapshot;
use serde::{Deserialize, Serialize};

use crate::synth::{self, Rng, SiteKind};

#[derive(Clone, Debug)]
pub struct Settings {
    pub sites: usize,
    pub variants: usize,
    pub seed: u64,
    /// How many prefilter candidates per variant become rows.
    pub candidates: usize,
    /// Every n-th site goes to the validation file.
    pub validation_every: usize,
}

/// One row of the JSONL file. `g` names the variant the candidate belongs to.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Row {
    pub g: String,
    pub a: String,
    pub c: String,
    pub y: u8,
}

#[derive(Debug, Default, Serialize)]
pub struct Stats {
    pub variants: usize,
    pub rows: usize,
    pub positive_rows: usize,
    /// Variants whose target was deleted, so every row is a negative.
    pub deleted_variants: usize,
    /// Variants whose target survived but was not among the candidates. No row teaches it.
    pub missed_by_candidates: usize,
}

fn gold_index(snap: &Snapshot) -> Option<u32> {
    snap.elements().iter().find(|e| e.attrs.contains_key("data-pwgold")).map(|e| e.index)
}

pub fn write(out: &Path, cfg: &Settings) -> Result<Stats> {
    fs::create_dir_all(out)?;
    let mut train = BufWriter::new(File::create(out.join("train.jsonl"))?);
    let mut val = BufWriter::new(File::create(out.join("val.jsonl"))?);
    let mut rng = Rng::new(cfg.seed);
    let mut stats = Stats::default();
    let relocate_cfg = Config::default();
    for i in 0..cfg.sites {
        let kind = SiteKind::ALL[i % SiteKind::ALL.len()];
        let base = synth::make_site(&mut rng, kind);
        let base_html = format!("<!doctype html>{}", base.render());
        // The marker finds the answer. The anchor and the candidates are built without it (see synth::strip_gold).
        let gold_v1 = gold_index(&Snapshot::parse(&base_html)).context("base page has no gold element")?;
        let v1 = Snapshot::parse(&synth::strip_gold(&base_html));
        let anchor = Anchor::from_element(v1.element(gold_v1).context("gold element outside the snapshot")?, &v1);
        let anchor_text = describe::anchor_text(&anchor);
        let sink: &mut BufWriter<File> = if cfg.validation_every > 0 && i % cfg.validation_every == 0 { &mut val } else { &mut train };
        for j in 0..cfg.variants {
            // Half the variants are the protocol's random mix, half are the compound drift.
            let variant = if rng.chance(0.5) { synth::variant(&mut rng, &base) } else { synth::compound_variant(&mut rng, &base) };
            let gold = gold_index(&Snapshot::parse(&variant.html));
            let snap = Snapshot::parse(&synth::strip_gold(&variant.html));
            let group = format!("{i}-{j}");
            stats.variants += 1;
            if gold.is_none() {
                stats.deleted_variants += 1;
            }
            let candidates = relocate::prefilter(&anchor, &snap, &relocate_cfg);
            if gold.is_some_and(|g| !candidates.iter().take(cfg.candidates).any(|(el, _)| el.index == g)) {
                stats.missed_by_candidates += 1;
            }
            for (el, _score) in candidates.iter().take(cfg.candidates) {
                let y = u8::from(gold == Some(el.index));
                let row = Row { g: group.clone(), a: anchor_text.clone(), c: describe::element_text(el), y };
                writeln!(sink, "{}", serde_json::to_string(&row)?)?;
                stats.rows += 1;
                stats.positive_rows += usize::from(y);
            }
        }
    }
    train.flush()?;
    val.flush()?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_pairs_and_a_positive_for_most_variants() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/match-data");
        let cfg = Settings { sites: 16, variants: 3, seed: 5001, candidates: 8, validation_every: 4 };
        let a = write(&dir, &cfg).unwrap();
        let first = fs::read_to_string(dir.join("train.jsonl")).unwrap();
        let b = write(&dir, &cfg).unwrap();
        assert_eq!(first, fs::read_to_string(dir.join("train.jsonl")).unwrap());
        assert_eq!(a.rows, b.rows);
        assert!(a.positive_rows * 2 > a.variants - a.deleted_variants, "most surviving targets are among the candidates: {a:?}");
        let rows: Vec<Row> = fs::read_to_string(dir.join("val.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert!(!rows.is_empty() && rows.iter().all(|r| r.y <= 1));
        let _ = fs::remove_dir_all(&dir);
    }
}
