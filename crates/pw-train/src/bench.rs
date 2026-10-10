//! Held-out benchmark for re-anchoring. It uses held-out synthetic sites (never seen in
//! training) and reports what happens to every patch after a site update:
//!   - the heuristic re-anchoring used by `pw check` (the fallback that must work without a model)
//!   - a selector-only baseline (CSS/XPath/testid/role locators, no fingerprint)
//! A wrong match is the number that matters most: it means a patch silently moved to the
//! wrong element. Abstaining (ambiguous or broken) is the safe failure.

use pw_core::anchor::Anchor;
use pw_core::relocate::{self, Config, Heuristic, Status};
use pw_core::snapshot::Snapshot;
use serde::Serialize;

use crate::synth::{self, Rng, SiteKind};

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Tally {
    pub cases: usize,
    pub correct: usize,
    pub wrong_match: usize,
    pub ambiguous: usize,
    pub broken: usize,
    pub unlocated: usize,
    /// Wrong matches whose text no longer contains the stored text. An identity assertion on
    /// that text would fail them in preview, before release.
    pub wrong_caught_by_identity: usize,
}

impl Tally {
    fn rate(n: usize, d: usize) -> f64 {
        if d == 0 {
            0.0
        } else {
            n as f64 / d as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        Self::rate(self.correct, self.cases)
    }

    pub fn wrong_rate(&self) -> f64 {
        Self::rate(self.wrong_match, self.cases)
    }

    pub fn abstain_rate(&self) -> f64 {
        Self::rate(self.ambiguous + self.broken, self.cases)
    }
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Report {
    pub sites: usize,
    pub variants: usize,
    /// Variants where the target is still on the page.
    pub present: Tally,
    /// Variants where the target was deleted. Any match here is wrong.
    pub deleted: Tally,
    pub selector_only_present: Tally,
    pub selector_only_deleted: Tally,
}

/// What a selector-only re-anchor would do: use the locators alone, no fingerprint.
fn selector_only(anchor: &Anchor, snap: &Snapshot) -> Option<u32> {
    let mut found: Vec<u32> = Vec::new();
    if let Some(css) = &anchor.locators.css {
        if let Ok(hits) = snap.css(css) {
            if let [only] = hits[..] {
                found.push(only);
            }
        }
    }
    if let Some(xpath) = &anchor.locators.xpath {
        if let [only] = snap.by_xpath(xpath)[..] {
            found.push(only);
        }
    }
    if let Some(testid) = &anchor.locators.testid {
        if let [only] = snap.by_testid(testid)[..] {
            found.push(only);
        }
    }
    if let Some(role) = &anchor.locators.role {
        if let [only] = snap.by_role_name(&role.role, &role.name)[..] {
            found.push(only);
        }
    }
    // The first locator that resolves uniquely wins, as a plain selector script would do.
    found.first().copied()
}

fn record(tally: &mut Tally, gold: Option<u32>, chosen: Option<u32>, abstained: Status, caught: bool) {
    tally.cases += 1;
    match (gold, chosen) {
        (Some(g), Some(c)) if g == c => tally.correct += 1,
        (_, Some(_)) => {
            tally.wrong_match += 1;
            tally.wrong_caught_by_identity += usize::from(caught);
        }
        (_, None) => match abstained {
            Status::Ambiguous => tally.ambiguous += 1,
            Status::Broken => tally.broken += 1,
            _ => tally.unlocated += 1,
        },
    }
}

/// Run the benchmark over `sites` held-out sites with `variants` mutated versions each.
pub fn run(sites: usize, variants: usize, seed: u64) -> anyhow::Result<Report> {
    let mut rng = Rng::new(seed);
    let cfg = Config::default();
    let heuristic = Heuristic { cfg };
    let mut report = Report { sites, variants, ..Report::default() };
    for i in 0..sites {
        let kind = SiteKind::ALL[i % SiteKind::ALL.len()];
        let base = synth::make_site(&mut rng, kind);
        let snap_v1 = Snapshot::parse(&format!("<!doctype html>{}", base.render()));
        let gold_v1 = snap_v1
            .elements()
            .iter()
            .find(|e| e.attrs.contains_key("data-pwgold"))
            .map(|e| e.index)
            .ok_or_else(|| anyhow::anyhow!("base page has no gold element"))?;
        let anchor = Anchor::from_element(snap_v1.element(gold_v1).unwrap(), &snap_v1);
        for _ in 0..variants {
            let variant = synth::variant(&mut rng, &base);
            let snap = Snapshot::parse(&variant.html);
            let gold = snap.elements().iter().find(|e| e.attrs.contains_key("data-pwgold")).map(|e| e.index);
            let outcome = relocate::relocate(&anchor, &snap, &heuristic, &cfg)?;
            let chosen = outcome.element;
            let selector = selector_only(&anchor, &snap);
            // The identity check is what a patch-level `textContains` assertion would test.
            let stored_text = anchor.fingerprint.text.clone().unwrap_or_default();
            let caught_by = |node: Option<u32>| {
                node.and_then(|n| snap.element(n))
                    .is_some_and(|el| !stored_text.is_empty() && !el.text.contains(&stored_text))
            };
            if gold.is_some() {
                record(&mut report.present, gold, chosen, outcome.status, caught_by(chosen));
                record(&mut report.selector_only_present, gold, selector, Status::Broken, caught_by(selector));
            } else {
                record(&mut report.deleted, None, chosen, outcome.status, caught_by(chosen));
                record(&mut report.selector_only_deleted, None, selector, Status::Broken, caught_by(selector));
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tallies_partition_every_case() {
        let report = run(40, 3, 5).unwrap();
        let t = &report.present;
        assert_eq!(t.correct + t.wrong_match + t.ambiguous + t.broken + t.unlocated, t.cases);
        assert_eq!(report.present.cases + report.deleted.cases, 40 * 3);
    }

    #[test]
    fn deleted_target_matches_are_counted_as_wrong_not_correct() {
        // A deleted target has no correct answer, so every match is a wrong one.
        let report = run(60, 4, 11).unwrap();
        assert!(report.deleted.cases > 0);
        assert_eq!(report.deleted.correct, 0);
        assert_eq!(report.deleted.wrong_match + report.deleted.ambiguous + report.deleted.broken + report.deleted.unlocated, report.deleted.cases);
    }

    #[test]
    fn benchmark_is_deterministic() {
        assert_eq!(run(20, 2, 3).unwrap(), run(20, 2, 3).unwrap());
    }
}
