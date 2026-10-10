//! Held-out benchmark for re-anchoring. It uses held-out synthetic sites (never seen in
//! training) and reports what happens to every patch after a site update:
//!   - the heuristic re-anchoring used by `pw check` (the fallback that must work without a model)
//!   - a selector-only baseline (CSS/XPath/testid/role locators, no fingerprint)
//!   - optionally the trained model, alone and in front of the heuristic
//! A wrong match is the number that matters most: it means a patch silently moved to the
//! wrong element. Abstaining (ambiguous or broken) is the safe failure.
//!
//! [`run_sweep`] asks the model once per case and replays its answer at several confidence bars,
//! so a bar can be chosen without running the model again for each one.

use std::cell::RefCell;

use pw_core::anchor::{Anchor, Fingerprint};
use pw_core::error::Result as PwResult;
use pw_core::relocate::{self, Choice, Chooser, Config, Heuristic, Outcome, Status};
use pw_core::snapshot::{Element, Snapshot};
use pw_model::{alone, decide, Judgement, ModelChooser, ModelFirst};
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
    /// The trained model, when the benchmark was run with one. Same held-out variants as the rows above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRows>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct ModelRows {
    /// The model decides alone: its answer is applied when confident, and otherwise the element is left alone.
    pub only_present: Tally,
    pub only_deleted: Tally,
    /// The model first, the heuristic when the model is unsure (what `pw --model` does).
    pub first_present: Tally,
    pub first_deleted: Tally,
}

/// The model rows at one confidence bar (`model_accept`).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Bar {
    pub accept: f64,
    pub model: ModelRows,
}

/// The benchmark at several confidence bars. `report` holds the heuristic and selector rows.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Sweep {
    pub report: Report,
    pub bars: Vec<Bar>,
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

/// One held-out case: a site's anchor, and one mutated variant of that site.
struct Case {
    anchor: Anchor,
    snap: Snapshot,
    /// The element that should be matched. `None` when the target was deleted.
    gold: Option<u32>,
    stored_text: String,
}

/// Every case for a seed, in a fixed order. The same seed always gives the same cases.
fn cases(sites: usize, variants: usize, seed: u64) -> anyhow::Result<Vec<Case>> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(sites * variants);
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
        // The identity check is what a patch-level `textContains` assertion would test.
        let stored_text = anchor.fingerprint.text.clone().unwrap_or_default();
        for _ in 0..variants {
            let variant = synth::variant(&mut rng, &base);
            let snap = Snapshot::parse(&variant.html);
            let gold = snap.elements().iter().find(|e| e.attrs.contains_key("data-pwgold")).map(|e| e.index);
            out.push(Case { anchor: anchor.clone(), snap, gold, stored_text: stored_text.clone() });
        }
    }
    Ok(out)
}

/// Would an identity assertion on the stored text fail this node? Only wrong matches are asked.
fn caught_by_identity(case: &Case, node: Option<u32>) -> bool {
    node.and_then(|n| case.snap.element(n))
        .is_some_and(|el| !case.stored_text.is_empty() && !el.text.contains(&case.stored_text))
}

/// The heuristic and selector-only rows for one case.
fn score_base(report: &mut Report, case: &Case, heuristic: &Outcome) {
    let selector = selector_only(&case.anchor, &case.snap);
    let chosen = heuristic.element;
    if case.gold.is_some() {
        record(&mut report.present, case.gold, chosen, heuristic.status, caught_by_identity(case, chosen));
        record(&mut report.selector_only_present, case.gold, selector, Status::Broken, caught_by_identity(case, selector));
    } else {
        record(&mut report.deleted, None, chosen, heuristic.status, caught_by_identity(case, chosen));
        record(&mut report.selector_only_deleted, None, selector, Status::Broken, caught_by_identity(case, selector));
    }
}

/// The model-alone and model-first rows for one case.
fn score_model(rows: &mut ModelRows, case: &Case, only: &Outcome, first: &Outcome) {
    let (only_t, first_t) = if case.gold.is_some() {
        (&mut rows.only_present, &mut rows.first_present)
    } else {
        (&mut rows.only_deleted, &mut rows.first_deleted)
    };
    record(only_t, case.gold, only.element, only.status, caught_by_identity(case, only.element));
    record(first_t, case.gold, first.element, first.status, caught_by_identity(case, first.element));
}

/// Asks the model and the heuristic about one case, and keeps both answers without applying either.
struct Recorder<'a> {
    model: &'a ModelChooser,
    heuristic: &'a Heuristic,
    seen: RefCell<Option<Seen>>,
}

struct Seen {
    judged: Judgement,
    heuristic: Choice,
}

impl Chooser for Recorder<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> PwResult<Choice> {
        let judged = self.model.judge(fp, candidates)?;
        let heuristic = self.heuristic.choose(fp, candidates)?;
        *self.seen.borrow_mut() = Some(Seen { judged, heuristic });
        Ok(Choice::Unsure)
    }
}

/// Returns one decision whatever the candidates are, so the same routing code can be scored at many bars.
struct Replay(Choice);

impl Chooser for Replay {
    fn choose(&self, _: &Fingerprint, _: &[(&Element, f64)]) -> PwResult<Choice> {
        Ok(self.0)
    }
}

/// Run the benchmark over `sites` held-out sites with `variants` mutated versions each.
pub fn run(sites: usize, variants: usize, seed: u64) -> anyhow::Result<Report> {
    run_with_model(sites, variants, seed, None)
}

/// As [`run`], and also the trained model when one is given. The model sees exactly the variants the
/// heuristic sees, so the rows are directly comparable.
pub fn run_with_model(sites: usize, variants: usize, seed: u64, model: Option<&ModelChooser>) -> anyhow::Result<Report> {
    let cfg = Config::default();
    let heuristic = Heuristic { cfg };
    let mut report = Report { sites, variants, ..Report::default() };
    if model.is_some() {
        report.model = Some(ModelRows::default());
    }
    for case in cases(sites, variants, seed)? {
        let outcome = relocate::relocate(&case.anchor, &case.snap, &heuristic, &cfg)?;
        score_base(&mut report, &case, &outcome);
        if let (Some(model), Some(rows)) = (model, report.model.as_mut()) {
            let only = relocate::relocate(&case.anchor, &case.snap, model, &cfg)?;
            let first = relocate::relocate(&case.anchor, &case.snap, &ModelFirst { model, fallback: &heuristic }, &cfg)?;
            score_model(rows, &case, &only, &first);
        }
    }
    Ok(report)
}

/// As [`run_with_model`], at several confidence bars. The model is asked once per case. Its answer and the
/// heuristic's are then replayed at each bar through the same routing as `ModelChooser` and `ModelFirst`,
/// so the rows at a bar equal what `run_with_model` gives with `model_accept` set to that bar.
pub fn run_sweep(sites: usize, variants: usize, seed: u64, model: &ModelChooser, accepts: &[f64]) -> anyhow::Result<Sweep> {
    let cfg = Config::default();
    let heuristic = Heuristic { cfg };
    let mut report = Report { sites, variants, ..Report::default() };
    let mut bars: Vec<Bar> = accepts.iter().map(|&accept| Bar { accept, model: ModelRows::default() }).collect();
    for case in cases(sites, variants, seed)? {
        let outcome = relocate::relocate(&case.anchor, &case.snap, &heuristic, &cfg)?;
        score_base(&mut report, &case, &outcome);
        if bars.is_empty() {
            continue;
        }
        let recorder = Recorder { model, heuristic: &heuristic, seen: RefCell::new(None) };
        relocate::relocate(&case.anchor, &case.snap, &recorder, &cfg)?;
        let seen = recorder.seen.into_inner();
        for bar in bars.iter_mut() {
            // When the chooser was not asked (the fast path found the element, or there were no candidates),
            // the outcome does not depend on it, so any decision gives the same result.
            let (only_choice, first_choice) = match &seen {
                Some(s) => (alone(&s.judged, bar.accept), decide(&s.judged, bar.accept, || Ok(s.heuristic))?),
                None => (Choice::Unsure, Choice::Unsure),
            };
            let only = relocate::relocate(&case.anchor, &case.snap, &Replay(only_choice), &cfg)?;
            let first = relocate::relocate(&case.anchor, &case.snap, &Replay(first_choice), &cfg)?;
            score_model(&mut bar.model, &case, &only, &first);
        }
    }
    Ok(Sweep { report, bars })
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

    #[test]
    fn a_seed_always_gives_the_same_cases() {
        let a = cases(12, 2, 9).unwrap();
        let b = cases(12, 2, 9).unwrap();
        assert_eq!(a.len(), 24);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.gold, y.gold);
            assert_eq!(x.snap.elements().len(), y.snap.elements().len());
        }
    }
}
