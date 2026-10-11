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
/// How each target's variants are made. `Mixed` is the protocol's default (1 to 3 random mutations). `Compound`
/// is the drift a site change can cause to one target at once: id dropped, text changed, element moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Drift {
    Mixed,
    Compound,
}

fn cases(sites: usize, variants: usize, seed: u64, drift: Drift) -> anyhow::Result<Vec<Case>> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(sites * variants);
    for i in 0..sites {
        let kind = SiteKind::ALL[i % SiteKind::ALL.len()];
        let base = synth::make_site(&mut rng, kind);
        let base_html = format!("<!doctype html>{}", base.render());
        // The marker finds the answer. Relocation never sees it: the anchor would keep it and match it for free.
        let gold_v1 = Snapshot::parse(&base_html)
            .elements()
            .iter()
            .find(|e| e.attrs.contains_key("data-pwgold"))
            .map(|e| e.index)
            .ok_or_else(|| anyhow::anyhow!("base page has no gold element"))?;
        let snap_v1 = Snapshot::parse(&synth::strip_gold(&base_html));
        let anchor = Anchor::from_element(snap_v1.element(gold_v1).unwrap(), &snap_v1);
        // The identity check is what a patch-level `textContains` assertion would test.
        let stored_text = anchor.fingerprint.text.clone().unwrap_or_default();
        for _ in 0..variants {
            let variant = match drift {
                Drift::Mixed => synth::variant(&mut rng, &base),
                Drift::Compound => synth::compound_variant(&mut rng, &base),
            };
            let gold = Snapshot::parse(&variant.html)
                .elements()
                .iter()
                .find(|e| e.attrs.contains_key("data-pwgold"))
                .map(|e| e.index);
            let snap = Snapshot::parse(&synth::strip_gold(&variant.html));
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
    run_configured(sites, variants, seed, model, Config::default())
}

/// As [`run_with_model`], with the relocation settings given. The tuning scripts use this to change one setting at a time.
pub fn run_configured(sites: usize, variants: usize, seed: u64, model: Option<&ModelChooser>, cfg: Config) -> anyhow::Result<Report> {
    run_drift(sites, variants, seed, model, cfg, Drift::Mixed)
}

/// As [`run_configured`], with the kind of drift given.
pub fn run_drift(sites: usize, variants: usize, seed: u64, model: Option<&ModelChooser>, cfg: Config, drift: Drift) -> anyhow::Result<Report> {
    let heuristic = Heuristic { cfg };
    let mut report = Report { sites, variants, ..Report::default() };
    if model.is_some() {
        report.model = Some(ModelRows::default());
    }
    for case in cases(sites, variants, seed, drift)? {
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

/// The protocol's objective: correct present matches, minus five times the wrong ones (present and deleted).
pub fn objective(r: &Report) -> i64 {
    r.present.correct as i64 - 5 * (r.present.wrong_match as i64 + r.deleted.wrong_match as i64)
}

/// The matcher's thresholds (accept, margin, review) tried on the development split. Fixed in code before any
/// test seed was run, so the choice cannot be tuned against the test seeds (docs/DESIGN.md, section 6.7).
pub fn matcher_grid() -> Vec<(f64, f64, f64)> {
    let mut grid = Vec::new();
    for accept in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        for margin in [0.0, 0.1, 0.2, 0.3, 0.4] {
            for review in [0.1, 0.2, 0.3, 0.4] {
                grid.push((accept, margin, review));
            }
        }
    }
    grid
}

/// The index of the chosen setting: the highest objective, then the most correct present matches, then the least
/// work for a person (ambiguous results). Ties after that go to the earlier setting in the grid.
pub fn select_setting(runs: &[(Config, Report)]) -> Option<usize> {
    runs.iter()
        .enumerate()
        .max_by_key(|(i, (_, r))| {
            let human = r.present.ambiguous as i64 + r.deleted.ambiguous as i64;
            (objective(r), r.present.correct as i64, -human, -(*i as i64))
        })
        .map(|(i, _)| i)
}

/// Records the matcher's score for every candidate of one case, and passes the choice through unchanged.
struct ScoreRecorder<'a> {
    matcher: &'a pw_model::Matcher,
    cfg: Config,
    seen: RefCell<Option<Vec<f64>>>,
}

impl Chooser for ScoreRecorder<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> PwResult<Choice> {
        let anchor = pw_core::describe::fingerprint_text(fp);
        let mut scores = Vec::with_capacity(candidates.len());
        for (el, _) in candidates {
            scores.push(self.matcher.score(&anchor, &pw_core::describe::element_text(el))?);
        }
        let choice = relocate::matcher_choice(&scores, &self.cfg);
        *self.seen.borrow_mut() = Some(scores);
        Ok(choice)
    }
}

/// Applies recorded matcher scores under one setting of the thresholds.
struct ScoreReplay {
    scores: Option<Vec<f64>>,
    cfg: Config,
}

impl Chooser for ScoreReplay {
    fn choose(&self, _: &Fingerprint, _: &[(&Element, f64)]) -> PwResult<Choice> {
        Ok(match &self.scores {
            Some(scores) => relocate::matcher_choice(scores, &self.cfg),
            None => Choice::Null,
        })
    }
}

/// The matcher under several threshold settings, on the same cases. Each candidate is scored once, and every
/// setting is then replayed on those scores. The settings only change the decision, never the candidates.
pub fn run_matcher_grid(
    sites: usize,
    variants: usize,
    seed: u64,
    drift: Drift,
    matcher: &pw_model::Matcher,
    base: Config,
    grid: &[(f64, f64, f64)],
) -> anyhow::Result<Vec<(Config, Report)>> {
    let all = cases(sites, variants, seed, drift)?;
    let recorder = ScoreRecorder { matcher, cfg: base, seen: RefCell::new(None) };
    let mut scored: Vec<Option<Vec<f64>>> = Vec::with_capacity(all.len());
    for case in &all {
        relocate::relocate(&case.anchor, &case.snap, &recorder, &base)?;
        scored.push(recorder.seen.borrow_mut().take());
    }
    let mut out = Vec::with_capacity(grid.len());
    for &(accept, margin, review) in grid {
        let cfg = Config { matcher_accept: accept, matcher_margin: margin, matcher_review: review, ..base };
        let mut report = Report { sites, variants, ..Report::default() };
        for (case, scores) in all.iter().zip(&scored) {
            let replay = ScoreReplay { scores: scores.clone(), cfg };
            let outcome = relocate::relocate(&case.anchor, &case.snap, &replay, &cfg)?;
            score_base(&mut report, case, &outcome);
        }
        out.push((cfg, report));
    }
    Ok(out)
}

/// The combined tracker's settings: every matcher threshold setting of `matcher_grid`, with each veto of
/// 0 (off), 0.1, 0.2, 0.3, 0.4 and 0.5. Fixed in code, as section 6.10 of docs/DESIGN.md requires.
pub fn combined_grid() -> Vec<(f64, f64, f64, f64)> {
    let mut grid = Vec::new();
    for (accept, margin, review) in matcher_grid() {
        for veto in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            grid.push((accept, margin, review, veto));
        }
    }
    grid
}

/// What one case offers the combined tracker: the heuristic's choice, and for each candidate (in the order the
/// candidates were given) its element index and the matcher's score.
#[derive(Clone)]
struct CaseOffer {
    heuristic: Choice,
    elements: Vec<u32>,
    scores: Vec<f64>,
}

/// Pass one over the cases: asks the heuristic and the matcher once per case, and passes the heuristic's choice on.
struct OfferRecorder<'a> {
    matcher: &'a pw_model::Matcher,
    cfg: Config,
    seen: RefCell<Option<CaseOffer>>,
}

impl Chooser for OfferRecorder<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> PwResult<Choice> {
        let heuristic = Heuristic { cfg: self.cfg }.choose(fp, candidates)?;
        let anchor = pw_core::describe::fingerprint_text(fp);
        let texts: Vec<String> = candidates.iter().map(|(el, _)| pw_core::describe::element_text(el)).collect();
        let scores = self.matcher.score_many(&anchor, &texts)?;
        let elements = candidates.iter().map(|(el, _)| el.index).collect();
        *self.seen.borrow_mut() = Some(CaseOffer { heuristic, elements, scores });
        Ok(heuristic)
    }
}

/// The outcome a choice makes of a case, as `relocate` makes it. Only the fields the benchmark reads are set.
fn outcome_of(choice: Choice, elements: &[u32]) -> Outcome {
    let (status, element, confidence) = match choice {
        Choice::Pick { index, confidence } => (Status::Relocated, elements.get(index).copied(), confidence),
        Choice::Unsure => (Status::Ambiguous, None, 0.0),
        Choice::Null => (Status::Broken, None, 0.0),
    };
    Outcome { status, element, confidence, top: Vec::new(), reason: String::new() }
}

/// One case, ready for any number of settings: its fixed outcome (fast path, or no candidates), or its offer.
enum Prepared {
    Fixed(Outcome),
    Offer(CaseOffer),
}

/// The combined tracker under each (accept, margin, review, veto) setting, on the same cases. Pass one asks the
/// heuristic and the matcher once per case. Every setting is then applied to those answers, so a grid costs one
/// pass through the model.
pub fn run_combined_grid(
    sites: usize,
    variants: usize,
    seed: u64,
    drift: Drift,
    matcher: &pw_model::Matcher,
    base: Config,
    settings: &[(f64, f64, f64, f64)],
) -> anyhow::Result<Vec<(Config, f64, Report)>> {
    let all = cases(sites, variants, seed, drift)?;
    let recorder = OfferRecorder { matcher, cfg: base, seen: RefCell::new(None) };
    let mut prepared = Vec::with_capacity(all.len());
    for case in &all {
        let out = relocate::relocate(&case.anchor, &case.snap, &recorder, &base)?;
        prepared.push(match recorder.seen.borrow_mut().take() {
            Some(offer) => Prepared::Offer(offer),
            None => Prepared::Fixed(out),
        });
    }
    let mut out = Vec::with_capacity(settings.len());
    for &(accept, margin, review, veto) in settings {
        let cfg = Config { matcher_accept: accept, matcher_margin: margin, matcher_review: review, ..base };
        let mut report = Report { sites, variants, ..Report::default() };
        for (case, prep) in all.iter().zip(&prepared) {
            let outcome = match prep {
                Prepared::Fixed(fixed) => fixed.clone(),
                Prepared::Offer(offer) => outcome_of(relocate::combined_choice(offer.heuristic, &offer.scores, &cfg, veto), &offer.elements),
            };
            score_base(&mut report, case, &outcome);
        }
        out.push((cfg, veto, report));
    }
    Ok(out)
}

/// As [`run_with_model`], at several confidence bars. The model is asked once per case. Its answer and the
/// heuristic's are then replayed at each bar through the same routing as `ModelChooser` and `ModelFirst`,
/// so the rows at a bar equal what `run_with_model` gives with `model_accept` set to that bar.
pub fn run_sweep(sites: usize, variants: usize, seed: u64, model: &ModelChooser, accepts: &[f64], cfg: Config) -> anyhow::Result<Sweep> {
    let heuristic = Heuristic { cfg };
    let mut report = Report { sites, variants, ..Report::default() };
    let mut bars: Vec<Bar> = accepts.iter().map(|&accept| Bar { accept, model: ModelRows::default() }).collect();
    for case in cases(sites, variants, seed, Drift::Mixed)? {
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
    fn relocation_never_sees_the_gold_marker() {
        for drift in [Drift::Mixed, Drift::Compound] {
            for case in cases(6, 2, 11, drift).unwrap() {
                assert!(case.snap.elements().iter().all(|e| !e.attrs.contains_key("data-pwgold")));
                assert!(!case.anchor.fingerprint.attrs.contains_key("data-pwgold"));
                assert!(case.gold.is_some() || case.snap.elements().len() > 0);
            }
        }
    }

    #[test]
    fn a_seed_always_gives_the_same_cases() {
        let a = cases(12, 2, 9, Drift::Mixed).unwrap();
        let b = cases(12, 2, 9, Drift::Mixed).unwrap();
        assert_eq!(a.len(), 24);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.gold, y.gold);
            assert_eq!(x.snap.elements().len(), y.snap.elements().len());
        }
    }
}
