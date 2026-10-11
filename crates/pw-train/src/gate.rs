//! Training the learned gate on the combined tracker (docs/DESIGN.md, section 6.13). The data is the combined rule's
//! own picks on the development slices, each with its features and whether it is the gold element. The model is a
//! logistic regression on standardised features. The threshold is the one with the best summed objective on the same
//! slices, ties going to the lower threshold. The test seeds are never read here.
//!
//! The matcher is asked once per case. Each threshold is then replayed on the recorded decisions, with the same rule
//! the gated chooser applies, so the threshold table is the gated tracker's own result.

use std::cell::RefCell;

use anyhow::{bail, Result};
use pw_core::anchor::Fingerprint;
use pw_core::error::{PwError, Result as PwResult};
use pw_core::relocate::{self, Choice, Chooser, Config};
use pw_core::snapshot::Element;
use pw_model::{CombinedChooser, Gate, Matcher, GATE_FEATURES, GATE_N, GATE_SCHEMA};

use crate::bench::{self, Case, Drift, Report};

/// The development slices, the matcher, and the rule's setting that the gate is trained alongside.
pub struct Dev<'a> {
    pub slices: Vec<(u64, Drift)>,
    pub sites: usize,
    pub variants: usize,
    pub matcher: &'a Matcher,
    pub cfg: Config,
    pub veto: f64,
    /// SHA-256 of the matcher's weights, so the gate is bound to them.
    pub matcher_sha256: String,
}

/// One pick of the combined rule on a development case.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub x: [f64; GATE_N],
    /// 1 when the pick is the gold element, 0 otherwise (every pick on a deleted target is 0).
    pub y: f64,
}

/// What the ungated rule decided for one case: its answer, the pick's features (for a pick), and whether the pick is
/// the gold element.
struct Seen {
    choice: Choice,
    features: Option<[f64; GATE_N]>,
    correct: bool,
}

/// Records the rule's own decision. The gate is not applied, so the decision is the one the gate has to judge.
struct Recorder<'c, 'm> {
    inner: &'c CombinedChooser<'m>,
    gold: Option<u32>,
    seen: RefCell<Option<Seen>>,
}

impl Chooser for Recorder<'_, '_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> PwResult<Choice> {
        let (choice, features) = self.inner.decide_with_features(fp, candidates)?;
        let correct = match choice {
            Choice::Pick { index, .. } => self.gold == Some(candidates[index].0.index),
            Choice::Null | Choice::Unsure => false,
        };
        let copy = match choice {
            Choice::Pick { index, confidence } => Choice::Pick { index, confidence },
            Choice::Null => Choice::Null,
            Choice::Unsure => Choice::Unsure,
        };
        *self.seen.borrow_mut() = Some(Seen { choice, features, correct });
        Ok(copy)
    }
}

/// Replays a recorded decision through a gate: a pick is applied only when the gate accepts its features, and
/// otherwise the answer is "needs a person". This is the rule `CombinedChooser` applies with a gate.
struct Replay<'a> {
    gate: &'a Gate,
    seen: Option<&'a Seen>,
}

impl Chooser for Replay<'_> {
    fn choose(&self, _fp: &Fingerprint, _candidates: &[(&Element, f64)]) -> PwResult<Choice> {
        let Some(seen) = self.seen else {
            return Err(PwError::Other("the replay asked about a case the rule never saw".into()));
        };
        Ok(match (&seen.choice, &seen.features) {
            (Choice::Pick { index, confidence }, Some(x)) => {
                if self.gate.accepts(x) {
                    Choice::Pick { index: *index, confidence: *confidence }
                } else {
                    Choice::Unsure
                }
            }
            (Choice::Pick { index, confidence }, None) => Choice::Pick { index: *index, confidence: *confidence },
            (Choice::Null, _) => Choice::Null,
            (Choice::Unsure, _) => Choice::Unsure,
        })
    }
}

/// One development case, with the rule's recorded decision.
struct Record {
    case: Case,
    seen: Option<Seen>,
}

/// Every case of one slice, with the ungated rule's decision on each.
fn record(sites: usize, variants: usize, seed: u64, drift: Drift, rule: &CombinedChooser<'_>) -> Result<Vec<Record>> {
    let mut out = Vec::new();
    for case in bench::cases(sites, variants, seed, drift)? {
        let recorder = Recorder { inner: rule, gold: case.gold, seen: RefCell::new(None) };
        relocate::relocate(&case.anchor, &case.snap, &recorder, &rule.cfg)?;
        let seen = recorder.seen.into_inner();
        out.push(Record { case, seen });
    }
    Ok(out)
}

/// The fitted standardisation and logistic weights.
#[derive(Clone, Debug)]
pub struct Fitted {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn standardise(x: &[f64; GATE_N], means: &[f64], stds: &[f64]) -> [f64; GATE_N] {
    let mut z = [0.0; GATE_N];
    for i in 0..GATE_N {
        z[i] = (x[i] - means[i]) / stds[i];
    }
    z
}

fn dot(w: &[f64], z: &[f64; GATE_N]) -> f64 {
    (0..GATE_N).map(|i| w[i] * z[i]).sum()
}

/// Logistic regression by full-batch gradient descent: 3000 steps at learning rate 0.1, from zero weights. The
/// gradient carries 1e-3 times each weight (the L2 term; the bias is not penalised), and there is no early stopping.
pub fn fit(rows: &[Row]) -> Result<Fitted> {
    if rows.is_empty() {
        bail!("the combined rule made no picks on the development slices, so there is nothing to train the gate on");
    }
    let n = rows.len() as f64;
    let mut means = vec![0.0; GATE_N];
    for r in rows {
        for i in 0..GATE_N {
            means[i] += r.x[i] / n;
        }
    }
    let mut stds = vec![0.0; GATE_N];
    for r in rows {
        for i in 0..GATE_N {
            stds[i] += (r.x[i] - means[i]).powi(2) / n;
        }
    }
    for s in &mut stds {
        *s = s.sqrt().max(1e-9);
    }
    let zs: Vec<[f64; GATE_N]> = rows.iter().map(|r| standardise(&r.x, &means, &stds)).collect();
    let mut w = vec![0.0; GATE_N];
    let mut b = 0.0;
    for _ in 0..3000 {
        let mut gw = vec![0.0; GATE_N];
        let mut gb = 0.0;
        for (z, r) in zs.iter().zip(rows) {
            let e = sigmoid(dot(&w, z) + b) - r.y;
            for i in 0..GATE_N {
                gw[i] += e * z[i] / n;
            }
            gb += e / n;
        }
        for i in 0..GATE_N {
            w[i] -= 0.1 * (gw[i] + 1e-3 * w[i]);
        }
        b -= 0.1 * gb;
    }
    Ok(Fitted { means, stds, weights: w, bias: b })
}

/// The thresholds tried, 0.05 to 0.95 in steps of 0.05.
pub fn thresholds() -> Vec<f64> {
    (1..=19).map(|k| k as f64 * 0.05).collect()
}

/// The threshold with the highest summed objective. Ties go to the lower threshold. `table` is (threshold, objective).
pub fn best_threshold(table: &[(f64, i64)]) -> Option<f64> {
    table
        .iter()
        .fold(None, |best: Option<(f64, i64)>, &(t, j)| match best {
            None => Some((t, j)),
            Some((bt, bj)) if j > bj || (j == bj && t < bt) => Some((t, j)),
            keep => keep,
        })
        .map(|(t, _)| t)
}

fn gate_with(fitted: &Fitted, dev: &Dev<'_>, threshold: f64, table: Vec<(f64, i64)>) -> Gate {
    Gate {
        schema: GATE_SCHEMA,
        matcher_sha256: dev.matcher_sha256.clone(),
        features: GATE_FEATURES.iter().map(|s| s.to_string()).collect(),
        means: fitted.means.clone(),
        stds: fitted.stds.clone(),
        weights: fitted.weights.clone(),
        bias: fitted.bias,
        threshold,
        trained_on: dev.slices.iter().map(|(seed, drift)| format!("{seed}:{}", format!("{drift:?}").to_lowercase())).collect(),
        threshold_table: table,
        written_by: "pw train gate".into(),
    }
}

/// Train the gate on the development slices and choose its threshold there. The gate it returns carries the table of
/// summed objectives it was chosen from.
pub fn train(dev: &Dev<'_>) -> Result<Gate> {
    let rule = CombinedChooser { matcher: dev.matcher, cfg: dev.cfg, veto: dev.veto, gate: None };
    let mut slices = Vec::new();
    for &(seed, drift) in &dev.slices {
        slices.push(record(dev.sites, dev.variants, seed, drift, &rule)?);
    }
    let rows: Vec<Row> = slices
        .iter()
        .flatten()
        .filter_map(|r| {
            let seen = r.seen.as_ref()?;
            let x = seen.features?;
            Some(Row { x, y: if seen.correct { 1.0 } else { 0.0 } })
        })
        .collect();
    let fitted = fit(&rows)?;
    let mut table = Vec::new();
    for threshold in thresholds() {
        let gate = gate_with(&fitted, dev, threshold, Vec::new());
        let mut total = 0i64;
        for records in &slices {
            let mut report = Report { sites: dev.sites, variants: dev.variants, ..Report::default() };
            for rec in records {
                let replay = Replay { gate: &gate, seen: rec.seen.as_ref() };
                let outcome = relocate::relocate(&rec.case.anchor, &rec.case.snap, &replay, &dev.cfg)?;
                bench::score_base(&mut report, &rec.case, &outcome);
            }
            total += bench::objective(&report);
        }
        table.push((threshold, total));
    }
    let Some(chosen) = best_threshold(&table) else {
        bail!("no threshold was tried");
    };
    Ok(gate_with(&fitted, dev, chosen, table))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_regression_learns_a_separable_rule() {
        // Positive rows have the first feature high, negative rows low. Every other feature is noise.
        let mut rows = Vec::new();
        for i in 0..200 {
            let positive = i % 2 == 0;
            let mut x = [0.0; GATE_N];
            x[0] = if positive { 1.0 } else { -1.0 } + (i as f64 * 0.01).sin() * 0.1;
            for j in 1..GATE_N {
                x[j] = (i as f64 * (j as f64 + 0.3)).sin();
            }
            rows.push(Row { x, y: if positive { 1.0 } else { 0.0 } });
        }
        let fitted = fit(&rows).unwrap();
        let mut correct = 0;
        for r in &rows {
            let z = standardise(&r.x, &fitted.means, &fitted.stds);
            let p = sigmoid(dot(&fitted.weights, &z) + fitted.bias);
            if (p >= 0.5) == (r.y == 1.0) {
                correct += 1;
            }
        }
        assert!(correct as f64 / rows.len() as f64 > 0.98, "{correct} of {}", rows.len());
    }

    #[test]
    fn the_threshold_is_the_best_objective_and_ties_go_low() {
        assert_eq!(best_threshold(&[(0.5, 1), (0.6, 5), (0.7, 3)]), Some(0.6));
        assert_eq!(best_threshold(&[(0.5, 10), (0.6, 10), (0.7, 9)]), Some(0.5));
        assert_eq!(best_threshold(&[]), None);
    }

    #[test]
    fn no_picks_means_no_gate() {
        assert!(fit(&[]).is_err());
    }
}
