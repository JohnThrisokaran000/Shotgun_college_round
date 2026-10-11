//! A learned gate on the combined tracker's picks (docs/DESIGN.md, section 6.13). When the combined rule picks a
//! candidate, the gate looks at structure the matcher does not see, and applies the pick only when its probability
//! is at least the threshold. Otherwise the answer is "needs a person".
//!
//! The gate is a logistic regression on eleven features, all available at run time. It is stored as `gate.json` in
//! the matcher's folder, bound to the matcher's weights by their SHA-256, like the setting in `selection.json`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use pw_core::anchor::Fingerprint;
use pw_core::error::{PwError, Result};
use pw_core::hash::file_sha256;
use pw_core::snapshot::Element;

/// The file name in a matcher's folder.
pub const GATE_FILE: &str = "gate.json";
/// The record format this build reads and writes.
pub const GATE_SCHEMA: u32 = 1;
/// The feature names, in the order `features` returns them.
pub const FEATURES: [&str; 11] = [
    "p",
    "p_margin",
    "sim",
    "sim_margin",
    "tag_equal",
    "class_jaccard",
    "attr_agree",
    "ancestor_jaccard",
    "neighbour_jaccard",
    "text_jaccard",
    "depth_ratio",
];
/// How many features the gate uses.
pub const N: usize = 11;

/// A trained gate. `threshold` is the one chosen on the development seeds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    pub schema: u32,
    /// SHA-256 of the `model.safetensors` the gate was trained alongside.
    pub matcher_sha256: String,
    pub features: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub threshold: f64,
    /// The development slices the gate was trained on, as `SEED:mixed` or `SEED:compound`.
    pub trained_on: Vec<String>,
    /// The summed objective on the development slices for each threshold tried, in the order tried.
    pub threshold_table: Vec<(f64, i64)>,
    pub written_by: String,
}

impl Gate {
    /// The probability that the pick with these features is correct.
    pub fn probability(&self, x: &[f64; N]) -> f64 {
        let z: f64 = (0..N).map(|i| self.weights[i] * ((x[i] - self.means[i]) / self.stds[i])).sum::<f64>() + self.bias;
        1.0 / (1.0 + (-z).exp())
    }

    /// Whether the gate applies the pick.
    pub fn accepts(&self, x: &[f64; N]) -> bool {
        self.probability(x) >= self.threshold
    }

    /// The gate in `dir`, if the folder has one. A gate that does not parse is an error, not a silent fallback.
    pub fn load(dir: &Path) -> Result<Option<Gate>> {
        let path = dir.join(GATE_FILE);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let gate: Gate = serde_json::from_str(&text).map_err(|e| PwError::Other(format!("{}: {e}", path.display())))?;
        if gate.schema != GATE_SCHEMA {
            return Err(PwError::Other(format!("{}: schema {} is not {GATE_SCHEMA}", path.display(), gate.schema)));
        }
        if gate.weights.len() != N || gate.means.len() != N || gate.stds.len() != N {
            return Err(PwError::Other(format!("{}: expected {N} features", path.display())));
        }
        Ok(Some(gate))
    }

    /// Refuse the gate when `dir` holds different matcher weights from the ones it was trained alongside.
    pub fn check_matcher(&self, dir: &Path) -> Result<()> {
        let actual = file_sha256(&dir.join("model.safetensors"))?;
        if actual != self.matcher_sha256 {
            return Err(PwError::Other(format!(
                "{} has a gate trained for different weights: model.safetensors changed after the gate was trained. Train the gate again.",
                dir.display()
            )));
        }
        Ok(())
    }

    /// Write the gate into `dir`, replacing an older one.
    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        let path = dir.join(GATE_FILE);
        fs::write(&path, format!("{}\n", serde_json::to_string_pretty(self)?))?;
        Ok(path)
    }
}

/// Jaccard overlap of two sets. Two empty sets have overlap 0, since there is no evidence either way.
fn jaccard<T: Eq + std::hash::Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// The lower-case runs of letters and digits in `text`.
fn tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The class names in a `class` attribute value.
fn classes(value: Option<&str>) -> HashSet<String> {
    value.unwrap_or("").split_whitespace().map(str::to_string).collect()
}

/// The eleven features of one picked candidate, in the order of `FEATURES`. `p` and `sim` are the matcher's
/// probability and the heuristic similarity of the pick; the margins are against the next candidate.
pub fn features(fp: &Fingerprint, el: &Element, p: f64, p_margin: f64, sim: f64, sim_margin: f64) -> [f64; N] {
    let cand = Fingerprint::from_element(el);
    let tag_equal = if fp.tag.eq_ignore_ascii_case(&cand.tag) { 1.0 } else { 0.0 };
    let class_jaccard = jaccard(
        &classes(fp.attrs.get("class").map(String::as_str)),
        &classes(el.attrs.get("class").map(String::as_str)),
    );
    let (mut total, mut same) = (0usize, 0usize);
    for (name, value) in &fp.attrs {
        if name == "id" || name.starts_with("data-") {
            continue;
        }
        total += 1;
        if el.attrs.get(name) == Some(value) {
            same += 1;
        }
    }
    let attr_agree = if total == 0 { 0.0 } else { same as f64 / total as f64 };
    let ancestors_a: HashSet<String> = fp.ancestors.iter().cloned().collect();
    let ancestors_b: HashSet<String> = el.ancestors.iter().cloned().collect();
    let neighbours_a: HashSet<String> = fp.neighbors.iter().cloned().collect();
    let neighbours_b: HashSet<String> = el.neighbors.iter().cloned().collect();
    let text_a = tokens(fp.text.as_deref().unwrap_or(""));
    let text_b = tokens(&el.text);
    let (da, db) = (fp.ancestors.len(), el.ancestors.len());
    let depth_ratio = if da == 0 && db == 0 { 1.0 } else { da.min(db) as f64 / da.max(db) as f64 };
    [
        p,
        p_margin,
        sim,
        sim_margin,
        tag_equal,
        class_jaccard,
        attr_agree,
        jaccard(&ancestors_a, &ancestors_b),
        jaccard(&neighbours_a, &neighbours_b),
        jaccard(&text_a, &text_b),
        depth_ratio,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pw_core::snapshot::Snapshot;

    fn element(html: &str, tag: &str) -> Element {
        let snap = Snapshot::parse(html);
        snap.elements().iter().find(|e| e.tag == tag).expect("element in the test page").clone()
    }

    fn gate_with(weights: Vec<f64>, bias: f64, threshold: f64) -> Gate {
        Gate {
            schema: GATE_SCHEMA,
            matcher_sha256: "x".into(),
            features: FEATURES.iter().map(|s| s.to_string()).collect(),
            means: vec![0.0; N],
            stds: vec![1.0; N],
            weights,
            bias,
            threshold,
            trained_on: vec![],
            threshold_table: vec![],
            written_by: "test".into(),
        }
    }

    #[test]
    fn an_element_matches_itself_on_every_structural_feature() {
        let html = r#"<main><section class="cart"><button id="buy" class="btn" type="submit">Buy now</button></section></main>"#;
        let el = element(html, "button");
        let fp = Fingerprint::from_element(&el);
        let x = features(&fp, &el, 0.9, 0.5, 1.0, 0.4);
        assert_eq!(x[4], 1.0, "same tag");
        assert_eq!(x[5], 1.0, "same class tokens");
        assert_eq!(x[6], 1.0, "every stored attribute agrees");
        assert_eq!(x[7], 1.0, "same ancestors");
        assert_eq!(x[9], 1.0, "same text tokens");
        assert_eq!(x[10], 1.0, "same depth");
    }

    #[test]
    fn a_different_element_disagrees_on_text_and_tag() {
        let a = element(r#"<div><button class="btn">Buy now</button></div>"#, "button");
        let b = element(r#"<div><p class="note">Free returns</p></div>"#, "p");
        let x = features(&Fingerprint::from_element(&a), &b, 0.2, 0.0, 0.3, 0.0);
        assert_eq!(x[4], 0.0, "different tag");
        assert_eq!(x[9], 0.0, "no shared text tokens");
        assert_eq!(x[5], 0.0, "no shared classes");
    }

    #[test]
    fn two_empty_sets_give_no_evidence() {
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(jaccard(&empty, &empty), 0.0);
    }

    #[test]
    fn the_gate_is_monotone_in_its_evidence_and_applies_the_threshold() {
        let gate = gate_with(vec![1.0; N], 0.0, 0.5);
        let low = [-3.0; N];
        let high = [3.0; N];
        assert!(gate.probability(&high) > gate.probability(&low));
        assert!(gate.accepts(&high));
        assert!(!gate.accepts(&low));
    }

    #[test]
    fn a_gate_round_trips_and_is_refused_for_other_weights() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("gate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), b"weights v1").unwrap();
        let mut gate = gate_with(vec![0.5; N], -1.0, 0.7);
        gate.matcher_sha256 = file_sha256(&dir.join("model.safetensors")).unwrap();
        gate.write(&dir).unwrap();
        let loaded = Gate::load(&dir).unwrap().expect("gate present");
        assert_eq!(loaded, gate);
        loaded.check_matcher(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), b"weights v2").unwrap();
        let err = loaded.check_matcher(&dir).unwrap_err().to_string();
        assert!(err.contains("different weights"), "{err}");
    }

    #[test]
    fn a_folder_without_a_gate_has_none() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("gate-none-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert!(Gate::load(&dir).unwrap().is_none());
    }
}
