//! The matcher: a small cross-encoder trained from scratch (`training/match.py`) that scores whether a candidate
//! element is the element an anchor was made from. It reads the two descriptions built by `pw_core::describe`,
//! the same text the training pairs were written with (`pw train match-data`).
//!
//! As a relocation chooser it is a suggestion, like the decoder in this crate: it applies its top candidate only
//! when that candidate is confident and clearly ahead (`Config::matcher_accept`, `matcher_margin`). It asks a person
//! when a candidate is plausible (`matcher_review`), and it never guesses below that.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use pw_core::anchor::Fingerprint;
use pw_core::describe;
use pw_core::error::{PwError, Result};
use pw_core::relocate::{self, Choice, Chooser, Config as RelocateConfig};
use pw_core::snapshot::Element;
use tokenizers::Tokenizer;

use crate::gate::{self, Gate};

fn model_error(what: &str, e: impl std::fmt::Display) -> PwError {
    PwError::Other(format!("matcher {what}: {e}"))
}

pub struct Matcher {
    bert: BertModel,
    head: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl Matcher {
    /// Scores for several candidate descriptions against one anchor description, in one batched forward pass.
    /// Each row is padded to the longest pair in the batch, and the padding is masked, so a row scores the same
    /// as it would alone (checked in the tests below).
    pub fn score_many(&self, anchor: &str, candidates: &[String]) -> Result<Vec<f64>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = candidates
            .iter()
            .map(|c| self.tokenizer.encode((anchor, c.as_str()), true))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| model_error("tokenizing", e))?;
        let rows = encodings.len();
        let width = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let mut ids = vec![0u32; rows * width];
        let mut types = vec![0u32; rows * width];
        let mut mask = vec![0u32; rows * width];
        for (row, e) in encodings.iter().enumerate() {
            let n = e.get_ids().len();
            ids[row * width..row * width + n].copy_from_slice(e.get_ids());
            types[row * width..row * width + n].copy_from_slice(e.get_type_ids());
            mask[row * width..row * width + n].copy_from_slice(e.get_attention_mask());
        }
        let shape = (rows, width);
        let ids = Tensor::from_vec(ids, shape, &self.device).map_err(|e| model_error("input", e))?;
        let types = Tensor::from_vec(types, shape, &self.device).map_err(|e| model_error("input", e))?;
        let mask = Tensor::from_vec(mask, shape, &self.device).map_err(|e| model_error("input", e))?;
        let hidden = self.bert.forward(&ids, &types, Some(&mask)).map_err(|e| model_error("forward", e))?;
        let cls = hidden.narrow(1, 0, 1).and_then(|t| t.squeeze(1)).map_err(|e| model_error("forward", e))?;
        let logits = self.head.forward(&cls).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| model_error("head", e))?;
        Ok(logits.into_iter().map(|l| 1.0 / (1.0 + (-(l as f64)).exp())).collect())
    }

    /// Load a folder written by `training/match.py`: config.json, model.safetensors, tokenizer.json.
    pub fn load(dir: &Path) -> Result<Matcher> {
        let config: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let device = Device::Cpu;
        // Safe in practice: the file is only read, and it is not changed while the process runs.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device) }
            .map_err(|e| model_error("weights", e))?;
        let bert = BertModel::load(vb.clone(), &config).map_err(|e| model_error("encoder", e))?;
        let head = linear(config.hidden_size, 1, vb.pp("head")).map_err(|e| model_error("head", e))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| model_error("tokenizer", e))?;
        Ok(Matcher { bert, head, tokenizer, device })
    }

    /// The score for one anchor description and one candidate description, between 0 and 1.
    pub fn score(&self, anchor: &str, candidate: &str) -> Result<f64> {
        let encoding = self.tokenizer.encode((anchor, candidate), true).map_err(|e| model_error("tokenizing", e))?;
        let ids = Tensor::new(encoding.get_ids(), &self.device).and_then(|t| t.unsqueeze(0)).map_err(|e| model_error("input", e))?;
        let types = Tensor::new(encoding.get_type_ids(), &self.device).and_then(|t| t.unsqueeze(0)).map_err(|e| model_error("input", e))?;
        let mask = Tensor::new(encoding.get_attention_mask(), &self.device).and_then(|t| t.unsqueeze(0)).map_err(|e| model_error("input", e))?;
        let hidden = self.bert.forward(&ids, &types, Some(&mask)).map_err(|e| model_error("forward", e))?;
        // The score is read from the [CLS] position, as in training/match.py.
        let cls = hidden.narrow(1, 0, 1).and_then(|t| t.squeeze(1)).map_err(|e| model_error("forward", e))?;
        let logit = self.head.forward(&cls).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| model_error("head", e))?;
        let logit = *logit.first().ok_or_else(|| model_error("head", "no output"))? as f64;
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

/// The matcher as a relocation chooser. The thresholds come from `pw_core::relocate::Config`, so the benchmark
/// can change them with `--set`.
pub struct MatcherChooser {
    pub matcher: Matcher,
    pub cfg: RelocateConfig,
}

impl MatcherChooser {
    pub fn new(matcher: Matcher, cfg: RelocateConfig) -> Self {
        MatcherChooser { matcher, cfg }
    }
}

/// The matcher alone as a chooser (experimental): its own thresholds decide, and the heuristic is not consulted.
pub struct MatcherOnly<'a> {
    pub matcher: &'a Matcher,
    pub cfg: RelocateConfig,
}

impl Chooser for MatcherOnly<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        if candidates.is_empty() {
            return Ok(Choice::Null);
        }
        let anchor = describe::fingerprint_text(fp);
        let texts: Vec<String> = candidates.iter().map(|(el, _)| describe::element_text(el)).collect();
        let scores = self.matcher.score_many(&anchor, &texts)?;
        Ok(relocate::matcher_choice(&scores, &self.cfg))
    }
}

/// The combined tracker as a chooser (docs/DESIGN.md, section 6.10): the heuristic decides when it picks a
/// candidate, unless the matcher doubts the pick (probability below `veto`); the matcher decides when the heuristic
/// is unsure. `cfg` carries the matcher thresholds. Opt-in: the heuristic alone stays the default.
pub struct CombinedChooser<'a> {
    pub matcher: &'a Matcher,
    pub cfg: RelocateConfig,
    pub veto: f64,
    /// The learned gate on the pick, when the matcher's folder has one (docs/DESIGN.md, section 6.13). `None` applies
    /// the rule as it was before the gate existed.
    pub gate: Option<&'a Gate>,
}

impl CombinedChooser<'_> {
    /// The rule's answer, and for a pick the features the gate sees. The gate is not applied here, so training can
    /// record the rule's own picks and their features.
    pub fn decide_with_features(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<(Choice, Option<[f64; gate::N]>)> {
        let heuristic = relocate::Heuristic { cfg: self.cfg }.choose(fp, candidates)?;
        if candidates.is_empty() || heuristic == Choice::Null {
            return Ok((heuristic, None));
        }
        let anchor = describe::fingerprint_text(fp);
        let texts: Vec<String> = candidates.iter().map(|(el, _)| describe::element_text(el)).collect();
        let scores = self.matcher.score_many(&anchor, &texts)?;
        let choice = relocate::combined_choice(heuristic, &scores, &self.cfg, self.veto);
        let Choice::Pick { index, .. } = choice else {
            return Ok((choice, None));
        };
        let sims: Vec<f64> = candidates.iter().map(|(_, s)| *s).collect();
        let x = gate::features(fp, candidates[index].0, scores[index], margin(&scores, index), sims[index], margin(&sims, index));
        Ok((choice, Some(x)))
    }
}

/// How far `values[index]` leads the best of the others. A lone value leads by itself.
fn margin(values: &[f64], index: usize) -> f64 {
    let own = values[index];
    let best_other = values.iter().enumerate().filter(|(i, _)| *i != index).map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    if best_other == f64::NEG_INFINITY {
        own
    } else {
        own - best_other
    }
}

impl Chooser for CombinedChooser<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        let (choice, features) = self.decide_with_features(fp, candidates)?;
        match (choice, self.gate, features) {
            (Choice::Pick { index, confidence }, Some(gate), Some(x)) => {
                if gate.accepts(&x) {
                    Ok(Choice::Pick { index, confidence })
                } else {
                    Ok(Choice::Unsure)
                }
            }
            (choice, _, _) => Ok(choice),
        }
    }
}

impl Chooser for MatcherChooser {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        if candidates.is_empty() {
            return Ok(Choice::Null);
        }
        let anchor = describe::fingerprint_text(fp);
        let texts: Vec<String> = candidates.iter().map(|(el, _)| describe::element_text(el)).collect();
        let scores = self.matcher.score_many(&anchor, &texts)?;
        Ok(relocate::matcher_choice(&scores, &self.cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The batched scores equal the single-pair scores. Runs only where a trained matcher is present
    /// (models/matcher-v2, or the folder named by PW_MATCHER); elsewhere it says so and passes.
    #[test]
    fn batched_scores_equal_single_scores() {
        let dir = std::env::var("PW_MATCHER").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/matcher-v2")
        });
        if !dir.join("model.safetensors").exists() {
            eprintln!("no matcher at {}; skipping", dir.display());
            return;
        }
        let matcher = Matcher::load(&dir).unwrap();
        let anchor = "tag button | id buy | class btn | text buy now | in section.cart > main";
        let candidates = vec![
            "tag button | id buy | class btn | text buy now | in section.cart > main".to_string(),
            "tag button | class btn btn-main | text place order | in div > section.cart > main".to_string(),
            "tag p | text apply coupon | in section.cart > main".to_string(),
        ];
        let batched = matcher.score_many(anchor, &candidates).unwrap();
        for (c, b) in candidates.iter().zip(&batched) {
            let single = matcher.score(anchor, c).unwrap();
            assert!((single - b).abs() < 1e-4, "single {single} batched {b}");
        }
    }
}
