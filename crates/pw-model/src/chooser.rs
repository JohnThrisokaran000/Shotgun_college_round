//! The model as a relocation chooser. Same prompt and answer format as the training data (`pw_core::prompt`).

use std::sync::Mutex;

use pw_core::anchor::Fingerprint;
use pw_core::error::{PwError, Result};
use pw_core::prompt::{self, Answer, MAX_CANDIDATES};
use pw_core::relocate::{Choice, Chooser};
use pw_core::snapshot::Element;

use crate::model::Model;

/// Longest answer the model may write: `{"match": 7}` or `{"match": null}` fits in a few tokens.
const MAX_ANSWER_TOKENS: usize = 24;

pub struct ModelChooser {
    model: Mutex<Model>,
    /// The answer is applied only when the model's confidence reaches this (`pw_core::relocate::Config`).
    pub model_accept: f64,
}

impl ModelChooser {
    pub fn new(model: Model, model_accept: f64) -> Self {
        Self { model: Mutex::new(model), model_accept }
    }
}

/// What the model answered, and how sure it was.
#[derive(Debug, PartialEq)]
pub struct Judgement {
    /// `None` when the answer did not parse, or named a candidate that is not there.
    pub answer: Option<Answer>,
    pub confidence: f64,
}

impl ModelChooser {
    /// Ask the model about a candidate list. Never fails on a bad answer: that is reported as `answer: None`.
    pub fn judge(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Judgement> {
        let shown = candidates.len().min(MAX_CANDIDATES);
        let text = prompt::relocate_prompt(fp, candidates);
        let generated = {
            let mut model = self.model.lock().map_err(|_| PwError::Other("the model lock was poisoned".into()))?;
            model.generate(&text, MAX_ANSWER_TOKENS)?
        };
        Ok(Judgement {
            answer: prompt::parse_answer(&generated.text, shown).ok(),
            confidence: generated.confidence(),
        })
    }
}

impl Chooser for ModelChooser {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        if candidates.is_empty() {
            return Ok(Choice::Null);
        }
        Ok(alone(&self.judge(fp, candidates)?, self.model_accept))
    }
}

/// The model alone. Its answer is applied when it is confident enough (`accept`). Otherwise the result is
/// `Unsure`. A confident refusal is `Null`.
pub fn alone(judged: &Judgement, accept: f64) -> Choice {
    if judged.confidence < accept {
        return Choice::Unsure;
    }
    match judged.answer {
        Some(Answer::Match(index)) => Choice::Pick { index, confidence: judged.confidence },
        Some(Answer::Null) => Choice::Null,
        None => Choice::Unsure,
    }
}

/// The model decides when it is confident, and the heuristic decides otherwise.
///
/// A confident refusal from the model is final: the heuristic cannot pick an element the model said is not
/// there. The heuristic is used only when the model is unsure, or when its answer did not parse.
pub struct ModelFirst<'a> {
    pub model: &'a ModelChooser,
    pub fallback: &'a dyn Chooser,
}

impl Chooser for ModelFirst<'_> {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        if candidates.is_empty() {
            return Ok(Choice::Null);
        }
        let judged = self.model.judge(fp, candidates)?;
        decide(&judged, self.model.model_accept, || self.fallback.choose(fp, candidates))
    }
}

/// The routing rules for `ModelFirst`, separated so they can be tested without a model.
pub fn decide(judged: &Judgement, accept: f64, fallback: impl FnOnce() -> Result<Choice>) -> Result<Choice> {
    let confident = judged.confidence >= accept;
    match judged.answer {
        Some(Answer::Match(index)) if confident => Ok(Choice::Pick { index, confidence: judged.confidence }),
        Some(Answer::Null) if confident => Ok(Choice::Null),
        // A refusal that is not confident is still a refusal: the heuristic must not override it.
        Some(Answer::Null) => Ok(Choice::Unsure),
        // A confident but unparseable answer, or a pick the model is not sure of, goes to the heuristic.
        _ => fallback(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judged(answer: Option<Answer>, confidence: f64) -> Judgement {
        Judgement { answer, confidence }
    }

    fn heuristic_pick() -> Result<Choice> {
        Ok(Choice::Pick { index: 9, confidence: 0.7 })
    }

    #[test]
    fn the_model_alone_applies_only_a_confident_answer() {
        assert_eq!(alone(&judged(Some(Answer::Match(2)), 0.95), 0.8), Choice::Pick { index: 2, confidence: 0.95 });
        assert_eq!(alone(&judged(Some(Answer::Match(2)), 0.4), 0.8), Choice::Unsure);
        assert_eq!(alone(&judged(Some(Answer::Null), 0.9), 0.8), Choice::Null);
        assert_eq!(alone(&judged(Some(Answer::Null), 0.4), 0.8), Choice::Unsure);
        assert_eq!(alone(&judged(None, 0.99), 0.8), Choice::Unsure);
    }

    #[test]
    fn a_confident_pick_is_applied_without_asking_the_heuristic() {
        let got = decide(&judged(Some(Answer::Match(2)), 0.95), 0.8, || panic!("fallback must not run")).unwrap();
        assert_eq!(got, Choice::Pick { index: 2, confidence: 0.95 });
    }

    #[test]
    fn a_confident_refusal_is_final_and_the_heuristic_cannot_pick_over_it() {
        let got = decide(&judged(Some(Answer::Null), 0.9), 0.8, || panic!("fallback must not run")).unwrap();
        assert_eq!(got, Choice::Null);
    }

    #[test]
    fn an_unconfident_refusal_is_unsure_not_a_pick() {
        let got = decide(&judged(Some(Answer::Null), 0.4), 0.8, || panic!("fallback must not run")).unwrap();
        assert_eq!(got, Choice::Unsure);
    }

    #[test]
    fn an_unconfident_pick_defers_to_the_heuristic() {
        assert_eq!(decide(&judged(Some(Answer::Match(2)), 0.4), 0.8, heuristic_pick).unwrap(), heuristic_pick().unwrap());
    }

    #[test]
    fn an_unparseable_answer_defers_to_the_heuristic() {
        assert_eq!(decide(&judged(None, 0.99), 0.8, heuristic_pick).unwrap(), heuristic_pick().unwrap());
    }

    #[test]
    fn a_pick_is_never_applied_below_the_bar_however_it_was_routed() {
        for confidence in [0.0, 0.3, 0.79] {
            let got = decide(&judged(Some(Answer::Match(0)), confidence), 0.8, || Ok(Choice::Unsure)).unwrap();
            assert!(!matches!(got, Choice::Pick { .. }), "a pick at confidence {confidence} was applied");
        }
    }
}
