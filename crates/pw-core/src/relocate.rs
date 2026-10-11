//! Re-anchoring: find where a patch's element went after a site update.
//!
//! Order of work (see docs/DESIGN.md):
//! 1. Fast path: at least two locators agree on one unique element and the
//!    fingerprint similarity is at least `fast_threshold`. No scoring model needed.
//! 2. Prefilter: compatible elements, scored with the similarity function, top K.
//! 3. Chooser: heuristic or model picks one candidate, or says Unsure, or Null.
//!    Unsure is AMBIGUOUS (a human picks). Null is BROKEN. Nothing is auto-applied
//!    from AMBIGUOUS or BROKEN.

use std::cmp::Ordering;

use serde::Serialize;

use crate::anchor::{Anchor, Fingerprint};
use crate::error::{PwError, Result};
use crate::similarity::{self, Weights};
use crate::snapshot::{Element, Snapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Valid,
    Relocated,
    Ambiguous,
    Broken,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Scored {
    pub element: u32,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Outcome {
    pub status: Status,
    pub element: Option<u32>,
    pub confidence: f64,
    /// Best candidates, highest first. Shown to a human when AMBIGUOUS.
    pub top: Vec<Scored>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub fast_threshold: f64,
    pub prefilter_k: usize,
    pub floor: f64,
    /// Below `floor` but at least this similar, the top candidate is still shown to a person (ambiguous), not
    /// reported as gone (broken). Nothing is applied from it, so this only changes the label.
    pub review_floor: f64,
    pub heuristic_accept: f64,
    pub heuristic_margin: f64,
    /// The heuristic also needs this much of the stored surroundings to still match.
    pub context_min: f64,
    pub model_accept: f64,
    /// The matcher model (pw-model) applies its top candidate only at or above this score...
    pub matcher_accept: f64,
    /// ...and only when it leads the second candidate by at least this much...
    pub matcher_margin: f64,
    /// ...and asks a person when its top candidate reaches this score but not `matcher_accept`.
    pub matcher_review: f64,
    /// Feature weights of the similarity score that the fast path and the prefilter use.
    pub weights: Weights,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            fast_threshold: 0.9,
            prefilter_k: 8,
            floor: 0.5,
            // Chosen on the development split (seed 101): the highest value that keeps present targets reported broken
            // under 1% (docs/DESIGN.md, section 6.6).
            review_floor: 0.45,
            heuristic_accept: 0.6,
            heuristic_margin: 0.15,
            context_min: 0.5,
            // Chosen on a development split (docs/DESIGN.md, section 6.5). 0.8 mis-matched 84% of deleted targets there.
            model_accept: 0.99,
            // Placeholders until the matcher is tuned on the development split (docs/DESIGN.md, section 6.7).
            matcher_accept: 0.5,
            matcher_margin: 0.2,
            matcher_review: 0.2,
            weights: Weights::default(),
        }
    }
}

/// What a chooser decided about the candidate list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Choice {
    Pick { index: usize, confidence: f64 },
    /// Nothing in the list is the element (model `null`, or every score below floor).
    Null,
    /// Something may match, but not confidently enough to apply.
    Unsure,
}

pub trait Chooser {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice>;
}

impl Config {
    /// Set one setting by name, for `pw train bench --set` and the tuning scripts.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let number = || value.parse::<f64>().map_err(|_| PwError::Other(format!("{key}: '{value}' is not a number")));
        match key {
            "fast_threshold" => self.fast_threshold = number()?,
            "prefilter_k" => {
                self.prefilter_k = value
                    .parse()
                    .map_err(|_| PwError::Other(format!("{key}: '{value}' is not a whole number")))?
            }
            "floor" => self.floor = number()?,
            "review_floor" => self.review_floor = number()?,
            "heuristic_accept" => self.heuristic_accept = number()?,
            "heuristic_margin" => self.heuristic_margin = number()?,
            "context_min" => self.context_min = number()?,
            "model_accept" => self.model_accept = number()?,
            "matcher_accept" => self.matcher_accept = number()?,
            "matcher_margin" => self.matcher_margin = number()?,
            "matcher_review" => self.matcher_review = number()?,
            "w_tag" => self.weights.tag = number()?,
            "w_id" => self.weights.id = number()?,
            "w_name" => self.weights.name = number()?,
            "w_class" => self.weights.class = number()?,
            "w_text" => self.weights.text = number()?,
            "w_attrs" => self.weights.attrs = number()?,
            "w_xpath" => self.weights.xpath = number()?,
            "w_ancestors" => self.weights.ancestors = number()?,
            "w_neighbors" => self.weights.neighbors = number()?,
            "w_rect" => self.weights.rect = number()?,
            _ => return Err(PwError::Other(format!("unknown relocation setting '{key}'"))),
        }
        Ok(())
    }

    /// Apply a comma-separated list of `key=value` settings, such as `floor=0.4,heuristic_margin=0.1`.
    pub fn apply(&mut self, spec: &str) -> Result<()> {
        for pair in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = pair
                .split_once('=')
                .ok_or_else(|| PwError::Other(format!("'{pair}' is not key=value")))?;
            self.set(key.trim(), value.trim())?;
        }
        Ok(())
    }
}

/// Deterministic chooser: accepts the top candidate only when it is strong and clearly ahead.
pub struct Heuristic {
    pub cfg: Config,
}

impl Chooser for Heuristic {
    fn choose(&self, fp: &Fingerprint, candidates: &[(&Element, f64)]) -> Result<Choice> {
        let Some(&(top_el, top)) = candidates.first() else {
            return Ok(Choice::Null);
        };
        if top < self.cfg.floor {
            // Too weak to apply, but a person should see a plausible candidate rather than be told the element is gone.
            return Ok(if top >= self.cfg.review_floor { Choice::Unsure } else { Choice::Null });
        }
        let second = candidates.get(1).map_or(0.0, |c| c.1);
        // Tag, text and class can all survive a look-alike elsewhere on the page.
        // Accepting also needs the surroundings to match.
        let context_ok = similarity::context_score(fp, top_el).map_or(true, |c| c >= self.cfg.context_min);
        if top >= self.cfg.heuristic_accept && top - second >= self.cfg.heuristic_margin && context_ok {
            Ok(Choice::Pick { index: 0, confidence: top })
        } else {
            Ok(Choice::Unsure)
        }
    }
}

/// The matcher's decision from its scores for the candidates, given in the order the candidates were given.
/// `pw_model::MatcherChooser` and the threshold sweep both use this, so they cannot disagree.
pub fn matcher_choice(scores: &[f64], cfg: &Config) -> Choice {
    let Some((best, top)) = scores.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return Choice::Null;
    };
    let second = scores.iter().enumerate().filter(|(i, _)| *i != best).map(|(_, s)| *s).fold(0.0_f64, f64::max);
    if top >= cfg.matcher_accept && top - second >= cfg.matcher_margin {
        Choice::Pick { index: best, confidence: top }
    } else if top >= cfg.matcher_review {
        Choice::Unsure
    } else {
        Choice::Null
    }
}

/// The combined tracker (docs/DESIGN.md, section 6.10). The heuristic decides when it picks a candidate, unless the
/// matcher's probability for that pick is below `veto` (0 turns the veto off), in which case a person decides. The
/// matcher decides when the heuristic is unsure. When the heuristic finds nothing, nothing is applied.
pub fn combined_choice(heuristic: Choice, scores: &[f64], cfg: &Config, veto: f64) -> Choice {
    match heuristic {
        Choice::Pick { index, confidence } => {
            let doubted = veto > 0.0 && scores.get(index).is_some_and(|s| *s < veto);
            if doubted {
                Choice::Unsure
            } else {
                Choice::Pick { index, confidence }
            }
        }
        Choice::Unsure => matcher_choice(scores, cfg),
        Choice::Null => Choice::Null,
    }
}

/// Tags that may stand in for each other when a site changes an element's tag.
fn compatible(anchor_tag: &str, el: &Element) -> bool {
    if anchor_tag.eq_ignore_ascii_case(&el.tag) {
        return true;
    }
    let button_like = matches!(anchor_tag, "button" | "a" | "input");
    if button_like {
        return matches!(el.tag.as_str(), "button" | "a" | "input") || el.role.as_deref() == Some("button");
    }
    false
}

/// Locators that agree on one unique element with a high fingerprint score. No model call.
pub fn fast_path(anchor: &Anchor, snap: &Snapshot, cfg: &Config) -> Result<Option<u32>> {
    let mut votes: Vec<u32> = Vec::new();
    if let Some(css) = &anchor.locators.css {
        let hits = snap.css(css)?;
        if let [only] = hits[..] {
            votes.push(only);
        }
    }
    if let Some(xpath) = &anchor.locators.xpath {
        if let [only] = snap.by_xpath(xpath)[..] {
            votes.push(only);
        }
    }
    if let Some(testid) = &anchor.locators.testid {
        if let [only] = snap.by_testid(testid)[..] {
            votes.push(only);
        }
    }
    if let Some(role) = &anchor.locators.role {
        if let [only] = snap.by_role_name(&role.role, &role.name)[..] {
            votes.push(only);
        }
    }
    for &candidate in &votes {
        let agree = votes.iter().filter(|&&v| v == candidate).count();
        if agree < 2 {
            continue;
        }
        if let Some(el) = snap.element(candidate) {
            if similarity::score_with(anchor, el, &cfg.weights) >= cfg.fast_threshold {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

/// Compatible elements with their scores, best first, at most `prefilter_k`.
pub fn prefilter<'a>(anchor: &Anchor, snap: &'a Snapshot, cfg: &Config) -> Vec<(&'a Element, f64)> {
    let mut scored: Vec<(&Element, f64)> = snap
        .elements()
        .iter()
        .filter(|el| compatible(&anchor.fingerprint.tag, el))
        .map(|el| (el, similarity::score_with(anchor, el, &cfg.weights)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.index.cmp(&b.0.index)));
    scored.truncate(cfg.prefilter_k);
    scored
}

/// Re-anchor one patch against a snapshot.
pub fn relocate(anchor: &Anchor, snap: &Snapshot, chooser: &dyn Chooser, cfg: &Config) -> Result<Outcome> {
    if anchor.locators.is_empty() {
        return Err(PwError::Other("anchor has no locators".into()));
    }
    if let Some(index) = fast_path(anchor, snap, cfg)? {
        return Ok(Outcome {
            status: Status::Valid,
            element: Some(index),
            confidence: 1.0,
            top: vec![Scored { element: index, score: 1.0 }],
            reason: "locators agree".into(),
        });
    }

    let candidates = prefilter(anchor, snap, cfg);
    let top: Vec<Scored> = candidates.iter().take(3).map(|(el, s)| Scored { element: el.index, score: *s }).collect();
    if candidates.is_empty() {
        return Ok(Outcome {
            status: Status::Broken,
            element: None,
            confidence: 0.0,
            top,
            reason: "no compatible element on the page".into(),
        });
    }

    match chooser.choose(&anchor.fingerprint, &candidates)? {
        Choice::Pick { index, confidence } => {
            let (el, _) = candidates
                .get(index)
                .ok_or_else(|| PwError::Other(format!("chooser returned index {index} for {} candidates", candidates.len())))?;
            Ok(Outcome {
                status: Status::Relocated,
                element: Some(el.index),
                confidence,
                top,
                reason: format!("picked candidate {index}"),
            })
        }
        Choice::Unsure => Ok(Outcome {
            status: Status::Ambiguous,
            element: None,
            confidence: candidates[0].1,
            top,
            reason: "no candidate is confidently the element; a human must pick".into(),
        }),
        Choice::Null => Ok(Outcome {
            status: Status::Broken,
            element: None,
            confidence: 0.0,
            top,
            reason: "chooser found no matching element".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1: &str = r#"<html><body><main><section class="cart"><p>Order total</p>
        <form id="checkout"><div class="actions"><button class="btn btn-primary" type="submit">Buy now</button></div></form>
        <p>Apply coupon</p></section></main></body></html>"#;

    const V2: &str = r#"<html><body><main><section class="cart-v2"><p>Order total</p>
        <form id="checkout"><div class="wrap"><div class="row"><button class="cta" type="submit">Place order</button></div></div></form>
        <p>Apply coupon</p></section></main></body></html>"#;

    fn anchor_from(html: &str, text: &str) -> Anchor {
        let snap = Snapshot::parse(html);
        let el = snap.elements().iter().find(|e| e.tag == "button" && e.text == text).unwrap().clone();
        Anchor::from_element(&el, &snap)
    }

    fn heuristic() -> Heuristic {
        Heuristic { cfg: Config::default() }
    }

    #[test]
    fn unchanged_page_is_valid_on_the_fast_path() {
        let anchor = anchor_from(V1, "Buy now");
        let out = relocate(&anchor, &Snapshot::parse(V1), &heuristic(), &Config::default()).unwrap();
        assert_eq!(out.status, Status::Valid);
        assert!(out.element.is_some());
    }

    #[test]
    fn moved_and_restyled_button_is_relocated_by_heuristic() {
        let anchor = anchor_from(V1, "Buy now");
        let snap = Snapshot::parse(V2);
        let out = relocate(&anchor, &snap, &heuristic(), &Config::default()).unwrap();
        println!("v2 outcome: {out:?}");
        assert_eq!(out.status, Status::Relocated);
        let target = snap.element(out.element.unwrap()).unwrap();
        assert_eq!(target.tag, "button");
    }

    #[test]
    fn demo_site_update_with_id_removed_is_relocated_by_heuristic() {
        // The fingerprint keeps the id, as the CLI's `Anchor::from_element` does.
        let v1 = r#"<html><body><main><section class="cart"><p>Order total</p><form id="checkout"><div class="actions"><button id="buy" class="btn btn-primary" type="submit">Buy now</button></div></form><p>Apply coupon</p></section></main></body></html>"#;
        let v2 = r#"<html><body><main><section class="cart-v2"><p>Order total</p><form id="checkout"><div class="wrap"><div class="row"><button class="cta" type="submit">Place order</button></div></div></form><p>Apply coupon</p></section></main></body></html>"#;
        let anchor = anchor_from(v1, "Buy now");
        assert!(anchor.fingerprint.attrs.contains_key("id"));
        let snap = Snapshot::parse(v2);
        let out = relocate(&anchor, &snap, &heuristic(), &Config::default()).unwrap();
        assert_eq!(out.status, Status::Relocated, "{out:?}");
        assert_eq!(snap.element(out.element.unwrap()).unwrap().tag, "button");
    }

    #[test]
    fn page_without_any_compatible_element_is_broken_not_guessed() {
        let anchor = anchor_from(V1, "Buy now");
        let no_buttons = r#"<html><body><p>Nothing to click here</p></body></html>"#;
        let out = relocate(&anchor, &Snapshot::parse(no_buttons), &heuristic(), &Config::default()).unwrap();
        assert_eq!(out.status, Status::Broken);
        assert!(out.element.is_none());
    }

    #[test]
    fn deleted_target_with_a_look_alike_elsewhere_is_not_auto_applied() {
        let anchor = anchor_from(V1, "Buy now");
        let footer_decoy = r#"<html><body><main><section class="cart-v2"><p>Shipping</p></section>
            <footer><div class="legal"><button class="btn btn-primary" type="submit">Buy now</button></div><p>Newsletter</p></footer>
            </main></body></html>"#;
        let out = relocate(&anchor, &Snapshot::parse(footer_decoy), &heuristic(), &Config::default()).unwrap();
        assert_eq!(out.status, Status::Ambiguous, "a look-alike must go to a human, not be applied: {out:?}");
        assert!(out.element.is_none());
        assert!(!out.top.is_empty(), "the human still gets the candidates");
    }

    #[test]
    fn two_equally_plausible_buttons_are_ambiguous() {
        let anchor = anchor_from(V1, "Buy now");
        let twins = r#"<html><body><main><section class="cart-v2"><p>Order total</p>
            <div class="a"><button class="btn btn-primary" type="submit">Buy now</button></div>
            <div class="b"><button class="btn btn-primary" type="submit">Buy now</button></div><p>Apply coupon</p>
            </section></main></body></html>"#;
        let out = relocate(&anchor, &Snapshot::parse(twins), &heuristic(), &Config::default()).unwrap();
        assert_ne!(out.status, Status::Relocated, "{out:?}");
    }

    #[test]
    fn model_null_is_broken() {
        struct Never;
        impl Chooser for Never {
            fn choose(&self, _: &Fingerprint, _: &[(&Element, f64)]) -> Result<Choice> {
                Ok(Choice::Null)
            }
        }
        let anchor = anchor_from(V1, "Buy now");
        let out = relocate(&anchor, &Snapshot::parse(V2), &Never, &Config::default()).unwrap();
        assert_eq!(out.status, Status::Broken);
    }

    #[test]
    fn out_of_range_chooser_index_is_an_error() {
        struct Bad;
        impl Chooser for Bad {
            fn choose(&self, _: &Fingerprint, _: &[(&Element, f64)]) -> Result<Choice> {
                Ok(Choice::Pick { index: 99, confidence: 0.99 })
            }
        }
        let anchor = anchor_from(V1, "Buy now");
        assert!(relocate(&anchor, &Snapshot::parse(V2), &Bad, &Config::default()).is_err());
    }

    #[test]
    fn anchor_without_locators_is_rejected() {
        let anchor = Anchor { locators: Default::default(), fingerprint: Fingerprint { tag: "button".into(), ..Default::default() } };
        assert!(relocate(&anchor, &Snapshot::parse(V1), &heuristic(), &Config::default()).is_err());
    }

    #[test]
    fn the_combined_rule_defers_to_the_heuristic_and_asks_the_matcher_when_unsure() {
        let cfg = Config { matcher_accept: 0.6, matcher_margin: 0.2, matcher_review: 0.3, ..Config::default() };
        let pick = Choice::Pick { index: 1, confidence: 0.8 };
        // A heuristic pick stands when the matcher agrees, or when the veto is off.
        assert!(matches!(combined_choice(pick, &[0.1, 0.9], &cfg, 0.0), Choice::Pick { index: 1, .. }));
        assert!(matches!(combined_choice(pick, &[0.1, 0.9], &cfg, 0.5), Choice::Pick { index: 1, .. }));
        // The veto turns a pick the matcher doubts into a person's decision.
        assert!(matches!(combined_choice(pick, &[0.9, 0.2], &cfg, 0.5), Choice::Unsure));
        // An unsure heuristic leaves the decision to the matcher's own thresholds.
        assert!(matches!(combined_choice(Choice::Unsure, &[0.2, 0.95], &cfg, 0.0), Choice::Pick { index: 1, .. }));
        assert!(matches!(combined_choice(Choice::Unsure, &[0.1, 0.05], &cfg, 0.0), Choice::Null));
        // A heuristic that found nothing is not overruled.
        assert!(matches!(combined_choice(Choice::Null, &[0.99], &cfg, 0.0), Choice::Null));
    }

    #[test]
    fn the_matcher_applies_only_a_confident_clear_leader() {
        let cfg = Config { matcher_accept: 0.6, matcher_margin: 0.2, matcher_review: 0.3, ..Config::default() };
        assert!(matches!(matcher_choice(&[0.9, 0.2], &cfg), Choice::Pick { index: 0, .. }));
        // The leader is confident but not clearly ahead of its rival: a person decides.
        assert!(matches!(matcher_choice(&[0.9, 0.8], &cfg), Choice::Unsure));
        // Plausible, below the accept bar: a person decides.
        assert!(matches!(matcher_choice(&[0.4, 0.1], &cfg), Choice::Unsure));
        // Nothing plausible: broken.
        assert!(matches!(matcher_choice(&[0.1, 0.05], &cfg), Choice::Null));
        assert!(matches!(matcher_choice(&[], &cfg), Choice::Null));
        // A pick reports the index of the leader in the given order.
        assert!(matches!(matcher_choice(&[0.2, 0.95], &cfg), Choice::Pick { index: 1, .. }));
    }

    #[test]
    fn a_plausible_but_weak_candidate_goes_to_a_person_not_to_broken() {
        let snap = Snapshot::parse("<html><body><button>Buy</button></body></html>");
        let el = snap.elements().iter().find(|e| e.tag == "button").unwrap();
        let fp = Fingerprint::default();
        let heuristic = Heuristic { cfg: Config::default() };
        // The band is [review_floor, floor) = [0.45, 0.5) with the defaults.
        assert!(matches!(heuristic.choose(&fp, &[(el, 0.47)]).unwrap(), Choice::Unsure), "0.47 is a plausible candidate");
        assert!(matches!(heuristic.choose(&fp, &[(el, 0.1)]).unwrap(), Choice::Null), "0.1 is not");
        let strict = Heuristic { cfg: Config { review_floor: 0.5, ..Config::default() } };
        assert!(matches!(strict.choose(&fp, &[(el, 0.47)]).unwrap(), Choice::Null), "with review_floor = floor, nothing is ambiguous");
    }

    #[test]
    fn settings_change_by_name_and_unknown_names_are_refused() {
        let mut cfg = Config::default();
        cfg.apply("floor=0.4, heuristic_margin=0.1,prefilter_k=12").unwrap();
        assert_eq!((cfg.floor, cfg.heuristic_margin, cfg.prefilter_k), (0.4, 0.1, 12));
        assert!(cfg.apply("flor=0.4").is_err(), "a misspelt name must not be ignored");
        assert!(cfg.apply("floor").is_err());
        assert!(cfg.apply("floor=abc").is_err());
        assert!(cfg.apply("prefilter_k=2.5").is_err());
        cfg.apply("w_tag=0.5,w_rect=0").unwrap();
        assert_eq!((cfg.weights.tag, cfg.weights.rect), (0.5, 0.0));
        assert_eq!(Config::default().weights, Weights::default(), "the default weights are the original constants");
    }
}
