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
use crate::similarity;
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
    pub heuristic_accept: f64,
    pub heuristic_margin: f64,
    /// The heuristic also needs this much of the stored surroundings to still match.
    pub context_min: f64,
    pub model_accept: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            fast_threshold: 0.9,
            prefilter_k: 8,
            floor: 0.5,
            heuristic_accept: 0.6,
            heuristic_margin: 0.15,
            context_min: 0.5,
            // Chosen on a development split (docs/DESIGN.md, section 6.5). 0.8 mis-matched 84% of deleted targets there.
            model_accept: 0.99,
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
            return Ok(Choice::Null);
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
            if similarity::score(anchor, el) >= cfg.fast_threshold {
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
        .map(|el| (el, similarity::score(anchor, el)))
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
}
