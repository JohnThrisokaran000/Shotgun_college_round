//! Similo-style weighted similarity between a stored anchor and a live element.
//!
//! Each feature contributes only when the stored anchor has it. A feature the
//! stored anchor has but the candidate lacks scores 0. Weights are a proposal;
//! they must be tuned on the held-out benchmark, not by feel.

use std::collections::HashSet;

use strsim::normalized_levenshtein;

use crate::anchor::{normalize_text, Anchor, Fingerprint};
use crate::snapshot::{Element, Rect};

pub const W_TAG: f64 = 2.0;
pub const W_ID: f64 = 2.0;
pub const W_NAME: f64 = 2.0;
pub const W_CLASS: f64 = 1.0;
pub const W_TEXT: f64 = 2.0;
pub const W_ATTRS: f64 = 1.0;
pub const W_XPATH: f64 = 1.0;
pub const W_ANCESTORS: f64 = 1.0;
pub const W_NEIGHBORS: f64 = 2.0;
pub const W_RECT: f64 = 1.0;

#[derive(Default)]
struct Accumulator {
    weighted: f64,
    total: f64,
}

impl Accumulator {
    fn add(&mut self, weight: f64, score: f64) {
        self.weighted += weight * score.clamp(0.0, 1.0);
        self.total += weight;
    }

    fn value(&self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            self.weighted / self.total
        }
    }
}

fn jaccard<'a, I, J>(a: I, b: J) -> f64
where
    I: IntoIterator<Item = &'a str>,
    J: IntoIterator<Item = &'a str>,
{
    let a: HashSet<&str> = a.into_iter().collect();
    let b: HashSet<&str> = b.into_iter().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    inter / union
}

/// `tag#id`, `tag.class`, or `tag` -> `tag`.
fn ancestor_tag(label: &str) -> &str {
    label.split(['#', '.']).next().unwrap_or(label)
}

fn rect_score(a: Rect, b: Rect) -> f64 {
    let dist = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
    let position = (1.0 - dist / 200.0).clamp(0.0, 1.0);
    let ratio = |p: f64, q: f64| match (p <= 0.0, q <= 0.0) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.0,
        _ => p.min(q) / p.max(q),
    };
    let size = 0.5 * (ratio(a.w, b.w) + ratio(a.h, b.h));
    0.5 * position + 0.5 * size
}

/// Share of the stored surroundings (ancestors or neighbours) that still match.
/// `None` when the fingerprint stored neither, so there is nothing to check.
pub fn context_score(fp: &Fingerprint, el: &Element) -> Option<f64> {
    // Exact labels here (not tags): the guard is about the surroundings looking the same.
    let ancestors = (!fp.ancestors.is_empty()).then(|| {
        fp.ancestors.iter().filter(|a| el.ancestors.contains(a)).count() as f64 / fp.ancestors.len() as f64
    });
    let neighbors = (!fp.neighbors.is_empty()).then(|| {
        let stored: Vec<String> = fp.neighbors.iter().map(|n| normalize_text(n)).collect();
        let live: Vec<String> = el.neighbors.iter().map(|n| normalize_text(n)).collect();
        jaccard(stored.iter().map(String::as_str), live.iter().map(String::as_str))
    });
    match (ancestors, neighbors) {
        (None, None) => None,
        (a, n) => Some(a.unwrap_or(0.0).max(n.unwrap_or(0.0))),
    }
}

/// Similarity in [0, 1] between the anchor's stored description and `el`.
pub fn score(anchor: &Anchor, el: &Element) -> f64 {
    let fp = &anchor.fingerprint;
    let mut acc = Accumulator::default();

    acc.add(W_TAG, if fp.tag.eq_ignore_ascii_case(&el.tag) { 1.0 } else { 0.0 });

    // A removed id or class is no evidence either way; a different one is a mismatch.
    if let Some(id) = fp.attrs.get("id").filter(|s| !s.is_empty()) {
        if let Some(live) = el.attrs.get("id").filter(|s| !s.is_empty()) {
            acc.add(W_ID, if live == id { 1.0 } else { 0.0 });
        }
    }
    if let Some(name) = fp.attrs.get("name").filter(|s| !s.is_empty()) {
        acc.add(W_NAME, if el.attrs.get("name") == Some(name) { 1.0 } else { 0.0 });
    }
    if let Some(classes) = fp.attrs.get("class").filter(|c| !c.trim().is_empty()) {
        let live = el.class_list();
        if !live.is_empty() {
            let stored: Vec<&str> = classes.split_whitespace().collect();
            acc.add(W_CLASS, jaccard(stored, live));
        }
    }
    if let Some(text) = fp.text.as_deref().filter(|t| !t.is_empty()) {
        acc.add(W_TEXT, normalized_levenshtein(&normalize_text(text), &normalize_text(&el.text)));
    }

    let others: Vec<(&String, &String)> = fp
        .attrs
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "id" | "name" | "class" | "style"))
        .collect();
    if !others.is_empty() {
        let hits = others.iter().filter(|(k, v)| el.attrs.get(k.as_str()) == Some(*v)).count();
        acc.add(W_ATTRS, hits as f64 / others.len() as f64);
    }

    if let Some(xpath) = anchor.locators.xpath.as_deref() {
        acc.add(W_XPATH, normalized_levenshtein(xpath, &el.xpath));
    }
    if !fp.ancestors.is_empty() {
        // Compare tags, so a renamed class (section.cart -> section.cart-v2) still counts.
        let live_tags: Vec<&str> = el.ancestors.iter().map(|a| ancestor_tag(a)).collect();
        let hits = fp.ancestors.iter().filter(|a| live_tags.contains(&ancestor_tag(a))).count();
        acc.add(W_ANCESTORS, hits as f64 / fp.ancestors.len() as f64);
    }
    if !fp.neighbors.is_empty() {
        let stored: Vec<String> = fp.neighbors.iter().map(|n| normalize_text(n)).collect();
        let live: Vec<String> = el.neighbors.iter().map(|n| normalize_text(n)).collect();
        acc.add(W_NEIGHBORS, jaccard(stored.iter().map(String::as_str), live.iter().map(String::as_str)));
    }
    if let (Some(stored), Some(live)) = (fp.rect, el.rect) {
        acc.add(W_RECT, rect_score(stored, live));
    }
    acc.value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    const V1: &str = r#"<html><body><main><section class="cart"><p>Order total</p>
        <div class="actions"><button class="btn btn-primary" type="submit">Buy now</button></div>
        <p>Apply coupon</p></section></main></body></html>"#;

    fn anchor_for(html: &str, text: &str) -> Anchor {
        let snap = Snapshot::parse(html);
        let el = snap.elements().iter().find(|e| e.text == text && e.tag == "button").unwrap().clone();
        Anchor::from_element(&el, &snap)
    }

    #[test]
    fn identical_element_scores_one() {
        let anchor = anchor_for(V1, "Buy now");
        let snap = Snapshot::parse(V1);
        let el = snap.elements().iter().find(|e| e.tag == "button" && e.text == "Buy now").unwrap();
        assert!((score(&anchor, el) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn renamed_classes_and_text_still_score_well_on_structure() {
        let anchor = anchor_for(V1, "Buy now");
        let v2 = r#"<html><body><main><section class="cart-v2"><p>Order total</p>
            <div class="wrap"><div class="row"><button class="cta" type="submit">Place order</button></div></div>
            <p>Apply coupon</p></section></main></body></html>"#;
        let snap = Snapshot::parse(v2);
        let el = snap.elements().iter().find(|e| e.tag == "button").unwrap();
        let s = score(&anchor, el);
        assert!(s > 0.5 && s < 0.95, "score {s}");
    }

    #[test]
    fn different_tag_scores_lower_than_same_tag() {
        let anchor = anchor_for(V1, "Buy now");
        let snap = Snapshot::parse(V1);
        let same = snap.elements().iter().find(|e| e.tag == "button" && e.text == "Buy now").unwrap();
        let other = snap.elements().iter().find(|e| e.text == "Apply coupon").unwrap();
        assert!(score(&anchor, same) > score(&anchor, other));
    }

    #[test]
    fn rect_term_is_ignored_when_either_side_lacks_geometry() {
        let anchor = anchor_for(V1, "Buy now");
        let snap = Snapshot::parse(V1);
        let el = snap.elements().iter().find(|e| e.tag == "button" && e.text == "Buy now").unwrap();
        let mut with_box = anchor.clone();
        with_box.fingerprint.rect = Some(Rect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 });
        // The candidate has no rect, so the rect feature drops out rather than counting as zero.
        assert!((score(&with_box, el) - 1.0).abs() < 1e-9);
    }
}
