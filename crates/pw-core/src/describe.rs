//! Text descriptions of an anchor and of a candidate element, for the matcher model.
//!
//! The training data (pw-train) and the inference (pw-model) both build their text here, so the model is trained
//! and used on exactly the same text. Changing this function means training the model again.

use std::collections::BTreeMap;

use crate::anchor::{Anchor, Fingerprint};
use crate::snapshot::Element;

const MAX_TEXT_CHARS: usize = 80;
const MAX_PART_CHARS: usize = 40;
const MAX_NEIGHBORS: usize = 4;
const MAX_ANCESTORS: usize = 4;

/// Collapse whitespace and cut to `max` characters.
fn clip(text: &str, max: usize) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(max).collect()
}

fn describe(tag: &str, attrs: &BTreeMap<String, String>, text: &str, ancestors: &[String], neighbors: &[String]) -> String {
    let mut parts = vec![format!("tag {tag}")];
    if let Some(id) = attrs.get("id").filter(|s| !s.is_empty()) {
        parts.push(format!("id {}", clip(id, MAX_PART_CHARS)));
    }
    if let Some(class) = attrs.get("class").filter(|s| !s.trim().is_empty()) {
        parts.push(format!("class {}", clip(class, 2 * MAX_PART_CHARS)));
    }
    let others: Vec<String> = attrs
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "id" | "class" | "style"))
        .map(|(k, v)| format!("{k}={}", clip(v, MAX_PART_CHARS)))
        .collect();
    if !others.is_empty() {
        parts.push(format!("attrs {}", others.join(" ")));
    }
    if !text.trim().is_empty() {
        parts.push(format!("text {}", clip(text, MAX_TEXT_CHARS)));
    }
    if !ancestors.is_empty() {
        let chain: Vec<String> = ancestors.iter().take(MAX_ANCESTORS).map(|a| clip(a, MAX_PART_CHARS)).collect();
        parts.push(format!("in {}", chain.join(" > ")));
    }
    if !neighbors.is_empty() {
        let near: Vec<String> = neighbors.iter().take(MAX_NEIGHBORS).map(|n| clip(n, MAX_PART_CHARS)).collect();
        parts.push(format!("near {}", near.join(" ; ")));
    }
    parts.join(" | ")
}

/// The stored description of what the patch was anchored to.
pub fn anchor_text(anchor: &Anchor) -> String {
    fingerprint_text(&anchor.fingerprint)
}

/// The same description, from the fingerprint alone (a relocation chooser only sees the fingerprint).
pub fn fingerprint_text(fp: &Fingerprint) -> String {
    describe(&fp.tag, &fp.attrs, fp.text.as_deref().unwrap_or(""), &fp.ancestors, &fp.neighbors)
}

/// The description of a live element, in the same format as `anchor_text`. It is built from the fields the stored
/// fingerprint keeps, so an anchor and the element it came from always describe the same things. Describing every
/// attribute here let a generator's gold marker reach the matcher on the candidate side alone (docs/DESIGN.md, 6.7).
pub fn element_text(el: &Element) -> String {
    fingerprint_text(&Fingerprint::from_element(el))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    const PAGE: &str = r#"<html><body><main><section class="cart"><p>Order total</p>
        <button id="buy" class="btn btn-primary" type="submit">Buy now</button><p>Apply coupon</p></section></main></body></html>"#;

    #[test]
    fn an_anchor_and_the_element_it_came_from_describe_the_same_text() {
        let snap = Snapshot::parse(PAGE);
        let el = snap.elements().iter().find(|e| e.tag == "button").unwrap();
        let anchor = Anchor::from_element(el, &snap);
        assert_eq!(anchor_text(&anchor), element_text(el));
        assert!(element_text(el).starts_with("tag button | id buy | class btn btn-primary"), "{}", element_text(el));
    }

    #[test]
    fn long_text_and_lists_are_cut() {
        let long = "word ".repeat(100);
        let text = describe("p", &BTreeMap::new(), &long, &[], &[]);
        assert!(text.len() < 120, "{text}");
        assert_eq!(clip("  a \n  b  ", 10), "a b");
    }
}
