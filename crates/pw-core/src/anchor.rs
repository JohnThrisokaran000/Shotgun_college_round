//! Anchors say how a patch finds its element again: several locators (so one
//! site update rarely breaks them all) plus a fingerprint used for scoring.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::snapshot::{normalize_ws, Element, Rect, Snapshot};

/// Attributes kept in a fingerprint. Everything else is noise for matching.
const KEPT_ATTRS: &[&str] = &[
    "id", "name", "class", "type", "href", "role", "aria-label", "data-testid",
    "placeholder", "value", "alt", "title", "for", "action", "formaction", "method",
];
const TEXT_LIMIT: usize = 200;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RoleLocator {
    pub role: String,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Locators {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub css: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xpath: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub testid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<RoleLocator>,
}

impl Locators {
    pub fn is_empty(&self) -> bool {
        self.css.is_none() && self.xpath.is_none() && self.testid.is_none() && self.role.is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Fingerprint {
    pub tag: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestors: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neighbors: Vec<String>,
    #[serde(default, rename = "box", skip_serializing_if = "Option::is_none")]
    pub rect: Option<Rect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Anchor {
    pub locators: Locators,
    pub fingerprint: Fingerprint,
}

impl Fingerprint {
    pub fn from_element(el: &Element) -> Self {
        let attrs = el
            .attrs
            .iter()
            .filter(|(k, _)| KEPT_ATTRS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Fingerprint {
            tag: el.tag.clone(),
            text: Some(el.text.clone()).filter(|t| !t.is_empty() && t.chars().count() <= TEXT_LIMIT),
            attrs,
            ancestors: el.ancestors.clone(),
            neighbors: el.neighbors.clone(),
            rect: el.rect,
            crop: None,
        }
    }
}

impl Anchor {
    /// Build an anchor for `el`, choosing the shortest CSS locator that is unique.
    pub fn from_element(el: &Element, snap: &Snapshot) -> Self {
        let role = match (&el.role, &el.name) {
            (Some(role), Some(name)) => Some(RoleLocator { role: role.clone(), name: name.clone() }),
            _ => None,
        };
        Anchor {
            locators: Locators {
                css: unique_css(el, snap),
                xpath: Some(el.xpath.clone()),
                testid: el.attrs.get("data-testid").cloned(),
                role,
            },
            fingerprint: Fingerprint::from_element(el),
        }
    }
}

fn is_css_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Turn `/html/body/main/section[2]/button` into `html > body > main > section:nth-of-type(2) > button`.
/// Each step names a unique child of the previous step, so the selector matches one element.
fn css_from_xpath(xpath: &str) -> String {
    xpath
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|seg| match seg.find('[') {
            Some(i) => format!("{}:nth-of-type({})", &seg[..i], &seg[i + 1..seg.len() - 1]),
            None => seg.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" > ")
}

fn unique_css(el: &Element, snap: &Snapshot) -> Option<String> {
    let mut candidates = Vec::new();
    if let Some(id) = el.id().filter(|id| is_css_ident(id)) {
        candidates.push(format!("#{id}"));
    }
    let classes = el.class_list();
    if !classes.is_empty() && classes.iter().all(|c| is_css_ident(c)) {
        candidates.push(format!("{}.{}", el.tag, classes.join(".")));
    }
    if let Some(testid) = el.attrs.get("data-testid") {
        candidates.push(format!("{}[data-testid=\"{}\"]", el.tag, testid.replace('"', "\\\"")));
    }
    candidates.push(css_from_xpath(&el.xpath));
    candidates.into_iter().find(|css| matches!(snap.css(css), Ok(hits) if hits == [el.index]))
}

pub fn normalize_text(s: &str) -> String {
    normalize_ws(s).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<html><body><section class="cart"><div><button id="buy" class="btn btn-primary" type="submit">Buy now</button></div><button class="ghost">Cancel</button></section></body></html>"#;

    #[test]
    fn prefers_unique_id_selector() {
        let snap = Snapshot::parse(PAGE);
        let buy = snap.elements().iter().find(|e| e.tag == "button" && e.text == "Buy now").unwrap();
        let anchor = Anchor::from_element(buy, &snap);
        assert_eq!(anchor.locators.css.as_deref(), Some("#buy"));
        assert_eq!(anchor.locators.role.as_ref().unwrap().name, "Buy now");
    }

    #[test]
    fn falls_back_to_structural_css_when_classes_are_shared() {
        let snap = Snapshot::parse(PAGE);
        let cancel = snap.elements().iter().find(|e| e.text == "Cancel").unwrap();
        let anchor = Anchor::from_element(cancel, &snap);
        let css = anchor.locators.css.clone().unwrap();
        assert_eq!(snap.css(&css).unwrap(), vec![cancel.index]);
    }

    #[test]
    fn css_from_xpath_is_a_unique_path() {
        assert_eq!(css_from_xpath("/html/body/main/section[2]/button"), "html > body > main > section:nth-of-type(2) > button");
    }

    #[test]
    fn fingerprint_keeps_only_matching_relevant_attributes() {
        let snap = Snapshot::parse(PAGE);
        let buy = snap.elements().iter().find(|e| e.tag == "button" && e.text == "Buy now").unwrap();
        let fp = Fingerprint::from_element(buy);
        assert_eq!(fp.attrs.get("type").map(String::as_str), Some("submit"));
        assert!(!fp.attrs.contains_key("style"));
    }
}
