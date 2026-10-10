//! Parsed DOM snapshot.
//!
//! Contract shared with the browser side: element `index` is the position of
//! the element in document order (pre-order over every element, starting at
//! `<html>`). The browser extractor and this parser must agree on that order,
//! and rectangles passed to [`Snapshot::with_rects`] use the same indices.

use std::collections::{BTreeMap, HashMap};

use scraper::{ElementRef, Html, Selector};
use serde::{Deserialize, Serialize};

use crate::error::{PwError, Result};

const TEXT_LIMIT: usize = 200;
const NAME_LIMIT: usize = 80;
const NEIGHBOR_LIMIT: usize = 60;
const NEIGHBORS_PER_SIDE: usize = 2;
const ANCESTOR_LIMIT: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub index: u32,
    pub tag: String,
    pub attrs: BTreeMap<String, String>,
    pub role: Option<String>,
    pub name: Option<String>,
    pub text: String,
    pub xpath: String,
    pub parent: Option<u32>,
    /// Nearest ancestor first, each as `tag#id` or `tag.class` (or bare tag).
    pub ancestors: Vec<String>,
    /// Text of nearby elements (same parent first), in document order.
    pub neighbors: Vec<String>,
    pub rect: Option<Rect>,
}

impl Element {
    pub fn id(&self) -> Option<&str> {
        self.attrs.get("id").map(String::as_str).filter(|s| !s.is_empty())
    }

    pub fn class_list(&self) -> Vec<&str> {
        match self.attrs.get("class") {
            Some(c) => c.split_whitespace().collect(),
            None => Vec::new(),
        }
    }
}

/// A parsed page. Keeps the HTML tree so CSS locators can be evaluated.
pub struct Snapshot {
    html: Html,
    elements: Vec<Element>,
}

struct Raw<'a> {
    el: ElementRef<'a>,
    parent: Option<usize>,
    children: Vec<usize>,
}

/// Pre-order (document order) traversal over element nodes.
fn collect(root: ElementRef<'_>) -> Vec<Raw<'_>> {
    let mut raws: Vec<Raw> = Vec::new();
    let mut stack: Vec<(ElementRef, Option<usize>)> = vec![(root, None)];
    while let Some((el, parent)) = stack.pop() {
        let idx = raws.len();
        if let Some(p) = parent {
            raws[p].children.push(idx);
        }
        raws.push(Raw { el, parent, children: Vec::new() });
        let kids: Vec<ElementRef> = el.children().filter_map(ElementRef::wrap).collect();
        for kid in kids.into_iter().rev() {
            stack.push((kid, Some(idx)));
        }
    }
    raws
}

pub fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn truncate_chars(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        s.to_string()
    } else {
        s.chars().take(limit).collect()
    }
}

/// Implicit ARIA role for the elements this project reasons about.
pub fn implicit_role(tag: &str, attrs: &BTreeMap<String, String>) -> Option<String> {
    if let Some(explicit) = attrs.get("role").and_then(|r| r.split_whitespace().next()) {
        return Some(explicit.to_string());
    }
    let input_type = attrs.get("type").map(|t| t.to_ascii_lowercase()).unwrap_or_default();
    let role = match tag {
        "a" if attrs.contains_key("href") => "link",
        "button" => "button",
        "input" => match input_type.as_str() {
            "button" | "submit" | "reset" | "image" => "button",
            "checkbox" => "checkbox",
            "radio" => "radio",
            "range" => "slider",
            "search" => "searchbox",
            "hidden" => return None,
            _ => "textbox",
        },
        "textarea" => "textbox",
        "select" => "combobox",
        "img" if attrs.get("alt").is_some_and(|a| !a.is_empty()) => "img",
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => "heading",
        "nav" => "navigation",
        "main" => "main",
        "form" => "form",
        "ul" | "ol" => "list",
        "li" => "listitem",
        "table" => "table",
        "article" => "article",
        _ => return None,
    };
    Some(role.to_string())
}

/// Approximation of the accessible name: aria-label, alt, input value, text, title.
pub fn accessible_name(tag: &str, attrs: &BTreeMap<String, String>, text: &str) -> Option<String> {
    let input_type = attrs.get("type").map(|t| t.to_ascii_lowercase()).unwrap_or_default();
    let value_is_label = tag == "input" && matches!(input_type.as_str(), "button" | "submit" | "reset");
    let candidates = [
        attrs.get("aria-label").cloned(),
        attrs.get("alt").cloned(),
        if value_is_label { attrs.get("value").cloned() } else { None },
        Some(text.to_string()),
        attrs.get("title").cloned(),
    ];
    candidates
        .into_iter()
        .flatten()
        .map(|s| truncate_chars(&normalize_ws(&s), NAME_LIMIT))
        .find(|s| !s.is_empty())
}

fn ancestor_label(tag: &str, attrs: &BTreeMap<String, String>) -> String {
    if let Some(id) = attrs.get("id").filter(|s| !s.is_empty()) {
        return format!("{tag}#{id}");
    }
    if let Some(first) = attrs.get("class").and_then(|c| c.split_whitespace().next()) {
        return format!("{tag}.{first}");
    }
    tag.to_string()
}

impl Snapshot {
    pub fn parse(html: &str) -> Snapshot {
        let doc = Html::parse_document(html);
        let elements = build_elements(&doc);
        Snapshot { html: doc, elements }
    }

    /// Attach browser-measured rectangles, indexed like the elements.
    pub fn with_rects(mut self, rects: &[Option<Rect>]) -> Self {
        for (el, rect) in self.elements.iter_mut().zip(rects.iter()) {
            el.rect = *rect;
        }
        self
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn element(&self, index: u32) -> Option<&Element> {
        self.elements.get(index as usize)
    }

    /// Indices of elements matched by a CSS selector, in document order.
    pub fn css(&self, selector: &str) -> Result<Vec<u32>> {
        let sel = Selector::parse(selector).map_err(|e| PwError::Selector {
            selector: selector.to_string(),
            reason: format!("{e:?}"),
        })?;
        let hits = collect(self.html.root_element())
            .iter()
            .enumerate()
            .filter(|(_, raw)| sel.matches(&raw.el))
            .map(|(i, _)| i as u32)
            .collect();
        Ok(hits)
    }

    pub fn by_xpath(&self, xpath: &str) -> Vec<u32> {
        self.elements.iter().filter(|e| e.xpath == xpath).map(|e| e.index).collect()
    }

    pub fn by_testid(&self, testid: &str) -> Vec<u32> {
        self.elements
            .iter()
            .filter(|e| e.attrs.get("data-testid").map(String::as_str) == Some(testid))
            .map(|e| e.index)
            .collect()
    }

    pub fn by_role_name(&self, role: &str, name: &str) -> Vec<u32> {
        self.elements
            .iter()
            .filter(|e| e.role.as_deref() == Some(role) && e.name.as_deref() == Some(name))
            .map(|e| e.index)
            .collect()
    }

    /// True when `ancestor` is a strict ancestor of `node`.
    pub fn is_ancestor(&self, ancestor: u32, node: u32) -> bool {
        let mut cur = self.element(node).and_then(|e| e.parent);
        while let Some(p) = cur {
            if p == ancestor {
                return true;
            }
            cur = self.element(p).and_then(|e| e.parent);
        }
        false
    }
}

fn build_elements(doc: &Html) -> Vec<Element> {
    let raws = collect(doc.root_element());
    let tags: Vec<String> = raws.iter().map(|r| r.el.value().name().to_string()).collect();

    // XPath segments: `tag[k]` only when several same-tag siblings exist.
    let mut seg = vec![String::new(); raws.len()];
    if let Some(root_tag) = tags.first() {
        seg[0] = root_tag.clone();
    }
    for raw in &raws {
        let mut totals: HashMap<&str, usize> = HashMap::new();
        for &c in &raw.children {
            *totals.entry(tags[c].as_str()).or_default() += 1;
        }
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for &c in &raw.children {
            let t = tags[c].as_str();
            let k = {
                let e = seen.entry(t).or_default();
                *e += 1;
                *e
            };
            seg[c] = if totals[t] > 1 { format!("{t}[{k}]") } else { t.to_string() };
        }
    }
    let mut xpaths = vec![String::new(); raws.len()];
    for i in 0..raws.len() {
        xpaths[i] = match raws[i].parent {
            None => format!("/{}", seg[i]),
            Some(p) => format!("{}/{}", xpaths[p], seg[i]),
        };
    }

    let attrs_of: Vec<BTreeMap<String, String>> = raws
        .iter()
        .map(|r| {
            r.el.value()
                .attrs()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
                .collect()
        })
        .collect();
    let texts: Vec<String> = raws
        .iter()
        .map(|r| truncate_chars(&normalize_ws(&r.el.text().collect::<Vec<_>>().join(" ")), TEXT_LIMIT))
        .collect();

    let mut out = Vec::with_capacity(raws.len());
    for i in 0..raws.len() {
        let raw = &raws[i];
        let mut ancestors = Vec::new();
        let mut cur = raw.parent;
        while let Some(p) = cur {
            if ancestors.len() == ANCESTOR_LIMIT {
                break;
            }
            ancestors.push(ancestor_label(&tags[p], &attrs_of[p]));
            cur = raws[p].parent;
        }
        let neighbors = neighbor_texts(&raws, &texts, i);
        out.push(Element {
            index: i as u32,
            tag: tags[i].clone(),
            role: implicit_role(&tags[i], &attrs_of[i]),
            name: accessible_name(&tags[i], &attrs_of[i], &texts[i]),
            text: texts[i].clone(),
            attrs: attrs_of[i].clone(),
            xpath: xpaths[i].clone(),
            parent: raw.parent.map(|p| p as u32),
            ancestors,
            neighbors,
            rect: None,
        });
    }
    out
}

/// Texts of up to two siblings on each side; falls back to the parent's siblings.
fn neighbor_texts(raws: &[Raw], texts: &[String], i: usize) -> Vec<String> {
    let mut level = Some(i);
    while let Some(node) = level {
        let Some(parent) = raws[node].parent else { break };
        let siblings = &raws[parent].children;
        if let Some(pos) = siblings.iter().position(|&s| s == node) {
            let start = pos.saturating_sub(NEIGHBORS_PER_SIDE);
            let mut found: Vec<usize> = siblings[start..pos].to_vec();
            found.extend(siblings[pos + 1..].iter().take(NEIGHBORS_PER_SIDE).copied());
            let texts_found: Vec<String> = found
                .into_iter()
                .map(|s| truncate_chars(&texts[s], NEIGHBOR_LIMIT))
                .filter(|t| !t.is_empty())
                .collect();
            if !texts_found.is_empty() {
                return texts_found;
            }
        }
        level = Some(parent);
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<!doctype html><html><body><main>
        <section class="cart"><p>Order total</p>
          <form id="checkout"><div class="actions"><button class="btn btn-primary" type="submit">Buy now</button></div></form>
          <p>Apply coupon</p>
        </section></main></body></html>"#;

    #[test]
    fn indexes_follow_document_order() {
        let snap = Snapshot::parse(PAGE);
        let tags: Vec<&str> = snap.elements().iter().map(|e| e.tag.as_str()).collect();
        assert_eq!(tags, ["html", "head", "body", "main", "section", "p", "form", "div", "button", "p"]);
    }

    #[test]
    fn builds_xpath_ancestors_and_neighbors() {
        let snap = Snapshot::parse(PAGE);
        let button = snap.element(8).expect("button");
        assert_eq!(button.xpath, "/html/body/main/section/form/div/button");
        assert_eq!(button.ancestors, ["div.actions", "form#checkout", "section.cart", "main"]);
        assert_eq!(button.neighbors, ["Order total", "Apply coupon"]);
        let first_p = snap.element(5).expect("first p");
        assert_eq!(first_p.neighbors, ["Buy now", "Apply coupon"]);
    }

    #[test]
    fn computes_role_and_name() {
        let snap = Snapshot::parse(PAGE);
        let button = snap.element(8).expect("button");
        assert_eq!(button.role.as_deref(), Some("button"));
        assert_eq!(button.name.as_deref(), Some("Buy now"));
    }

    #[test]
    fn css_matches_in_document_order() {
        let snap = Snapshot::parse(PAGE);
        assert_eq!(snap.css("p").unwrap(), vec![5, 9]);
        assert_eq!(snap.css("section.cart > form#checkout button.btn").unwrap(), vec![8]);
        assert_eq!(snap.css("section.cart > button").unwrap(), Vec::<u32>::new());
        assert_eq!(snap.css("form#checkout .btn-primary").unwrap(), vec![8]);
    }

    #[test]
    fn invalid_selector_is_an_error_not_a_panic() {
        let snap = Snapshot::parse(PAGE);
        assert!(matches!(snap.css("button[[["), Err(PwError::Selector { .. })));
    }

    #[test]
    fn rects_attach_by_index() {
        let mut rects = vec![None; 10];
        rects[8] = Some(Rect { x: 1.0, y: 2.0, w: 3.0, h: 4.0 });
        let snap = Snapshot::parse(PAGE).with_rects(&rects);
        assert_eq!(snap.element(8).unwrap().rect.unwrap().w, 3.0);
        assert!(snap.element(0).unwrap().rect.is_none());
    }
}
