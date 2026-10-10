//! The single element serialiser. The training data generator and the CLI prompt
//! both call these functions, so the model never sees a different description of
//! an element at run time than it saw during training.

use crate::anchor::Fingerprint;
use crate::snapshot::{accessible_name, implicit_role, normalize_ws, truncate_chars, Element, Rect};

pub const NAME_CHARS: usize = 40;
pub const CLASS_CHARS: usize = 60;
pub const ANCESTORS_SHOWN: usize = 2;
pub const NEIGHBORS_SHOWN: usize = 2;

struct Fields<'a> {
    label: &'a str,
    tag: &'a str,
    role: Option<String>,
    name: Option<String>,
    id: Option<String>,
    classes: Option<String>,
    ancestors: &'a [String],
    neighbors: &'a [String],
    rect: Option<Rect>,
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "'"))
}

fn render(f: Fields) -> String {
    let mut parts = vec![format!("{} {}", f.label, f.tag)];
    if let Some(role) = f.role {
        parts.push(format!("role={role}"));
    }
    if let Some(name) = f.name {
        parts.push(format!("name={}", quote(&name)));
    }
    if let Some(id) = f.id {
        parts.push(format!("id={id}"));
    }
    if let Some(classes) = f.classes {
        parts.push(format!("class={classes}"));
    }
    if !f.ancestors.is_empty() {
        let shown: Vec<&str> = f.ancestors.iter().take(ANCESTORS_SHOWN).map(String::as_str).collect();
        parts.push(format!("in: {}", shown.join(" > ")));
    }
    if !f.neighbors.is_empty() {
        let shown: Vec<String> = f.neighbors.iter().take(NEIGHBORS_SHOWN).map(|n| quote(n)).collect();
        parts.push(format!("near: {}", shown.join(", ")));
    }
    if let Some(r) = f.rect {
        parts.push(format!(
            "box: {},{},{},{}",
            r.x.round() as i64,
            r.y.round() as i64,
            r.w.round() as i64,
            r.h.round() as i64
        ));
    }
    parts.join(" | ")
}

fn class_text(raw: Option<&str>) -> Option<String> {
    raw.map(|c| truncate_chars(&normalize_ws(c), CLASS_CHARS)).filter(|c| !c.is_empty())
}

/// Like `element_line`, but without the box. Prompts use this so training and serving match.
pub fn element_line_no_box(el: &Element, label: &str) -> String {
    render(Fields {
        label,
        tag: &el.tag,
        role: el.role.clone(),
        name: el.name.as_deref().map(|n| truncate_chars(n, NAME_CHARS)),
        id: el.id().map(str::to_string),
        classes: class_text(el.attrs.get("class").map(String::as_str)),
        ancestors: &el.ancestors,
        neighbors: &el.neighbors,
        rect: None,
    })
}

/// One line per element, e.g.
/// `[3] button | role=button | name="Buy now" | class=btn btn-primary | in: form#checkout > section.cart | near: "Order total", "Apply coupon" | box: 812,604,160,44`.
pub fn element_line(el: &Element, label: &str) -> String {
    render(Fields {
        label,
        tag: &el.tag,
        role: el.role.clone(),
        name: el.name.as_deref().map(|n| truncate_chars(n, NAME_CHARS)),
        id: el.id().map(str::to_string),
        classes: class_text(el.attrs.get("class").map(String::as_str)),
        ancestors: &el.ancestors,
        neighbors: &el.neighbors,
        rect: el.rect,
    })
}

/// The same format for a stored fingerprint (the old element a patch remembers).
pub fn fingerprint_line(fp: &Fingerprint, label: &str) -> String {
    let role = implicit_role(&fp.tag, &fp.attrs);
    let name = accessible_name(&fp.tag, &fp.attrs, fp.text.as_deref().unwrap_or(""));
    render(Fields {
        label,
        tag: &fp.tag,
        role,
        name: name.map(|n| truncate_chars(&n, NAME_CHARS)),
        id: fp.attrs.get("id").filter(|s| !s.is_empty()).cloned(),
        classes: class_text(fp.attrs.get("class").map(String::as_str)),
        ancestors: &fp.ancestors,
        neighbors: &fp.neighbors,
        rect: None,
    })
}

/// Fingerprint line for prompts: same layout, never includes the box (see `element_line_no_box`).
pub fn fingerprint_line_no_box(fp: &Fingerprint, label: &str) -> String {
    fingerprint_line(fp, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    #[test]
    fn matches_the_documented_example_shape() {
        let html = r#"<html><body><section class="cart"><p>Order total</p><form id="checkout"><button class="btn btn-primary">Buy now</button></form><p>Apply coupon</p></section></body></html>"#;
        let snap = Snapshot::parse(html).with_rects(&[None, None, None, None, None, None, Some(Rect { x: 812.4, y: 604.0, w: 160.0, h: 44.0 }), None]);
        let button = snap.elements().iter().find(|e| e.tag == "button").unwrap();
        let line = element_line(button, "[6]");
        assert_eq!(
            line,
            "[6] button | role=button | name=\"Buy now\" | class=btn btn-primary | in: form#checkout > section.cart | near: \"Order total\", \"Apply coupon\" | box: 812,604,160,44"
        );
    }

    #[test]
    fn fingerprint_line_uses_the_same_layout() {
        let fp = Fingerprint {
            tag: "button".into(),
            text: Some("Buy now".into()),
            attrs: [("class".to_string(), "btn btn-primary".to_string())].into_iter().collect(),
            ancestors: vec!["form#checkout".into()],
            neighbors: vec!["Order total".into()],
            rect: None,
            ..Default::default()
        };
        assert_eq!(
            fingerprint_line(&fp, "[old]"),
            "[old] button | role=button | name=\"Buy now\" | class=btn btn-primary | in: form#checkout | near: \"Order total\""
        );
    }

    #[test]
    fn quotes_inside_names_do_not_break_the_line() {
        let fp = Fingerprint { tag: "a".into(), text: Some("Say \"hi\"".into()), attrs: [("href".to_string(), "/x".to_string())].into_iter().collect(), ..Default::default() };
        assert!(fingerprint_line(&fp, "[old]").contains("name=\"Say 'hi'\""));
    }
}
