//! Validation and sanitising for the patch DSL. Every stored or applied patch
//! passes through `normalize`. The rules are security rules: patches must not be
//! able to run script, load third-party code, or send data off-site.

use scraper::Selector;

/// JSON Schema for one DSL operation. The AUTHOR job's output is checked against it.
pub fn op_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(crate::patch::Op)).unwrap_or(serde_json::Value::Null)
}

use crate::error::{PwError, Result};
use crate::patch::{Assertion, Op, Patch};

pub const MAX_OPS: usize = 20;
pub const MAX_VALUE_CHARS: usize = 4000;
pub const MAX_SHORT_CHARS: usize = 200;

pub const STYLE_PROPS: &[&str] = &[
    "background-color", "color", "display", "visibility", "font-size", "font-weight",
    "width", "height", "padding", "margin", "border", "border-radius", "opacity",
    "text-align", "cursor",
];

const DENIED_ATTRS: &[&str] = &["style", "srcdoc", "srcset", "content", "http-equiv"];
const URL_ATTRS: &[&str] = &["href", "src", "action", "formaction"];

fn bad(patch: &Patch, reason: impl Into<String>) -> PwError {
    PwError::InvalidPatch { patch: patch.id.clone(), reason: reason.into() }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn valid_attr_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && name.len() <= 32
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn valid_class(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && name.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// A URL the patch may use: a same-site path. No schemes, no protocol-relative
/// `//host`, no backslashes or control characters.
fn check_local_path(patch: &Patch, value: &str) -> Result<()> {
    let ok = value.starts_with('/')
        && !value.starts_with("//")
        && !value.contains('\\')
        && !value.chars().any(|c| c.is_control() || c.is_whitespace());
    if ok && value.len() <= MAX_SHORT_CHARS {
        Ok(())
    } else {
        Err(bad(patch, format!("`{value}` must be a same-site path starting with a single `/`")))
    }
}

fn check_selector(patch: &Patch, css: &str) -> Result<()> {
    Selector::parse(css).map_err(|e| bad(patch, format!("invalid target selector `{css}`: {e:?}")))?;
    let lowered = css.trim().to_ascii_lowercase();
    if matches!(lowered.as_str(), "html" | "head" | "body") {
        return Err(bad(patch, format!("target `{css}` is not allowed")));
    }
    Ok(())
}

fn check_len(patch: &Patch, value: &str, max: usize, what: &str) -> Result<()> {
    if value.chars().count() > max {
        return Err(bad(patch, format!("{what} is longer than {max} characters")));
    }
    Ok(())
}

fn check_style_value(patch: &Patch, value: &str) -> Result<()> {
    let lowered = value.to_ascii_lowercase();
    let forbidden = [";", "{", "}", "<", ">", "url(", "expression(", "javascript:", "@import", "\\"];
    if forbidden.iter().any(|f| lowered.contains(f)) {
        return Err(bad(patch, format!("style value `{value}` contains a forbidden token")));
    }
    Ok(())
}

/// Validate and sanitise in place. Errors name the patch and the offending op.
pub fn normalize(patch: &mut Patch) -> Result<()> {
    if !valid_id(&patch.id) {
        return Err(bad(patch, "id must be 1-64 chars of [a-z0-9-]"));
    }
    if !patch.page.starts_with('/') {
        return Err(bad(patch, "page must be a path starting with `/`"));
    }
    if patch.anchor.locators.is_empty() {
        return Err(bad(patch, "anchor needs at least one locator (css, xpath, testid or role)"));
    }
    if patch.anchor.fingerprint.tag.is_empty() {
        return Err(bad(patch, "anchor fingerprint needs a tag"));
    }
    if let Some(css) = &patch.anchor.locators.css {
        check_selector(patch, css)?;
    }
    if patch.ops.is_empty() {
        return Err(bad(patch, "patch has no operations"));
    }
    if patch.ops.len() > MAX_OPS {
        return Err(bad(patch, format!("more than {MAX_OPS} operations")));
    }

    let mut ops = std::mem::take(&mut patch.ops);
    for (i, op) in ops.iter_mut().enumerate() {
        let here = |reason: String| bad(patch, format!("op {}: {reason}", i + 1));
        match op {
            Op::SetText { value } => check_len(patch, value, MAX_VALUE_CHARS, "text")?,
            Op::SetHtml { value } => {
                check_len(patch, value, MAX_VALUE_CHARS, "html")?;
                *value = ammonia::clean(value);
            }
            Op::SetAttr { name, value } => {
                if !valid_attr_name(name) || name.starts_with("on") || DENIED_ATTRS.contains(&name.as_str()) {
                    return Err(here(format!("attribute `{name}` cannot be set")));
                }
                check_len(patch, value, MAX_SHORT_CHARS, "attribute value")?;
                if URL_ATTRS.contains(&name.as_str()) {
                    check_local_path(patch, value).map_err(|e| here(e.to_string()))?;
                }
            }
            Op::RemoveAttr { name } => {
                if !valid_attr_name(name) || name.starts_with("on") {
                    return Err(here(format!("attribute `{name}` cannot be removed")));
                }
            }
            Op::SetStyle { prop, value } => {
                if !STYLE_PROPS.contains(&prop.as_str()) {
                    return Err(here(format!("style property `{prop}` is not allowed")));
                }
                check_len(patch, value, MAX_SHORT_CHARS, "style value")?;
                check_style_value(patch, value).map_err(|e| here(e.to_string()))?;
            }
            Op::AddClass { name } | Op::RemoveClass { name } => {
                if !valid_class(name) {
                    return Err(here(format!("class `{name}` is not a valid class name")));
                }
            }
            Op::Hide {} | Op::Show {} | Op::OnClickSubmit {} | Op::Disable {} | Op::Enable {} => {}
            Op::MoveBefore { target } | Op::MoveAfter { target } => {
                check_selector(patch, target).map_err(|e| here(e.to_string()))?;
            }
            Op::InsertBefore { target, html } | Op::InsertAfter { target, html } => {
                check_selector(patch, target).map_err(|e| here(e.to_string()))?;
                check_len(patch, html, MAX_VALUE_CHARS, "inserted html")?;
                *html = ammonia::clean(html);
            }
            Op::OnClickNavigate { href } => check_local_path(patch, href).map_err(|e| here(e.to_string()))?,
            Op::OnClickFetch { url } => check_local_path(patch, url).map_err(|e| here(e.to_string()))?,
        }
    }
    patch.ops = ops;

    for assertion in &patch.assertions {
        match assertion {
            Assertion::TextContains(text) => check_len(patch, text, MAX_SHORT_CHARS, "assertion text")?,
            Assertion::ClickNavigatesTo(path) => check_local_path(patch, path)?,
            Assertion::AttrEquals { name, value } => {
                if !valid_attr_name(name) {
                    return Err(bad(patch, format!("assertion attribute `{name}` is not valid")));
                }
                check_len(patch, value, MAX_SHORT_CHARS, "assertion value")?;
            }
            Assertion::Visible(_) | Assertion::Exists(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchor::{Anchor, Fingerprint, Locators};

    #[test]
    fn schema_lists_every_operation_and_no_script() {
        let text = op_schema().to_string();
        for name in ["setText", "setHTML", "moveBefore", "onClickFetch", "hide"] {
            assert!(text.contains(name), "schema is missing {name}");
        }
        assert!(!text.contains("\"script\""), "the escape hatch must not appear in the schema");
    }

    fn base(ops: Vec<Op>) -> Patch {
        Patch {
            id: "p1".into(),
            author: "t".into(),
            page: "/cart".into(),
            anchor: Anchor {
                locators: Locators { css: Some("#buy".into()), ..Default::default() },
                fingerprint: Fingerprint { tag: "button".into(), ..Default::default() },
            },
            ops,
            assertions: vec![],
        }
    }

    fn check(ops: Vec<Op>) -> Result<Patch> {
        let mut p = base(ops);
        normalize(&mut p).map(|_| p)
    }

    #[test]
    fn allowed_ops_pass() {
        let p = check(vec![
            Op::SetText { value: "Place order".into() },
            Op::SetStyle { prop: "background-color".into(), value: "#0a7d32".into() },
            Op::SetAttr { name: "formaction".into(), value: "/api/v2/checkout".into() },
            Op::AddClass { name: "highlight".into() },
            Op::Hide {},
        ]);
        assert!(p.is_ok(), "{p:?}");
    }

    #[test]
    fn event_handler_attributes_are_refused() {
        assert!(check(vec![Op::SetAttr { name: "onclick".into(), value: "x()".into() }]).is_err());
    }

    #[test]
    fn style_attribute_is_refused_in_favour_of_setstyle() {
        assert!(check(vec![Op::SetAttr { name: "style".into(), value: "color:red".into() }]).is_err());
    }

    #[test]
    fn external_and_scheme_urls_are_refused() {
        for bad_url in ["https://evil.example/steal", "//evil.example/x", "javascript:alert(1)", "/ok\\evil"] {
            assert!(check(vec![Op::OnClickNavigate { href: bad_url.into() }]).is_err(), "{bad_url}");
            assert!(check(vec![Op::OnClickFetch { url: bad_url.into() }]).is_err(), "{bad_url}");
        }
    }

    #[test]
    fn same_site_path_is_accepted_for_fetch() {
        assert!(check(vec![Op::OnClickFetch { url: "/api/v2/checkout".into() }]).is_ok());
    }

    #[test]
    fn style_injection_is_refused() {
        assert!(check(vec![Op::SetStyle { prop: "background-color".into(), value: "red; position:fixed".into() }]).is_err());
        assert!(check(vec![Op::SetStyle { prop: "background-image".into(), value: "red".into() }]).is_err());
        assert!(check(vec![Op::SetStyle { prop: "color".into(), value: "url(http://x)".into() }]).is_err());
    }

    #[test]
    fn inserted_html_is_sanitised_not_trusted() {
        let p = check(vec![Op::SetHtml { value: "<p>hi</p><script>alert(1)</script><a href=\"javascript:alert(2)\">x</a>".into() }]).unwrap();
        match &p.ops[0] {
            Op::SetHtml { value } => {
                assert!(value.contains("<p>hi</p>"));
                assert!(!value.to_lowercase().contains("<script"));
                assert!(!value.to_lowercase().contains("javascript:"));
            }
            other => panic!("unexpected op {other:?}"),
        }
    }

    #[test]
    fn root_targets_are_refused() {
        assert!(check(vec![Op::MoveBefore { target: "body".into() }]).is_err());
        assert!(check(vec![Op::InsertAfter { target: "html".into(), html: "<p>x</p>".into() }]).is_err());
    }

    #[test]
    fn invalid_selectors_are_refused() {
        assert!(check(vec![Op::MoveAfter { target: "div[[[".into() }]).is_err());
    }

    #[test]
    fn too_many_ops_are_refused() {
        let ops = (0..=MAX_OPS).map(|_| Op::Show {}).collect();
        assert!(check(ops).is_err());
    }

    #[test]
    fn anchor_without_locators_is_refused() {
        let mut p = base(vec![Op::Show {}]);
        p.anchor.locators = Locators::default();
        assert!(normalize(&mut p).is_err());
    }

    #[test]
    fn bad_patch_id_is_refused() {
        let mut p = base(vec![Op::Show {}]);
        p.id = "Has Spaces".into();
        assert!(normalize(&mut p).is_err());
    }
}
