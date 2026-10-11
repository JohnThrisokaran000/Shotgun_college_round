//! Patch model: one YAML file holds an anchor, declared operations and assertions.
//! Parsing always goes through `dsl::normalize`, so a patch that is stored or
//! applied has already been validated and sanitised.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::anchor::Anchor;
use crate::dsl;
use crate::error::{PwError, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    pub id: String,
    #[serde(default)]
    pub author: String,
    pub page: String,
    pub anchor: Anchor,
    pub ops: Vec<Op>,
    #[serde(default, rename = "assert", skip_serializing_if = "Vec::is_empty")]
    pub assertions: Vec<Assertion>,
    /// Review state. A file without a status is a draft, and drafts are not written out.
    #[serde(default, skip_serializing_if = "Status::is_draft")]
    pub status: Status,
    /// Who approved the patch. Set by `pw review approve`, and absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reviewer: Option<String>,
}

/// Review state of a patch. The file records the state. Who set it, and for which content, is in
/// the audit log (see `review`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Draft,
    Proposed,
    Approved,
    Rejected,
}

impl Status {
    pub fn is_draft(&self) -> bool {
        *self == Status::Draft
    }

    pub fn label(self) -> &'static str {
        match self {
            Status::Draft => "draft",
            Status::Proposed => "proposed",
            Status::Approved => "approved",
            Status::Rejected => "rejected",
        }
    }
}

/// The whole DSL. Anything not listed here is rejected at parse time.
/// `script` (free JavaScript) is deliberately absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op")]
pub enum Op {
    #[serde(rename = "setText")]
    SetText { value: String },
    #[serde(rename = "setHTML")]
    SetHtml { value: String },
    #[serde(rename = "setAttr")]
    SetAttr { name: String, value: String },
    #[serde(rename = "removeAttr")]
    RemoveAttr { name: String },
    #[serde(rename = "setStyle")]
    SetStyle { prop: String, value: String },
    #[serde(rename = "addClass")]
    AddClass { name: String },
    #[serde(rename = "removeClass")]
    RemoveClass { name: String },
    #[serde(rename = "hide")]
    Hide {},
    #[serde(rename = "show")]
    Show {},
    #[serde(rename = "moveBefore")]
    MoveBefore { target: String },
    #[serde(rename = "moveAfter")]
    MoveAfter { target: String },
    #[serde(rename = "insertBefore")]
    InsertBefore { target: String, html: String },
    #[serde(rename = "insertAfter")]
    InsertAfter { target: String, html: String },
    #[serde(rename = "onClickNavigate")]
    OnClickNavigate { href: String },
    #[serde(rename = "onClickSubmit")]
    OnClickSubmit {},
    #[serde(rename = "onClickFetch")]
    OnClickFetch { url: String },
    #[serde(rename = "disable")]
    Disable {},
    #[serde(rename = "enable")]
    Enable {},
}

/// Assertions read as single-key maps in YAML and JSON: `- visible: true`,
/// `- clickNavigatesTo: /order/confirm`, `- attrEquals: {name: x, value: y}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "AssertionRepr", into = "AssertionRepr")]
pub enum Assertion {
    Visible(bool),
    Exists(bool),
    TextContains(String),
    ClickNavigatesTo(String),
    /// Click the element, then the page must show this text (for example after a cart update).
    ClickShowsText(String),
    AttrEquals { name: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct AttrPair {
    name: String,
    value: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum AssertionRepr {
    Visible { visible: bool },
    Exists { exists: bool },
    TextContains { #[serde(rename = "textContains")] text_contains: String },
    ClickNavigatesTo { #[serde(rename = "clickNavigatesTo")] click_navigates_to: String },
    ClickShowsText { #[serde(rename = "clickShowsText")] click_shows_text: String },
    AttrEquals { #[serde(rename = "attrEquals")] attr_equals: AttrPair },
}

impl From<AssertionRepr> for Assertion {
    fn from(repr: AssertionRepr) -> Self {
        match repr {
            AssertionRepr::Visible { visible } => Assertion::Visible(visible),
            AssertionRepr::Exists { exists } => Assertion::Exists(exists),
            AssertionRepr::TextContains { text_contains } => Assertion::TextContains(text_contains),
            AssertionRepr::ClickNavigatesTo { click_navigates_to } => Assertion::ClickNavigatesTo(click_navigates_to),
            AssertionRepr::ClickShowsText { click_shows_text } => Assertion::ClickShowsText(click_shows_text),
            AssertionRepr::AttrEquals { attr_equals } => Assertion::AttrEquals { name: attr_equals.name, value: attr_equals.value },
        }
    }
}

impl From<Assertion> for AssertionRepr {
    fn from(a: Assertion) -> Self {
        match a {
            Assertion::Visible(visible) => AssertionRepr::Visible { visible },
            Assertion::Exists(exists) => AssertionRepr::Exists { exists },
            Assertion::TextContains(text_contains) => AssertionRepr::TextContains { text_contains },
            Assertion::ClickNavigatesTo(click_navigates_to) => AssertionRepr::ClickNavigatesTo { click_navigates_to },
            Assertion::ClickShowsText(click_shows_text) => AssertionRepr::ClickShowsText { click_shows_text },
            Assertion::AttrEquals { name, value } => AssertionRepr::AttrEquals { attr_equals: AttrPair { name, value } },
        }
    }
}

impl Patch {
    /// Parse and validate a patch file. The escape hatch gets its own message.
    pub fn from_yaml(text: &str) -> Result<Patch> {
        let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)?;
        reject_escape_hatch(&raw)?;
        let mut patch: Patch = serde_yaml_ng::from_str(text)?;
        dsl::normalize(&mut patch)?;
        Ok(patch)
    }

    pub fn read(path: &Path) -> Result<Patch> {
        let text = std::fs::read_to_string(path)?;
        Patch::from_yaml(&text).map_err(|e| match e {
            PwError::InvalidPatch { reason, .. } => PwError::InvalidPatch { patch: path.display().to_string(), reason },
            other => other,
        })
    }

    pub fn to_yaml(&self) -> Result<String> {
        Ok(serde_yaml_ng::to_string(self)?)
    }

    pub fn sha256(&self) -> Result<String> {
        let text = self.to_yaml()?;
        Ok(hex::encode(Sha256::digest(text.as_bytes())))
    }

    /// Hash of the patch's content with its review state removed. An approval is recorded against this
    /// digest, so changing the patch after it was approved makes the approval stop counting.
    pub fn review_digest(&self) -> Result<String> {
        let mut content = self.clone();
        content.status = Status::Draft;
        content.reviewer = None;
        content.sha256()
    }
}

fn reject_escape_hatch(raw: &serde_yaml_ng::Value) -> Result<()> {
    let Some(ops) = raw.get("ops").and_then(|v| v.as_sequence()) else {
        return Ok(());
    };
    for op in ops {
        if op.get("op").and_then(|v| v.as_str()) == Some("script") {
            return Err(PwError::InvalidPatch {
                patch: raw.get("id").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                reason: "the `script` escape hatch is disabled: patches may only use the declared DSL".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r##"
id: fix-checkout-button
author: dev-a
page: /cart
anchor:
  locators:
    css: "#buy"
    xpath: "/html/body/main/section[2]/div[3]/button"
    role: { role: button, name: "Buy now" }
  fingerprint:
    tag: button
    text: "Buy now"
    attrs: { class: "btn btn-primary", type: submit }
    ancestors: ["form#checkout", "section.cart", "main"]
    neighbors: ["Order total", "Apply coupon"]
    box: { x: 812, y: 604, w: 160, h: 44 }
ops:
  - { op: setAttr, name: formaction, value: "/api/v2/checkout" }
  - { op: setStyle, prop: background-color, value: "#0a7d32" }
assert:
  - { visible: true }
  - { clickNavigatesTo: "/order/confirm" }
"##;

    #[test]
    fn parses_the_documented_example() {
        let p = Patch::from_yaml(GOOD).unwrap();
        assert_eq!(p.id, "fix-checkout-button");
        assert_eq!(p.ops.len(), 2);
        assert_eq!(p.assertions.len(), 2);
        assert_eq!(p.anchor.fingerprint.rect.unwrap().w, 160.0);
    }

    #[test]
    fn round_trips_through_yaml() {
        let p = Patch::from_yaml(GOOD).unwrap();
        let again = Patch::from_yaml(&p.to_yaml().unwrap()).unwrap();
        assert_eq!(p, again);
        assert_eq!(p.sha256().unwrap(), again.sha256().unwrap());
    }

    #[test]
    fn script_escape_hatch_is_rejected_with_a_clear_reason() {
        let text = GOOD.replace("  - { op: setStyle, prop: background-color, value: \"#0a7d32\" }", "  - { op: script, code: \"alert(1)\", approved_by: someone }");
        let err = Patch::from_yaml(&text).unwrap_err();
        assert!(err.to_string().contains("escape hatch is disabled"), "{err}");
    }

    #[test]
    fn click_shows_text_round_trips_as_a_yaml_assertion() {
        let text = GOOD.replace("  - { clickNavigatesTo: \"/order/confirm\" }", "  - { clickShowsText: \"Added to cart\" }");
        let p = Patch::from_yaml(&text).unwrap();
        assert!(p.assertions.contains(&Assertion::ClickShowsText("Added to cart".into())));
        assert!(p.to_yaml().unwrap().contains("clickShowsText"));
        assert_eq!(Patch::from_yaml(&p.to_yaml().unwrap()).unwrap(), p);
    }

    #[test]
    fn unknown_op_is_rejected() {
        let text = GOOD.replace("op: setStyle", "op: teleport");
        assert!(Patch::from_yaml(&text).is_err());
    }

    #[test]
    fn a_draft_is_not_written_out_and_review_state_round_trips() {
        let mut p = Patch::from_yaml(GOOD).unwrap();
        assert_eq!(p.status, Status::Draft);
        assert!(!p.to_yaml().unwrap().contains("status:"), "drafts keep the old file format");
        p.status = Status::Approved;
        p.reviewer = Some("dev-b".into());
        let text = p.to_yaml().unwrap();
        assert!(text.contains("status: approved") && text.contains("reviewer: dev-b"), "{text}");
        assert_eq!(Patch::from_yaml(&text).unwrap(), p);
    }

    #[test]
    fn review_digest_ignores_review_state_but_not_content() {
        let draft = Patch::from_yaml(GOOD).unwrap();
        let mut approved = draft.clone();
        approved.status = Status::Approved;
        approved.reviewer = Some("dev-b".into());
        assert_ne!(draft.sha256().unwrap(), approved.sha256().unwrap(), "the file changes when it is approved");
        assert_eq!(draft.review_digest().unwrap(), approved.review_digest().unwrap());
        let mut edited = approved.clone();
        edited.ops.push(Op::Hide {});
        assert_ne!(edited.review_digest().unwrap(), approved.review_digest().unwrap());
    }
}
