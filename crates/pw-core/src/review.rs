//! Review of patches. The patch file says what state a patch is in. The audit log says who put it there,
//! and for which content. A release trusts an approval only when the two agree.
//!
//! A status edited by hand, without a matching approval in the log, does not count. Neither does an
//! approval whose patch has changed since it was given.

use serde_json::Value;

use crate::audit::Event;
use crate::error::Result;
use crate::patch::{Patch, Status};

pub const PROPOSE: &str = "patch.propose";
pub const APPROVE: &str = "patch.approve";
pub const REJECT: &str = "patch.reject";

pub fn is_review_event(kind: &str) -> bool {
    matches!(kind, PROPOSE | APPROVE | REJECT)
}

/// The most recent review event for `patch_id`, if the log has one.
pub fn latest<'a>(events: &'a [Event], patch_id: &str) -> Option<&'a Event> {
    events
        .iter()
        .rev()
        .find(|e| is_review_event(&e.kind) && e.data.get("patch").and_then(Value::as_str) == Some(patch_id))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Approval {
    /// The file says approved, and the latest review event approves this exact content.
    Approved,
    /// The file says approved, but the log does not back it. The reason says why.
    Unproven(String),
    /// The patch is a draft, proposed, or rejected.
    NotApproved(Status),
}

/// Whether `patch` may be released, judged against the audit log's review events.
pub fn approval(patch: &Patch, events: &[Event]) -> Result<Approval> {
    if patch.status != Status::Approved {
        return Ok(Approval::NotApproved(patch.status));
    }
    let digest = patch.review_digest()?;
    let reason = match latest(events, &patch.id) {
        Some(e) if e.kind == APPROVE && e.data.get("digest").and_then(Value::as_str) == Some(digest.as_str()) => {
            return Ok(Approval::Approved);
        }
        Some(e) if e.kind == APPROVE => "the patch changed after it was approved; propose and approve it again".to_string(),
        Some(e) => format!("the latest review event is `{}`, not an approval", e.kind),
        None => "this audit log has no approval for it".to_string(),
    };
    Ok(Approval::Unproven(reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PATCH: &str = r##"
id: checkout-button-text
author: dev-a
page: /
anchor:
  locators:
    css: "#buy"
  fingerprint:
    tag: button
ops:
  - { op: setText, value: "Checkout now" }
"##;

    fn event(seq: u64, kind: &str, data: Value) -> Event {
        Event { seq, ts_ms: seq, actor: "dev-b".into(), kind: kind.into(), data, prev: String::new(), hash: String::new() }
    }

    #[test]
    fn a_draft_is_not_approved_whatever_the_log_says() {
        let patch = Patch::from_yaml(PATCH).unwrap();
        assert_eq!(approval(&patch, &[]).unwrap(), Approval::NotApproved(Status::Draft));
    }

    #[test]
    fn an_approved_status_needs_a_matching_approval_in_the_log() {
        let mut patch = Patch::from_yaml(PATCH).unwrap();
        patch.status = Status::Approved;
        assert!(matches!(approval(&patch, &[]).unwrap(), Approval::Unproven(r) if r.contains("no approval")));

        let digest = patch.review_digest().unwrap();
        let log = vec![event(1, APPROVE, json!({"patch": "checkout-button-text", "digest": digest, "reviewer": "dev-b"}))];
        assert_eq!(approval(&patch, &log).unwrap(), Approval::Approved);
    }

    #[test]
    fn editing_an_approved_patch_stops_the_approval_counting() {
        let mut patch = Patch::from_yaml(PATCH).unwrap();
        patch.status = Status::Approved;
        let old = patch.review_digest().unwrap();
        let log = vec![event(1, APPROVE, json!({"patch": "checkout-button-text", "digest": old}))];
        patch.ops = Patch::from_yaml(&PATCH.replace("Checkout now", "Pay now")).unwrap().ops;
        assert_ne!(patch.review_digest().unwrap(), old);
        match approval(&patch, &log).unwrap() {
            Approval::Unproven(reason) => assert!(reason.contains("changed after"), "{reason}"),
            other => panic!("expected Unproven, got {other:?}"),
        }
    }

    #[test]
    fn a_later_rejection_overrides_an_earlier_approval() {
        let mut patch = Patch::from_yaml(PATCH).unwrap();
        patch.status = Status::Approved;
        let digest = patch.review_digest().unwrap();
        let log = vec![
            event(1, APPROVE, json!({"patch": "checkout-button-text", "digest": digest})),
            event(2, REJECT, json!({"patch": "checkout-button-text", "digest": digest, "reason": "no"})),
        ];
        assert!(matches!(approval(&patch, &log).unwrap(), Approval::Unproven(r) if r.contains("patch.reject")));
    }

    #[test]
    fn review_events_for_other_patches_are_ignored() {
        let mut patch = Patch::from_yaml(PATCH).unwrap();
        patch.status = Status::Approved;
        let digest = patch.review_digest().unwrap();
        let log = vec![event(1, APPROVE, json!({"patch": "some-other-patch", "digest": digest}))];
        assert!(matches!(approval(&patch, &log).unwrap(), Approval::Unproven(_)));
    }
}
