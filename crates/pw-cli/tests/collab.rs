//! Collaboration across two branches, end to end through the `pw` binary. An approval carries over a merge when the
//! content is the same. It does not when the content differs, a rejection is reported rather than carried, and a
//! branch log that does not verify is refused.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pw_core::anchor::Anchor;
use pw_core::patch::{Op, Patch, Status};
use pw_core::snapshot::Snapshot;

const SITE: &str = r#"<!doctype html><html><body><main><section class="cart"><p>Order total</p>
<button id="buy" class="btn" type="submit">Buy now</button><p>Apply coupon</p></section></main></body></html>"#;

fn pw(project: &Path, actor: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pw"))
        .arg("--project")
        .arg(project)
        .arg("--actor")
        .arg(actor)
        .args(args)
        .output()
        .expect("running the pw binary")
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// The patch that sets the sample page's button text to `value`.
fn patch(value: &str) -> Patch {
    let snap = Snapshot::parse(SITE);
    let el = snap.elements().iter().find(|e| e.tag == "button").expect("button in the sample page");
    Patch {
        id: "buy-text".into(),
        author: "dev-a".into(),
        page: "/".into(),
        anchor: Anchor::from_element(el, &snap),
        ops: vec![Op::SetText { value: value.into() }],
        assertions: vec![],
        status: Status::Draft,
        reviewer: None,
    }
}

/// One branch: a project with the sample page and one patch, as a developer's copy would hold them.
struct Branch {
    root: PathBuf,
}

impl Branch {
    fn new(name: &str, value: &str) -> Branch {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("collab-{name}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("index.html"), SITE).unwrap();
        let out = pw(&root, "setup", &["init", "--name", "demo", "--upstream", "http://127.0.0.1:1"]);
        assert!(out.status.success(), "{}", text(&out));
        fs::write(root.join("patches/buy-text.yaml"), patch(value).to_yaml().unwrap()).unwrap();
        Branch { root }
    }

    fn run(&self, actor: &str, args: &[&str]) -> Output {
        pw(&self.root, actor, args)
    }

    fn snap(&self) -> String {
        format!("/={}", self.root.join("index.html").display())
    }

    fn patches(&self) -> PathBuf {
        self.root.join("patches")
    }

    fn audit(&self) -> PathBuf {
        self.root.join("audit")
    }

    /// The audit log's text. Empty when nothing has been written yet.
    fn events(&self) -> String {
        fs::read_to_string(self.audit().join("events.jsonl")).unwrap_or_default()
    }
}

/// Merge `other` into `me` with its audit folder, and write the result.
fn merge_in(me: &Branch, other: &Branch) -> Output {
    let with = other.patches();
    let audit = other.audit();
    let snap = me.snap();
    me.run(
        "ops",
        &["merge", "--with", with.to_str().unwrap(), "--with-audit", audit.to_str().unwrap(), "--snapshot", &snap, "--write"],
    )
}

#[test]
fn an_approval_carries_over_a_merge_when_the_content_is_the_same() {
    let me = Branch::new("carry-me", "Checkout now");
    let other = Branch::new("carry-other", "Checkout now");
    assert!(other.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(other.run("dev-b", &["review", "approve", "buy-text"]).status.success());

    let out = merge_in(&me, &other);
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("approval carried over: `buy-text` approved by dev-b"), "{}", text(&out));

    let list = text(&me.run("ops", &["review", "list"]));
    assert!(list.contains("approved"), "{list}");
    let dry = text(&me.run("ops", &["release", "--channel", "production", "--dry-run", "--snapshot", &me.snap()]));
    assert!(!dry.contains("not approved for production"), "{dry}");
    assert!(me.events().contains("carried_from"), "the carried approval records where it came from");
}

#[test]
fn a_patch_whose_content_changed_on_the_other_branch_does_not_carry_its_approval() {
    let me = Branch::new("changed-me", "Checkout now");
    let other = Branch::new("changed-other", "Checkout now");
    assert!(other.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(other.run("dev-b", &["review", "approve", "buy-text"]).status.success());
    // The other branch edits the approved text by hand, so the approval is for content that no longer exists.
    let file = other.patches().join("buy-text.yaml");
    fs::write(&file, fs::read_to_string(&file).unwrap().replace("Checkout now", "Pay now")).unwrap();

    let out = merge_in(&me, &other);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("CONFLICT same id"), "{}", text(&out));
    assert!(!me.events().contains("carried_from"), "nothing is carried from a conflicting merge");
}

#[test]
fn a_rejection_on_the_other_branch_is_reported_and_not_carried() {
    let me = Branch::new("reject-me", "Checkout now");
    let other = Branch::new("reject-other", "Checkout now");
    assert!(other.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(other
        .run("dev-b", &["review", "reject", "buy-text", "--reason", "not the wording we use"])
        .status
        .success());

    let out = merge_in(&me, &other);
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("rejected on the other branch; not carried over"), "{}", text(&out));
    let list = text(&me.run("ops", &["review", "list"]));
    assert!(!list.contains("approved"), "{list}");
}

#[test]
fn a_branch_log_that_does_not_verify_is_refused_and_nothing_is_carried() {
    let me = Branch::new("tamper-me", "Checkout now");
    let other = Branch::new("tamper-other", "Checkout now");
    assert!(other.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(other.run("dev-b", &["review", "approve", "buy-text"]).status.success());
    // Rewrite who approved it. The chain no longer matches the events.
    let log = other.events().replace("dev-b", "dev-x");
    fs::write(other.audit().join("events.jsonl"), log).unwrap();

    let out = merge_in(&me, &other);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("does not verify"), "{}", text(&out));
    assert!(!me.events().contains("carried_from"));
}

#[test]
fn a_review_that_disagrees_across_branches_is_a_conflict_and_nothing_is_written() {
    let me = Branch::new("clash-me", "Checkout now");
    let other = Branch::new("clash-other", "Checkout now");
    assert!(me.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(me.run("dev-b", &["review", "approve", "buy-text"]).status.success());
    assert!(other.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(other.run("dev-c", &["review", "reject", "buy-text", "--reason", "wording"]).status.success());

    let out = merge_in(&me, &other);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        text(&out).contains("CONFLICT review: `buy-text` is approved here and rejected on the other branch"),
        "{}",
        text(&out)
    );
    assert!(!me.events().contains("carried_from"));
}
