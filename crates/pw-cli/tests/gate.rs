//! The review and release gate, end to end through the `pw` binary. No browser is needed: a verification
//! record is written the same way `pw verify` writes one, from the content the gate compares it with.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pw_core::anchor::Anchor;
use pw_core::patch::{Op, Patch, Status};
use pw_core::release;
use pw_core::snapshot::Snapshot;
use pw_core::verify::{self, Check, Outcome, Record};

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

/// A patch whose anchor the library built from `SITE`, so the snapshot matches it without relocation.
fn patch_for(id: &str, tag: &str, label: &str, value: &str) -> Patch {
    let snap = Snapshot::parse(SITE);
    let el = snap.elements().iter().find(|e| e.tag == tag && e.text == label).expect("element in the sample page");
    Patch {
        id: id.into(),
        author: "dev-a".into(),
        page: "/".into(),
        anchor: Anchor::from_element(el, &snap),
        ops: vec![Op::SetText { value: value.into() }],
        assertions: vec![],
        status: Status::Draft,
        reviewer: None,
    }
}

struct Fixture {
    root: PathBuf,
    snapshot: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Fixture {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("gate-{name}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let snapshot = root.join("index.html");
        fs::write(&snapshot, SITE).unwrap();
        let out = pw(&root, "setup", &["init", "--name", "demo", "--upstream", "http://127.0.0.1:1"]);
        assert!(out.status.success(), "{}", text(&out));
        let patch = patch_for("buy-text", "button", "Buy now", "Checkout now");
        fs::write(root.join("patches/buy-text.yaml"), patch.to_yaml().unwrap()).unwrap();
        Fixture { root, snapshot }
    }

    fn snap_arg(&self) -> String {
        format!("/={}", self.snapshot.display())
    }

    fn run(&self, actor: &str, args: &[&str]) -> Output {
        pw(&self.root, actor, args)
    }

    /// Propose by the author, then approve by someone else.
    fn approve_by(&self, reviewer: &str) {
        assert!(self.run("dev-a", &["review", "propose", "buy-text"]).status.success());
        let out = self.run(reviewer, &["review", "approve", "buy-text"]);
        assert!(out.status.success(), "{}", text(&out));
    }

    fn release_to(&self, channel: &str, extra: &[&str]) -> Output {
        let snap = self.snap_arg();
        let mut args = vec!["release", "--channel", channel, "--snapshot", snap.as_str()];
        args.extend_from_slice(extra);
        self.run("ops", &args)
    }
}

/// The digest `pw verify` would record for the release currently on `channel`.
fn digest_of(root: &Path, channel: &str) -> String {
    let id = release::current(root, channel).unwrap().expect("a release on the channel");
    let manifest = release::read_release(root, &id).unwrap();
    release::patches_digest(&manifest.patches).unwrap()
}

fn write_record(root: &Path, digest: &str, visual: bool) {
    let record = Record {
        digest: digest.into(),
        release: "test".into(),
        channel: "preview/x".into(),
        base: "http://127.0.0.1:1".into(),
        actor: "ops".into(),
        ts_ms: 1_700_000_000_000,
        interact: false,
        visual,
        checks: vec![Check { patch: "buy-text".into(), check: "textContains Checkout now".into(), outcome: Outcome::Pass, detail: String::new() }],
    };
    verify::write(root, &record).unwrap();
}

#[test]
fn an_unapproved_patch_blocks_production_and_the_dry_run_lists_every_reason() {
    let f = Fixture::new("unapproved");
    let out = f.release_to("production", &["--dry-run"]);
    assert!(!out.status.success(), "{}", text(&out));
    let s = text(&out);
    assert!(s.contains("would be BLOCKED"), "{s}");
    assert!(s.contains("buy-text is draft"), "{s}");
    assert!(s.contains("no verification record"), "{s}");
    assert!(!f.root.join("channels/production.txt").exists(), "a dry run publishes nothing");
}

#[test]
fn the_author_cannot_approve_their_own_patch_unless_the_self_review_is_recorded() {
    let f = Fixture::new("self-review");
    assert!(f.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    let refused = f.run("dev-a", &["review", "approve", "buy-text"]);
    assert!(!refused.status.success() && text(&refused).contains("--allow-self-review"), "{}", text(&refused));
    let allowed = f.run("dev-a", &["review", "approve", "buy-text", "--allow-self-review"]);
    assert!(allowed.status.success(), "{}", text(&allowed));
    let log = f.run("ops", &["log", "--tail", "5"]);
    assert!(text(&log).contains("\"self_review\":true"), "{}", text(&log));
}

#[test]
fn editing_an_approved_patch_withdraws_the_approval() {
    let f = Fixture::new("edited");
    f.approve_by("dev-b");
    let path = f.root.join("patches/buy-text.yaml");
    fs::write(&path, fs::read_to_string(&path).unwrap().replace("Checkout now", "Pay now")).unwrap();
    let list = f.run("ops", &["review", "list"]);
    assert!(text(&list).contains("NOT VALID"), "{}", text(&list));
    let out = f.release_to("production", &["--dry-run"]);
    assert!(text(&out).contains("changed after it was approved"), "{}", text(&out));
}

#[test]
fn a_complete_verification_of_the_same_behaviour_unlocks_production() {
    let f = Fixture::new("verified");
    f.approve_by("dev-b");
    assert!(f.release_to("preview/x", &[]).status.success());
    write_record(&f.root, &digest_of(&f.root, "preview/x"), true);
    let out = f.release_to("production", &[]);
    assert!(out.status.success(), "{}", text(&out));
    let log = f.run("ops", &["log", "--tail", "3"]);
    assert!(text(&log).contains("verified_ms\":1700000000000"), "{}", text(&log));
}

#[test]
fn an_incomplete_verification_does_not_unlock_production() {
    let f = Fixture::new("incomplete");
    f.approve_by("dev-b");
    assert!(f.release_to("preview/x", &[]).status.success());
    // Assertions only, no visual check: the record is not complete, so the gate must still refuse.
    write_record(&f.root, &digest_of(&f.root, "preview/x"), false);
    let out = f.release_to("production", &[]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("incomplete"), "{}", text(&out));
}

#[test]
fn a_waiver_goes_through_and_is_recorded_in_the_audit_log() {
    let f = Fixture::new("waiver");
    f.approve_by("dev-b");
    let out = f.release_to("production", &["--waive-verification", "reviewed by hand"]);
    assert!(out.status.success(), "{}", text(&out));
    let log = f.run("ops", &["log", "--tail", "10"]);
    assert!(text(&log).contains("release.waiver") && text(&log).contains("reviewed by hand"), "{}", text(&log));
    assert!(f.run("ops", &["audit-verify"]).status.success());
}

#[test]
fn a_protected_release_needs_approval_even_when_the_content_is_verified() {
    let f = Fixture::new("unapproved-verified");
    assert!(f.release_to("preview/x", &[]).status.success());
    write_record(&f.root, &digest_of(&f.root, "preview/x"), true);
    let out = f.release_to("production", &[]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("not approved for production"), "{}", text(&out));
}

#[test]
fn merge_refuses_a_patch_id_with_different_content_and_writes_nothing() {
    let f = Fixture::new("merge-conflict");
    let other = f.root.join("other");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join("buy-text.yaml"), patch_for("buy-text", "button", "Buy now", "Pay now").to_yaml().unwrap()).unwrap();
    let out = f.run("ops", &["merge", "--with", other.to_str().unwrap(), "--snapshot", &f.snap_arg(), "--write"]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("CONFLICT same id"), "{}", text(&out));
    assert!(text(&out).contains("nothing written"), "{}", text(&out));
}

#[test]
fn merge_copies_a_new_patch_from_the_other_branch() {
    let f = Fixture::new("merge-add");
    let other = f.root.join("other");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join("coupon.yaml"), patch_for("coupon", "p", "Apply coupon", "Enter a code").to_yaml().unwrap()).unwrap();
    let out = f.run("ops", &["merge", "--with", other.to_str().unwrap(), "--snapshot", &f.snap_arg(), "--write"]);
    assert!(out.status.success(), "{}", text(&out));
    assert!(f.root.join("patches/coupon.yaml").exists());
    assert!(text(&f.run("ops", &["log", "--tail", "1"])).contains("\"written\":true"));
}
