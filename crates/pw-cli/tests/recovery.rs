//! Conflict resolution and recovery, end to end through the `pw` binary. No browser is needed: verification
//! records are written the same way `pw verify` writes them.

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

/// A patch anchored to the button in `SITE`, setting its text to `value`.
fn button_patch(id: &str, value: &str) -> Patch {
    let snap = Snapshot::parse(SITE);
    let el = snap.elements().iter().find(|e| e.tag == "button").expect("button in the sample page");
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
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("recovery-{name}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let snapshot = root.join("index.html");
        fs::write(&snapshot, SITE).unwrap();
        let out = pw(&root, "setup", &["init", "--name", "demo", "--upstream", "http://127.0.0.1:1"]);
        assert!(out.status.success(), "{}", text(&out));
        fs::write(root.join("patches/buy-text.yaml"), button_patch("buy-text", "Checkout now").to_yaml().unwrap()).unwrap();
        Fixture { root, snapshot }
    }

    fn snap_arg(&self) -> String {
        format!("/={}", self.snapshot.display())
    }

    fn run(&self, actor: &str, args: &[&str]) -> Output {
        pw(&self.root, actor, args)
    }

    fn release_to(&self, channel: &str) -> Output {
        let snap = self.snap_arg();
        self.run("ops", &["release", "--channel", channel, "--snapshot", snap.as_str()])
    }

    fn current(&self, channel: &str) -> String {
        release::current(&self.root, channel).unwrap().expect("a release on the channel")
    }
}

/// Write a passing, complete verification record for the content the channel serves now.
fn verify_channel(root: &Path, channel: &str) {
    let id = release::current(root, channel).unwrap().unwrap();
    let manifest = release::read_release(root, &id).unwrap();
    let record = Record {
        digest: release::patches_digest(&manifest.patches).unwrap(),
        release: id,
        channel: channel.into(),
        base: "http://127.0.0.1:1".into(),
        actor: "ops".into(),
        ts_ms: 1_700_000_000_000 + manifest.created_ms % 1_000_000,
        interact: false,
        visual: true,
        checks: vec![Check { patch: "buy-text".into(), check: "textContains Checkout now".into(), outcome: Outcome::Pass, detail: String::new() }],
    };
    verify::write(root, &record).unwrap();
}

#[test]
fn resolving_a_conflict_rejects_the_dropped_patch_and_releases_leave_it_out() {
    let f = Fixture::new("conflict");
    // A second patch writes the same element with different text: a hard write-write conflict.
    fs::write(f.root.join("patches/buy-text-alt.yaml"), button_patch("buy-text-alt", "Pay now").to_yaml().unwrap()).unwrap();

    let before = f.run("ops", &["conflicts", "list", "--snapshot", &f.snap_arg()]);
    assert!(!before.status.success(), "a hard conflict must fail the listing: {}", text(&before));
    assert!(text(&before).contains("CONFLICT"), "{}", text(&before));

    let resolved = f.run(
        "dev-b",
        &["conflicts", "resolve", "--keep", "buy-text", "--drop", "buy-text-alt", "--reason", "keep the reviewed copy"],
    );
    assert!(resolved.status.success(), "{}", text(&resolved));

    let after = f.run("ops", &["conflicts", "list", "--snapshot", &f.snap_arg()]);
    assert!(after.status.success(), "{}", text(&after));
    assert!(text(&after).contains("no conflicts among 1 active patches"), "{}", text(&after));

    let out = f.release_to("preview/x");
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("withdrawn  buy-text-alt"), "{}", text(&out));
    let manifest = release::read_release(&f.root, &f.current("preview/x")).unwrap();
    let ids: Vec<&str> = manifest.patches.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, vec!["buy-text"], "the dropped patch is not part of the release");

    let log = text(&f.run("ops", &["log", "--tail", "4"]));
    assert!(log.contains("conflict.resolve") && log.contains("keep the reviewed copy"), "{log}");
}

#[test]
fn rollback_last_verified_goes_to_the_newest_release_with_a_passing_record() {
    let f = Fixture::new("last-verified");
    assert!(f.release_to("preview/x").status.success());
    verify_channel(&f.root, "preview/x");
    let good = f.current("preview/x");

    // A later content change that nobody has verified.
    fs::write(f.root.join("patches/buy-text.yaml"), button_patch("buy-text", "Pay now").to_yaml().unwrap()).unwrap();
    assert!(f.release_to("preview/x").status.success());
    let bad = f.current("preview/x");
    assert_ne!(good, bad);

    let out = f.run("ops", &["rollback", "--channel", "preview/x", "--last-verified"]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(f.current("preview/x"), good, "the channel serves the verified release again");
    let log = text(&f.run("ops", &["log", "--tail", "1"]));
    assert!(log.contains("last_verified"), "{log}");
}

#[test]
fn rollback_to_a_named_release_and_the_default_parent_rollback() {
    let f = Fixture::new("named");
    assert!(f.release_to("preview/x").status.success());
    let first = f.current("preview/x");
    fs::write(f.root.join("patches/buy-text.yaml"), button_patch("buy-text", "Pay now").to_yaml().unwrap()).unwrap();
    assert!(f.release_to("preview/x").status.success());
    let second = f.current("preview/x");

    let named = f.run("ops", &["rollback", "--channel", "preview/x", "--to", &first]);
    assert!(named.status.success(), "{}", text(&named));
    assert_eq!(f.current("preview/x"), first);

    let back = f.run("ops", &["rollback", "--channel", "preview/x", "--to", &second]);
    assert!(back.status.success(), "{}", text(&back));
    let parent = f.run("ops", &["rollback", "--channel", "preview/x"]);
    assert!(parent.status.success(), "{}", text(&parent));
    assert_eq!(f.current("preview/x"), first, "the default rollback goes to the parent");
}

#[test]
fn a_rejected_patch_neither_blocks_nor_ships_in_a_protected_release() {
    let f = Fixture::new("withdrawn");
    fs::write(f.root.join("patches/buy-text-draft.yaml"), button_patch("buy-text-draft", "Pay now").to_yaml().unwrap()).unwrap();
    // The draft is rejected. It is not approved, and the production gate must not care about it.
    assert!(f.run("dev-a", &["review", "reject", "buy-text-draft", "--reason", "not wanted"]).status.success());
    assert!(f.run("dev-a", &["review", "propose", "buy-text"]).status.success());
    assert!(f.run("dev-b", &["review", "approve", "buy-text"]).status.success());

    // Protected releases need a verification record for the content they publish. Make one on preview first.
    assert!(f.release_to("preview/x").status.success());
    verify_channel(&f.root, "preview/x");
    let out = f.release_to("production");
    assert!(out.status.success(), "{}", text(&out));
    let manifest = release::read_release(&f.root, &f.current("production")).unwrap();
    let ids: Vec<&str> = manifest.patches.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, vec!["buy-text"], "only the approved, active patch ships");
    assert!(manifest.patches.iter().all(|p| p.ops.iter().all(|op| !matches!(op, Op::SetText { value } if value == "Pay now"))));
}
