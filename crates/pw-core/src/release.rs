//! Release manifests, channel pointers, rollback, and the beacon policy that
//! triggers automatic rollback.
//!
//! A release is immutable: its id is a hash of its content, and writing the
//! same id with different content is an error. A channel (for example
//! `production` or `preview/feature-x`) is a pointer to one release.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::anchor::Locators;
use crate::error::{PwError, Result};
use crate::patch::{Assertion, Op, Patch};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifestPatch {
    pub id: String,
    pub sha256: String,
    pub page: String,
    pub locators: Locators,
    pub ops: Vec<Op>,
    #[serde(default, rename = "assert", skip_serializing_if = "Vec::is_empty")]
    pub assertions: Vec<Assertion>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub channel: String,
    pub parent: Option<String>,
    pub created_ms: u64,
    pub patches: Vec<ManifestPatch>,
}

impl Manifest {
    /// Build a manifest from validated patches. Patches are sorted by id so the same set always gives the same id.
    pub fn build(channel: &str, parent: Option<String>, created_ms: u64, patches: &[Patch]) -> Result<Manifest> {
        let mut entries: Vec<ManifestPatch> = patches
            .iter()
            .map(|p| {
                Ok(ManifestPatch {
                    id: p.id.clone(),
                    sha256: p.sha256()?,
                    page: p.page.clone(),
                    locators: p.anchor.locators.clone(),
                    ops: p.ops.clone(),
                    assertions: p.assertions.clone(),
                })
            })
            .collect::<Result<_>>()?;
        entries.sort_by(|a, b| a.id.cmp(&b.id));
        let mut manifest = Manifest { id: String::new(), channel: channel.into(), parent, created_ms, patches: entries };
        manifest.id = manifest.content_id()?;
        Ok(manifest)
    }

    /// Hash over everything except `id` and `created_ms`.
    pub fn content_id(&self) -> Result<String> {
        let body = serde_json::json!({
            "channel": self.channel,
            "parent": self.parent,
            "patches": self.patches,
        });
        Ok(hex::encode(Sha256::digest(body.to_string().as_bytes()))[..16].to_string())
    }
}

/// Hash of what the patches do: their ids, pages, anchors, operations and assertions. It leaves out the
/// channel, the parent, and the file hash, which also covers the author and review state. So the same behaviour
/// gives the same digest on every channel, and approving a patch does not invalidate its verification.
pub fn patches_digest(patches: &[ManifestPatch]) -> Result<String> {
    let behaviour: Vec<Value> = patches
        .iter()
        .map(|p| json!({"id": p.id, "page": p.page, "locators": p.locators, "ops": p.ops, "assert": p.assertions}))
        .collect();
    let body = serde_json::to_string(&behaviour)?;
    Ok(hex::encode(Sha256::digest(body.as_bytes()))[..16].to_string())
}

fn releases_dir(root: &Path) -> PathBuf {
    root.join("releases")
}

fn channel_file(root: &Path, channel: &str) -> PathBuf {
    root.join("channels").join(format!("{channel}.txt"))
}

/// Write a release. An existing file with the same id must match exactly.
pub fn write_release(root: &Path, manifest: &Manifest) -> Result<PathBuf> {
    let dir = releases_dir(root);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", manifest.id));
    let text = serde_json::to_string_pretty(manifest)?;
    if path.exists() {
        if fs::read_to_string(&path)? != text {
            return Err(PwError::Other(format!("release {} already exists with different content", manifest.id)));
        }
    } else {
        fs::write(&path, &text)?;
    }
    Ok(path)
}

pub fn read_release(root: &Path, id: &str) -> Result<Manifest> {
    let path = releases_dir(root).join(format!("{id}.json"));
    let text = fs::read_to_string(&path).map_err(|e| PwError::Other(format!("release `{id}`: {e}")))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn current(root: &Path, channel: &str) -> Result<Option<String>> {
    match fs::read_to_string(channel_file(root, channel)) {
        Ok(t) => Ok(Some(t.trim().to_string()).filter(|s| !s.is_empty())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn set_channel(root: &Path, channel: &str, release_id: &str) -> Result<()> {
    let path = channel_file(root, channel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{release_id}\n"))?;
    Ok(())
}

/// Point the channel back to the current release's parent. Returns the new release id.
pub fn rollback(root: &Path, channel: &str) -> Result<String> {
    let Some(now) = current(root, channel)? else {
        return Err(PwError::Other(format!("channel `{channel}` has no release to roll back from")));
    };
    let manifest = read_release(root, &now)?;
    let Some(parent) = manifest.parent else {
        return Err(PwError::Other(format!("release `{now}` has no parent to roll back to")));
    };
    set_channel(root, channel, &parent)?;
    Ok(parent)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconPolicy {
    pub window_ms: u64,
    pub min_errors: usize,
    pub min_sessions: usize,
}

impl Default for BeaconPolicy {
    fn default() -> Self {
        BeaconPolicy { window_ms: 60_000, min_errors: 5, min_sessions: 3 }
    }
}

/// Sliding-window count of error beacons per release. Triggers at most once per release.
#[derive(Default)]
pub struct BeaconWindow {
    policy: Option<BeaconPolicy>,
    events: HashMap<String, VecDeque<(u64, String)>>,
    tripped: HashMap<String, bool>,
}

impl BeaconWindow {
    pub fn new(policy: BeaconPolicy) -> Self {
        BeaconWindow { policy: Some(policy), ..Default::default() }
    }

    fn policy(&self) -> BeaconPolicy {
        self.policy.unwrap_or_default()
    }

    /// Record one error beacon. Returns true exactly once, when the policy is first met.
    pub fn record_error(&mut self, release: &str, session: &str, ts_ms: u64) -> bool {
        let policy = self.policy();
        if self.tripped.get(release).copied().unwrap_or(false) {
            return false;
        }
        let window = self.events.entry(release.to_string()).or_default();
        window.push_back((ts_ms, session.to_string()));
        while let Some((t, _)) = window.front() {
            if ts_ms.saturating_sub(*t) > policy.window_ms {
                window.pop_front();
            } else {
                break;
            }
        }
        let errors = window.len();
        let mut sessions: Vec<&str> = window.iter().map(|(_, s)| s.as_str()).collect();
        sessions.sort_unstable();
        sessions.dedup();
        if errors >= policy.min_errors && sessions.len() >= policy.min_sessions {
            self.tripped.insert(release.to_string(), true);
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(name: &str) -> PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("release-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn release_id_depends_on_content_not_time() {
        let a = Manifest::build("production", None, 1, &[]).unwrap();
        let b = Manifest::build("production", None, 999, &[]).unwrap();
        assert_eq!(a.id, b.id);
        let c = Manifest::build("preview/x", None, 1, &[]).unwrap();
        assert_ne!(a.id, c.id);
    }

    #[test]
    fn patches_digest_follows_behaviour_not_the_file() {
        let base = ManifestPatch {
            id: "p1".into(),
            sha256: "file-a".into(),
            page: "/".into(),
            locators: Locators { css: Some("#buy".into()), ..Default::default() },
            ops: vec![Op::Hide {}],
            assertions: vec![],
        };
        let mut reapproved = base.clone();
        reapproved.sha256 = "file-b".into();
        assert_eq!(patches_digest(&[base.clone()]).unwrap(), patches_digest(&[reapproved]).unwrap());
        let mut changed = base.clone();
        changed.ops = vec![Op::Show {}];
        assert_ne!(patches_digest(&[base]).unwrap(), patches_digest(&[changed]).unwrap());
    }

    #[test]
    fn rollback_moves_channel_to_parent_and_refuses_when_there_is_none() {
        let dir = root("rollback");
        let first = Manifest::build("production", None, 1, &[]).unwrap();
        write_release(&dir, &first).unwrap();
        set_channel(&dir, "production", &first.id).unwrap();
        assert!(rollback(&dir, "production").is_err(), "first release has no parent");

        let second = Manifest::build("production", Some(first.id.clone()), 2, &[]).unwrap();
        write_release(&dir, &second).unwrap();
        set_channel(&dir, "production", &second.id).unwrap();
        let back = rollback(&dir, "production").unwrap();
        assert_eq!(back, first.id);
        assert_eq!(current(&dir, "production").unwrap().as_deref(), Some(first.id.as_str()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_id_cannot_be_overwritten_with_different_content() {
        let dir = root("immutable");
        let m = Manifest::build("production", None, 1, &[]).unwrap();
        write_release(&dir, &m).unwrap();
        let mut tampered = m.clone();
        tampered.created_ms = 1;
        tampered.patches.push(ManifestPatch {
            id: "evil".into(),
            sha256: "x".into(),
            page: "/".into(),
            locators: Locators::default(),
            ops: vec![],
            assertions: vec![],
        });
        assert!(write_release(&dir, &tampered).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn beacons_below_threshold_do_not_trip() {
        let mut w = BeaconWindow::new(BeaconPolicy { window_ms: 60_000, min_errors: 5, min_sessions: 3 });
        for i in 0..4 {
            assert!(!w.record_error("r1", &format!("s{}", i % 3), 1_000 + i));
        }
    }

    #[test]
    fn many_errors_from_one_session_do_not_trip_the_sessions_rule() {
        let mut w = BeaconWindow::new(BeaconPolicy { window_ms: 60_000, min_errors: 5, min_sessions: 3 });
        for i in 0..10 {
            assert!(!w.record_error("r1", "same-session", 1_000 + i));
        }
    }

    #[test]
    fn enough_errors_from_enough_sessions_trip_once() {
        let mut w = BeaconWindow::new(BeaconPolicy { window_ms: 60_000, min_errors: 5, min_sessions: 3 });
        let tripped: Vec<bool> = (0..6).map(|i| w.record_error("r1", &format!("s{}", i % 3), 1_000 + i)).collect();
        assert_eq!(tripped.iter().filter(|t| **t).count(), 1);
        assert!(tripped[4], "trips on the fifth error");
    }

    #[test]
    fn old_errors_fall_out_of_the_window() {
        let mut w = BeaconWindow::new(BeaconPolicy { window_ms: 1_000, min_errors: 3, min_sessions: 1 });
        assert!(!w.record_error("r1", "a", 0));
        assert!(!w.record_error("r1", "a", 100));
        assert!(!w.record_error("r1", "a", 5_000));
        assert!(!w.record_error("r1", "a", 5_100));
    }

    #[test]
    fn releases_are_counted_separately() {
        let mut w = BeaconWindow::new(BeaconPolicy { window_ms: 60_000, min_errors: 3, min_sessions: 1 });
        assert!(!w.record_error("r1", "a", 1));
        assert!(!w.record_error("r2", "a", 2));
        assert!(!w.record_error("r2", "a", 3));
        assert!(w.record_error("r1", "a", 4) == false);
    }
}
