//! Verification records. `pw verify` checks a channel's release in a browser and writes one record.
//! A release to a protected channel needs a complete, passing record for the same patch content.
//!
//! Records are keyed by the patch digest (see `release::patches_digest`), not by release id. A release id
//! includes its channel, so a release to `preview/x` and one to `production` have different ids even
//! when they serve the same patches. The digest is the same for both.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{PwError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Pass,
    Fail,
    /// The check was not run, for example a click check without `--interact`, or a visual check with
    /// nothing to compare against. A record with a check not run is not complete.
    NotRun,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Check {
    /// A patch id, or `visual:<page>` for a page-level visual check.
    pub patch: String,
    /// The assertion's label, or `visual`.
    pub check: String,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub digest: String,
    pub release: String,
    pub channel: String,
    pub base: String,
    pub actor: String,
    pub ts_ms: u64,
    /// Whether click checks ran (`--interact`).
    pub interact: bool,
    /// Whether visual checks ran. They compare the channel with the baseline channel.
    pub visual: bool,
    pub checks: Vec<Check>,
}

impl Record {
    pub fn count(&self, outcome: Outcome) -> usize {
        self.checks.iter().filter(|c| c.outcome == outcome).count()
    }

    /// A record can gate a release only when visual checks ran, there is at least one check, and every
    /// check passed. A check that was not run counts against the record.
    pub fn complete(&self) -> bool {
        self.visual && !self.checks.is_empty() && self.checks.iter().all(|c| c.outcome == Outcome::Pass)
    }
}

fn records_dir(root: &Path) -> PathBuf {
    root.join("verifications")
}

/// Write a record. Records are never overwritten: a new run makes a new file.
pub fn write(root: &Path, record: &Record) -> Result<PathBuf> {
    let dir = records_dir(root);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}-{}.json", record.digest, record.ts_ms));
    if path.exists() {
        return Err(PwError::Other(format!("verification record {} already exists", path.display())));
    }
    fs::write(&path, serde_json::to_string_pretty(record)?)?;
    Ok(path)
}

/// The newest record for a patch digest, if there is one.
pub fn latest(root: &Path, digest: &str) -> Result<Option<Record>> {
    let entries = match fs::read_dir(records_dir(root)) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let prefix = format!("{digest}-");
    let mut best: Option<Record> = None;
    for entry in entries {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if !name.starts_with(&prefix) || !name.ends_with(".json") {
            continue;
        }
        let record: Record = serde_json::from_str(&fs::read_to_string(&path)?)
            .map_err(|e| PwError::Other(format!("unreadable verification record {}: {e}", path.display())))?;
        if record.digest == digest && best.as_ref().is_none_or(|b| record.ts_ms > b.ts_ms) {
            best = Some(record);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("verify-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn record(digest: &str, ts_ms: u64, outcomes: &[Outcome]) -> Record {
        Record {
            digest: digest.into(),
            release: "r1".into(),
            channel: "preview/x".into(),
            base: "http://127.0.0.1:8787".into(),
            actor: "dev-b".into(),
            ts_ms,
            interact: false,
            visual: true,
            checks: outcomes
                .iter()
                .enumerate()
                .map(|(i, o)| Check { patch: format!("p{i}"), check: "visible=true".into(), outcome: *o, detail: String::new() })
                .collect(),
        }
    }

    #[test]
    fn complete_needs_visual_checks_at_least_one_check_and_no_failures_or_skips() {
        assert!(record("d", 1, &[Outcome::Pass]).complete());
        assert!(!record("d", 1, &[]).complete(), "nothing verified is not a pass");
        assert!(!record("d", 1, &[Outcome::Pass, Outcome::Fail]).complete());
        assert!(!record("d", 1, &[Outcome::Pass, Outcome::NotRun]).complete());
        let mut no_visual = record("d", 1, &[Outcome::Pass]);
        no_visual.visual = false;
        assert!(!no_visual.complete());
    }

    #[test]
    fn latest_picks_the_newest_record_for_the_digest_only() {
        let dir = root("latest");
        write(&dir, &record("aaa", 10, &[Outcome::Fail])).unwrap();
        write(&dir, &record("aaa", 30, &[Outcome::Pass])).unwrap();
        write(&dir, &record("aaa", 20, &[Outcome::Pass, Outcome::Fail])).unwrap();
        write(&dir, &record("bbb", 99, &[Outcome::Pass])).unwrap();
        let newest = latest(&dir, "aaa").unwrap().unwrap();
        assert_eq!(newest.ts_ms, 30);
        assert!(newest.complete());
        assert!(latest(&dir, "ccc").unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_is_never_overwritten() {
        let dir = root("immutable");
        write(&dir, &record("aaa", 5, &[Outcome::Pass])).unwrap();
        assert!(write(&dir, &record("aaa", 5, &[Outcome::Fail])).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
