//! Append-only, hash-chained audit log (JSONL).
//!
//! Each event's hash covers its body and the previous hash, so editing or removing
//! an event in the middle breaks the chain. Truncating the tail is caught by
//! comparing the last hash with the HEAD file. The HEAD file lives next to the log,
//! so someone who can rewrite both can still forge history. Keep HEAD somewhere
//! else too (commit it to git, for example) if you need protection against that.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::error::{PwError, Result};

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub seq: u64,
    pub ts_ms: u64,
    pub actor: String,
    pub kind: String,
    pub data: Value,
    pub prev: String,
    pub hash: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Report {
    pub events: usize,
    pub head: String,
}

/// serde_json maps are sorted by key by default, so this body serialises the same way every time.
pub fn hash_of(seq: u64, ts_ms: u64, actor: &str, kind: &str, data: &Value, prev: &str) -> String {
    let body = json!({
        "seq": seq,
        "ts_ms": ts_ms,
        "actor": actor,
        "kind": kind,
        "data": data,
        "prev": prev,
    });
    hex::encode(Sha256::digest(body.to_string().as_bytes()))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub struct AuditLog {
    dir: PathBuf,
}

impl AuditLog {
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        AuditLog { dir: dir.into() }
    }

    fn events_path(&self) -> PathBuf {
        self.dir.join("events.jsonl")
    }

    fn head_path(&self) -> PathBuf {
        self.dir.join("HEAD")
    }

    pub fn read_all(&self) -> Result<Vec<Event>> {
        let text = match fs::read_to_string(self.events_path()) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        text.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(i, line)| {
                serde_json::from_str::<Event>(line).map_err(|e| PwError::AuditChain { line: i + 1, reason: format!("unparseable event: {e}") })
            })
            .collect()
    }

    pub fn append(&self, actor: &str, kind: &str, data: Value) -> Result<Event> {
        fs::create_dir_all(&self.dir)?;
        let events = self.read_all()?;
        let (seq, prev) = match events.last() {
            Some(last) => (last.seq + 1, last.hash.clone()),
            None => (1, GENESIS.to_string()),
        };
        let ts_ms = now_ms();
        let hash = hash_of(seq, ts_ms, actor, kind, &data, &prev);
        let event = Event { seq, ts_ms, actor: actor.into(), kind: kind.into(), data, prev, hash };
        let mut file = OpenOptions::new().create(true).append(true).open(self.events_path())?;
        writeln!(file, "{}", serde_json::to_string(&event)?)?;
        file.sync_data()?;
        fs::write(self.head_path(), format!("{}\n", event.hash))?;
        Ok(event)
    }

    pub fn verify(&self) -> Result<Report> {
        let events = self.read_all()?;
        let mut prev = GENESIS.to_string();
        for (i, e) in events.iter().enumerate() {
            let line = i + 1;
            if e.seq != line as u64 {
                return Err(PwError::AuditChain { line, reason: format!("sequence {} where {line} was expected", e.seq) });
            }
            if e.prev != prev {
                return Err(PwError::AuditChain { line, reason: "previous hash does not match the line before".into() });
            }
            if hash_of(e.seq, e.ts_ms, &e.actor, &e.kind, &e.data, &e.prev) != e.hash {
                return Err(PwError::AuditChain { line, reason: "hash does not match the event body (edited)".into() });
            }
            prev = e.hash.clone();
        }
        let head = match fs::read_to_string(self.head_path()) {
            Ok(h) => h.trim().to_string(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if events.is_empty() {
                    return Ok(Report { events: 0, head: GENESIS.to_string() });
                }
                return Err(PwError::AuditChain { line: events.len(), reason: "HEAD file is missing".into() });
            }
            Err(e) => return Err(e.into()),
        };
        if head != prev {
            return Err(PwError::AuditChain {
                line: events.len(),
                reason: "HEAD does not match the last event (tail truncated or history rewritten)".into(),
            });
        }
        Ok(Report { events: events.len(), head: prev })
    }
}

/// Convenience for the CLI: is the log at `dir` intact?
pub fn verify_dir(dir: &Path) -> Result<Report> {
    AuditLog::open(dir).verify()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("audit-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn appended_events_verify() {
        let dir = tmp("ok");
        let log = AuditLog::open(&dir);
        log.append("dev-a", "patch.add", json!({"id": "p1"})).unwrap();
        log.append("dev-b", "release", json!({"id": "r1"})).unwrap();
        let report = log.verify().unwrap();
        assert_eq!(report.events, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_an_old_event_is_detected_at_its_line() {
        let dir = tmp("edit");
        let log = AuditLog::open(&dir);
        log.append("dev-a", "patch.add", json!({"id": "p1"})).unwrap();
        log.append("dev-b", "release", json!({"id": "r1"})).unwrap();
        let text = fs::read_to_string(log.events_path()).unwrap();
        fs::write(log.events_path(), text.replace("\"id\":\"p1\"", "\"id\":\"p9\"")).unwrap();
        match log.verify() {
            Err(PwError::AuditChain { line, .. }) => assert_eq!(line, 1),
            other => panic!("expected chain error, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncating_the_tail_is_detected_against_head() {
        let dir = tmp("trunc");
        let log = AuditLog::open(&dir);
        log.append("dev-a", "a", json!({})).unwrap();
        log.append("dev-a", "b", json!({})).unwrap();
        let text = fs::read_to_string(log.events_path()).unwrap();
        let first_line = text.lines().next().unwrap().to_string() + "\n";
        fs::write(log.events_path(), first_line).unwrap();
        assert!(matches!(log.verify(), Err(PwError::AuditChain { .. })));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_log_is_valid() {
        let dir = tmp("empty");
        let report = AuditLog::open(&dir).verify().unwrap();
        assert_eq!(report.events, 0);
        assert_eq!(report.head, GENESIS);
    }
}
