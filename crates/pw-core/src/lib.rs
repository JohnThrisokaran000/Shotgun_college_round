//! PatchWork core: DOM snapshots, element fingerprints, the re-anchoring
//! (relocation) pipeline, the patch DSL, conflict checks, the audit chain, and
//! release manifests. Deterministic code only. The model is an optional
//! `Chooser` behind a trait, so every path works without it.
pub mod anchor;
pub mod audit;
pub mod conflict;
pub mod dsl;
pub mod error;
pub mod patch;
pub mod prompt;
pub mod relocate;
pub mod release;
pub mod serialize;
pub mod similarity;
pub mod snapshot;

pub use error::{PwError, Result};
