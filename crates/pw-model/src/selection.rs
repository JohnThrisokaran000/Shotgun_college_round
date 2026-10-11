//! The setting the combined tracker uses with a matcher, as chosen by `pw train track --grid` on the development
//! seeds (docs/DESIGN.md, section 6.10). The grid writes it as `selection.json` in the matcher's folder, and the CLI
//! reads it from there, so no threshold is compiled into the program.
//!
//! A record names the weights it was chosen for, by their SHA-256. If the weights change, the record stops applying
//! until the grid runs again, so a stale setting cannot be used with a different matcher.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use pw_core::error::{PwError, Result};
use pw_core::hash::file_sha256;

/// The file name in a matcher's folder.
pub const SELECTION_FILE: &str = "selection.json";
/// The record format this build reads and writes.
pub const SELECTION_SCHEMA: u32 = 1;

/// The combined tracker's thresholds and veto: `matcher_accept`, `matcher_margin`, `matcher_review` and the veto.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Setting {
    pub accept: f64,
    pub margin: f64,
    pub review: f64,
    pub veto: f64,
}

/// What the grid chose for one matcher, and what it was chosen on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub schema: u32,
    /// How the setting was chosen. "grid": every setting in the pre-registered grid, ranked by the summed objective.
    pub method: String,
    pub setting: Setting,
    /// The chosen setting's objective, summed over the slices (`pw_train::bench::objective`).
    pub objective_sum: i64,
    /// The slices the choice was made on, as `SEED:mixed` or `SEED:compound`.
    pub slices: Vec<String>,
    pub sites: u64,
    pub variants: u64,
    /// How many settings the grid compared.
    pub settings_searched: usize,
    /// SHA-256 of the `model.safetensors` this record was chosen for.
    pub matcher_sha256: String,
    /// The command that wrote the record.
    pub written_by: String,
}

impl Selection {
    /// The record in `dir`. A folder without one fails with the command that makes one.
    pub fn load(dir: &Path) -> Result<Selection> {
        let path = dir.join(SELECTION_FILE);
        let text = fs::read_to_string(&path).map_err(|_| {
            PwError::Other(format!(
                "{} has no {SELECTION_FILE}. Choose a setting first, on the development seeds: \
                 pw train track --matcher {} --slices 101:mixed,102:compound --grid",
                dir.display(),
                dir.display()
            ))
        })?;
        let sel: Selection = serde_json::from_str(&text).map_err(|e| PwError::Other(format!("{}: {e}", path.display())))?;
        if sel.schema != SELECTION_SCHEMA {
            return Err(PwError::Other(format!("{}: schema {} is not {SELECTION_SCHEMA}", path.display(), sel.schema)));
        }
        Ok(sel)
    }

    /// Refuse the record when `dir` holds different weights from the ones it was chosen for.
    pub fn check_matcher(&self, dir: &Path) -> Result<()> {
        let actual = file_sha256(&dir.join("model.safetensors"))?;
        if actual != self.matcher_sha256 {
            return Err(PwError::Other(format!(
                "{} was selected for different weights: model.safetensors changed after the grid ran. Run the grid again.",
                dir.display()
            )));
        }
        Ok(())
    }

    /// Write the record into `dir`, replacing an older one.
    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        let path = dir.join(SELECTION_FILE);
        fs::write(&path, format!("{}\n", serde_json::to_string_pretty(self)?))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("selection-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample(hash: &str) -> Selection {
        Selection {
            schema: SELECTION_SCHEMA,
            method: "grid".into(),
            setting: Setting { accept: 0.8, margin: 0.2, review: 0.4, veto: 0.4 },
            objective_sum: 0,
            slices: vec!["101:mixed".into(), "102:compound".into()],
            sites: 300,
            variants: 3,
            settings_searched: 840,
            matcher_sha256: hash.into(),
            written_by: "test".into(),
        }
    }

    #[test]
    fn a_folder_without_a_record_says_which_command_makes_one() {
        let d = dir("missing");
        let err = Selection::load(&d).unwrap_err().to_string();
        assert!(err.contains("has no selection.json") && err.contains("--grid"), "{err}");
    }

    #[test]
    fn a_record_round_trips_and_applies_only_to_the_weights_it_names() {
        let d = dir("roundtrip");
        fs::write(d.join("model.safetensors"), b"weights v1").unwrap();
        let hash = file_sha256(&d.join("model.safetensors")).unwrap();
        sample(&hash).write(&d).unwrap();
        let loaded = Selection::load(&d).unwrap();
        assert_eq!(loaded, sample(&hash));
        loaded.check_matcher(&d).unwrap();
        fs::write(d.join("model.safetensors"), b"weights v2").unwrap();
        let err = loaded.check_matcher(&d).unwrap_err().to_string();
        assert!(err.contains("different weights"), "{err}");
    }

    #[test]
    fn an_unknown_schema_is_refused() {
        let d = dir("schema");
        let mut sel = sample("x");
        sel.schema = 99;
        sel.write(&d).unwrap();
        let err = Selection::load(&d).unwrap_err().to_string();
        assert!(err.contains("schema 99"), "{err}");
    }
}
