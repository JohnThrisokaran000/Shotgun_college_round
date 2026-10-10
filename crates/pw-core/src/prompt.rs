//! Prompt and answer format for the RELOCATE job. Shared by the training data
//! generator and the CLI, so training and serving use the same text.

use crate::anchor::Fingerprint;
use crate::error::{PwError, Result};
use crate::serialize::{element_line_no_box, fingerprint_line_no_box};
use crate::snapshot::Element;

pub const MAX_CANDIDATES: usize = 8;

/// Build the RELOCATE prompt: the stored fingerprint, then up to eight candidates.
pub fn relocate_prompt(fp: &Fingerprint, candidates: &[(&Element, f64)]) -> String {
    let mut out = String::from("TASK: relocate\nOLD: ");
    out.push_str(&fingerprint_line_no_box(fp, "[old]"));
    out.push_str("\nCANDIDATES:\n");
    for (k, (el, _)) in candidates.iter().take(MAX_CANDIDATES).enumerate() {
        out.push_str(&element_line_no_box(el, &format!("[{k}]")));
        out.push('\n');
    }
    out.push_str("ANSWER: ");
    out
}

#[derive(Debug, PartialEq, Eq)]
pub enum Answer {
    Match(usize),
    Null,
}

/// Parse `{"match": k}` or `{"match": null}`. Anything else is an error, never a guess.
pub fn parse_answer(text: &str, candidate_count: usize) -> Result<Answer> {
    // Read the raw JSON value: serde would turn `null` into `None` and hide it from a missing key.
    let value: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| PwError::Other(format!("model answer is not valid JSON: {e}")))?;
    let Some(v) = value.get("match") else {
        return Err(PwError::Other("model answer has no `match` field".into()));
    };
    if v.is_null() {
        return Ok(Answer::Null);
    }
    let k = v
        .as_u64()
        .ok_or_else(|| PwError::Other(format!("`match` must be an integer or null, got {v}")))?
        as usize;
    if k < candidate_count {
        Ok(Answer::Match(k))
    } else {
        Err(PwError::Other(format!("`match` {k} is outside the {candidate_count} candidates")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_answers() {
        assert_eq!(parse_answer(r#"{"match": 3}"#, 8).unwrap(), Answer::Match(3));
        assert_eq!(parse_answer(r#"{"match": null}"#, 8).unwrap(), Answer::Null);
        assert_eq!(parse_answer("  {\"match\": 0}\n", 1).unwrap(), Answer::Match(0));
    }

    #[test]
    fn rejects_out_of_range_and_malformed_answers() {
        assert!(parse_answer(r#"{"match": 8}"#, 8).is_err());
        assert!(parse_answer(r#"{"match": "3"}"#, 8).is_err());
        assert!(parse_answer(r#"{"choice": 3}"#, 8).is_err());
        assert!(parse_answer("the third one", 8).is_err());
        assert!(parse_answer(r#"{"match": -1}"#, 8).is_err());
    }
}
