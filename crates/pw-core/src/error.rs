use thiserror::Error;

#[derive(Debug, Error)]
pub enum PwError {
    #[error("invalid CSS selector `{selector}`: {reason}")]
    Selector { selector: String, reason: String },
    #[error("invalid patch `{patch}`: {reason}")]
    InvalidPatch { patch: String, reason: String },
    #[error("audit chain broken at line {line}: {reason}")]
    AuditChain { line: usize, reason: String },
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PwError>;
