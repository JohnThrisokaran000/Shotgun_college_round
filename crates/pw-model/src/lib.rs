//! Runs the PatchWork model in Rust, with no Python at run time.
//!
//! The model is the Gemma-3-style decoder trained by `training/` and exported as a Hugging Face folder
//! (`config.json`, `model.safetensors`, `tokenizer.json`). [`Model`] decodes it greedily. [`ModelChooser`]
//! plugs it into re-anchoring as a `pw_core::relocate::Chooser`, which is where the heuristic used to be.
//!
//! The model is a suggestion, never a source of truth. A chooser answer is applied only when the model is
//! confident enough (see `ModelChooser::model_accept`). Otherwise the result is `Unsure`, which asks a human.

mod chooser;
mod matcher;
mod model;

pub use chooser::{alone, decide, Judgement, ModelChooser, ModelFirst};
pub use matcher::{CombinedChooser, Matcher, MatcherChooser, MatcherOnly};
pub use model::{Generation, Model};
