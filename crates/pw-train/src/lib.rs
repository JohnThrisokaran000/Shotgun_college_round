//! Training support for PatchWork: synthetic data, the held-out benchmark, and (in later
//! modules) the distributed coordinator and the Kaggle launcher.
//!
//! The model itself is trained in Python (PyTorch) and served from Rust. See `training/`.

pub mod bench;
pub mod coordinator;
pub mod datagen;
pub mod synth;
