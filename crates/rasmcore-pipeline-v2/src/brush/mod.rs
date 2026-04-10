//! CPU stamp-based brush engine — spacing interpolation, dynamics curves,
//! dab generation, and accumulation buffer compositing.
//!
//! This is the CPU reference implementation. The GPU engine (separate track)
//! mirrors this logic in compute shaders.

pub mod composite;
pub mod dab;
pub mod dynamics;
pub mod engine;
pub mod path;
pub mod presets;
pub mod types;

pub use engine::CpuBrushEngine;
pub use presets::{BrushPreset, find_preset, registered_presets};
pub use types::*;
