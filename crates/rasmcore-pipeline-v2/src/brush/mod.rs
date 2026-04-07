//! CPU stamp-based brush engine — spacing interpolation, dynamics curves,
//! dab generation, and accumulation buffer compositing.
//!
//! This is the CPU reference implementation. The GPU engine (separate track)
//! mirrors this logic in compute shaders.

pub mod types;
pub mod dynamics;
pub mod path;
pub mod dab;
pub mod composite;
pub mod engine;

pub use types::*;
pub use engine::CpuBrushEngine;
