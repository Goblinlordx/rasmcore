//! GPU shader source strings for V2 filters.
//!
//! Each shader is a WGSL source string constant. The pipeline composes
//! these with io_f32 bindings automatically. Multi-pass filters have
//! multiple shader constants (one per pass).

pub mod analysis;
pub mod enhancement;
pub mod reduction;
pub mod spatial;
pub mod vignette;
