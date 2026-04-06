//! V2 filter implementations — f32-only, single code path per filter.
//!
//! Each filter implements the `Filter` trait (f32 compute) and optionally
//! `GpuFilter` (WGSL shader), `AnalyticOp` (expression tree for fusion),
//! or `ClutOp` (3D CLUT for color op fusion).
//!
//! No format dispatch. No u8/u16 paths. No PixelFormat. Just f32.

pub mod adjustment;
pub mod color;
pub mod distortion;
pub mod effect;
pub mod enhancement;
pub mod morphology;
pub mod grading;
pub mod scope;
pub mod spatial;
