//! V2 filter implementations — f32-only, single code path per filter.
//!
//! Each filter implements the `Filter` trait (f32 compute) and optionally
//! `GpuFilter` (WGSL shader), `AnalyticOp` (expression tree for fusion),
//! or `ClutOp` (3D CLUT for color op fusion).
//!
//! No format dispatch. No u8/u16 paths. No PixelFormat. Just f32.

pub mod adjustment;
pub mod alpha;
pub mod analysis;
pub mod color;
pub mod composite;
pub mod distortion;
pub mod draw;
pub mod edge;
pub mod effect;
pub mod enhancement;
pub mod evaluate;
pub mod generator;
pub mod grading;
pub mod helpers;
pub mod mask;
pub mod morphology;
pub mod scope;
pub mod spatial;
pub mod tool;
