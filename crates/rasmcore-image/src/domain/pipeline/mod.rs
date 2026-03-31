//! Demand-driven tile pipeline engine for image processing.
//!
//! Core primitives (Rect, SpatialCache) come from rasmcore-pipeline.
//! Image-specific node trait and implementations live here.

// Re-export shared primitives
pub use rasmcore_pipeline::{Rect, SpatialCache};
pub use rasmcore_pipeline::{cache, rect};

pub mod dispatch;
pub mod graph;
pub mod nodes;
mod tests;
