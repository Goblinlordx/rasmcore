//! Filter capability traits for the unified `#[derive(Filter)]` architecture.
//!
//! Each filter struct implements one or more of these traits:
//! - `CpuFilter` (required) — CPU compute logic
//! - `GpuFilter` (optional) — GPU shader dispatch
//! - `PointOp` (optional) — 1D per-channel LUT for pipeline fusion
//! - `ColorOp` (optional) — 3D CLUT for pipeline color op fusion
//!
//! The `#[derive(Filter)]` macro generates ConfigParams, registration, and
//! pipeline node wiring. These traits provide the runtime behavior.

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;
use rasmcore_pipeline::Rect;

/// Type alias for the upstream pixel-fetching closure.
pub type UpstreamFn = dyn FnMut(Rect) -> Result<Vec<u8>, ImageError>;

/// CPU compute implementation for a filter.
///
/// The filter struct's fields ARE the config params (populated from user input).
/// `&self` provides access to all parameters.
pub trait CpuFilter {
    /// Compute the filtered output for the requested region.
    ///
    /// - `request`: The output region to compute
    /// - `upstream`: Closure to fetch input pixels (may request larger region for overlap)
    /// - `info`: Image metadata (width, height, format, color space)
    fn compute(
        &self,
        request: Rect,
        upstream: &mut UpstreamFn,
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError>;
}

/// GPU shader dispatch for a filter (optional).
///
/// If implemented, the pipeline can offload this filter to the GPU.
/// The CPU path (CpuFilter) is used as fallback when GPU is unavailable.
pub trait GpuFilter {
    /// Return GPU operations for this filter's current configuration.
    ///
    /// Returns `None` to fall back to CPU for this particular invocation
    /// (e.g., unsupported parameter range on GPU).
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>>;
}

/// Per-channel 1D LUT for pipeline point-op fusion (optional).
///
/// If implemented, the pipeline optimizer can fuse consecutive point ops
/// into a single composed 256-entry LUT applied in one memory pass.
pub trait PointOp {
    /// Build a 256-entry u8→u8 lookup table for this operation.
    fn build_lut(&self) -> [u8; 256];
}

/// 3D CLUT for pipeline color-op fusion (optional).
///
/// If implemented, the pipeline optimizer can fuse consecutive color ops
/// into a single composed 3D CLUT applied via tetrahedral interpolation.
pub trait ColorOp {
    /// Build a 3D color lookup table for this operation.
    fn build_clut(&self) -> crate::domain::color_lut::ColorLut3D;
}
