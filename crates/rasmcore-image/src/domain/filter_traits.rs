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
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError>;

    /// Compute the input rect needed to produce the given output rect.
    ///
    /// Spatial operations expand the output rect by their kernel/overlap size.
    /// Point operations return the output rect unchanged (default).
    /// Transform/distortion operations compute the bounding box of their
    /// inverse coordinate mapping.
    ///
    /// `bounds_w`/`bounds_h` are the full image dimensions for clamping.
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.clamp(bounds_w, bounds_h)
    }
}

/// Helper for codegen: dispatches CpuFilter::compute with proper borrow handling.
///
/// Generic over F to avoid trait-object borrow issues — the compiler can see
/// the closure doesn't escape when F is concrete.
#[doc(hidden)]
#[inline]
pub fn __cpu_filter_dispatch<F: CpuFilter>(
    filter: &F,
    request: Rect,
    upstream_id: u32,
    upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    // SAFETY: The upstream closure is only used for the duration of filter.compute().
    // CpuFilter::compute() must not store the upstream reference — it's purely
    // a callback for the duration of the call. This unsafe block is necessary
    // because the borrow checker can't verify this through trait object dispatch.
    // The old-style codegen (calling bare functions) doesn't need this because
    // the compiler can inline and verify the function body directly.
    let uid = upstream_id;
    let ptr: *mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError> = upstream_fn;
    let mut up = move |rect: Rect| -> Result<Vec<u8>, ImageError> {
        // SAFETY: ptr is valid for the duration of filter.compute()
        unsafe { (&mut *ptr)(uid, rect) }
    };
    filter.compute(request, &mut up, info)
}

/// Input rect provider — declares the input region needed for an output region.
///
/// Config structs implement this to declare their kernel overlap.
/// Default: identity (no expansion) — point-ops don't need to override.
/// Spatial filters override with their kernel expansion logic.
pub trait InputRectProvider {
    /// Compute the input rect needed to produce the given output rect.
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.clamp(bounds_w, bounds_h)
    }
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

    /// Return GPU operations with explicit buffer format selection.
    ///
    /// Override to provide f32 shader variants when `buffer_format` is `F32Vec4`.
    /// Default delegates to `gpu_ops()` (backward compatible — all existing filters
    /// return u32 shaders regardless of format).
    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        let _ = buffer_format;
        self.gpu_ops(width, height)
    }
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
