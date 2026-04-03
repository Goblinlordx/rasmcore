//! Node trait — the core abstraction for all pipeline operations.
//!
//! Every operation in the pipeline (filter, transform, codec, color conversion)
//! implements `Node`. The pipeline engine dispatches through this trait without
//! knowing what specific operations exist.
//!
//! All pixel data is `&[f32]` — RGBA, 4 channels per pixel, normalized [0,1]
//! for SDR or unbounded for HDR. No PixelFormat. No format dispatch. Period.

use crate::color_space::ColorSpace;
use crate::rect::Rect;

/// Information about a node's output.
///
/// No PixelFormat — always f32 RGBA (4 channels). The only metadata
/// that varies is dimensions and color space.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub width: u32,
    pub height: u32,
    /// Color space of this node's output data.
    pub color_space: ColorSpace,
}

/// What fusion capabilities a node has.
///
/// The graph optimizer checks these to detect fusable chains.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeCapabilities {
    /// This node can provide an analytic expression (point op fusion).
    pub analytic: bool,
    /// This node can provide an affine matrix (transform fusion).
    pub affine: bool,
    /// This node can provide a 3D CLUT (color op fusion).
    pub clut: bool,
    /// This node has a GPU compute shader.
    pub gpu: bool,
}

/// GPU shader description for a node.
///
/// The pipeline auto-composes this with the io_f32 I/O fragment to produce
/// a complete WGSL compute shader. The node only provides the body —
/// `load_pixel()`/`store_pixel()` bindings are added automatically.
#[derive(Debug, Clone)]
pub struct GpuShader {
    /// WGSL shader body (without input/output bindings — those are composed).
    pub body: String,
    /// Entry point function name.
    pub entry_point: &'static str,
    /// Workgroup size (x, y, z).
    pub workgroup_size: [u32; 3],
    /// Serialized uniform parameters (filter-specific, little-endian, 4-byte aligned).
    pub params: Vec<u8>,
    /// Optional extra storage buffers (kernel weights, LUT data, etc.).
    pub extra_buffers: Vec<Vec<u8>>,
}

/// Advisory tile size hint from a node.
///
/// The demand strategy uses these to pick optimal tile sizes.
#[derive(Debug, Clone, Copy)]
pub struct TileHint {
    /// Minimum tile size below which this node wastes significant overlap.
    pub min_efficient_tile: u32,
    /// Kernel/overlap radius in pixels (0 for point ops).
    pub overlap_radius: u32,
}

/// Upstream data provider — passed to `Node::compute` for demand-driven execution.
pub trait Upstream {
    /// Request pixel data for a region from an upstream node.
    fn request(&mut self, upstream_id: u32, rect: Rect) -> Result<Vec<f32>, PipelineError>;
}

/// The core pipeline node trait.
///
/// All operations implement this. The pipeline engine dispatches through this
/// trait without knowing what specific operation the node represents.
///
/// All pixel data is `&[f32]` — 4 channels (RGBA) per pixel, interleaved.
/// Buffer layout: `[R0, G0, B0, A0, R1, G1, B1, A1, ...]`
/// Buffer size: `width * height * 4` floats.
pub trait Node {
    /// Output metadata (dimensions, color space).
    fn info(&self) -> NodeInfo;

    /// Compute output pixels for the requested region.
    ///
    /// - `request`: the output region to compute
    /// - `upstream`: callback to pull input data from upstream nodes
    ///
    /// Returns f32 pixel data for the requested region.
    /// Buffer size: `request.width * request.height * 4` floats.
    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError>;

    /// GPU compute shader for this node (if available).
    ///
    /// Return `None` to fall back to CPU `compute()`.
    /// The pipeline auto-composes the returned shader body with io_f32
    /// bindings to produce a complete WGSL compute shader.
    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let _ = (width, height);
        None
    }

    /// IDs of upstream nodes this node depends on.
    fn upstream_ids(&self) -> Vec<u32>;

    /// Fusion capabilities of this node.
    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities::default()
    }

    /// Color space this node expects its input to be in.
    ///
    /// Default: Linear (VFX working space). Grading nodes override to Log.
    /// The graph walker auto-inserts conversion nodes at mismatches.
    fn expected_input_color_space(&self) -> ColorSpace {
        ColorSpace::Linear
    }

    /// Advisory tile size hint for the demand strategy.
    ///
    /// Neighborhood ops (blur, sharpen) should report their overlap radius
    /// so the strategy can pick tile sizes that minimize wasted overlap.
    fn tile_hint(&self) -> Option<TileHint> {
        None
    }

    /// Compute the input region needed to produce the given output region.
    ///
    /// Point ops return `output` unchanged. Spatial ops expand by kernel radius.
    /// Transform ops compute the inverse mapping bounding box.
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.clamp(bounds_w, bounds_h)
    }
}

/// Pipeline error type.
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Node not found in graph.
    NodeNotFound(u32),
    /// Computation failed.
    ComputeError(String),
    /// GPU execution failed.
    GpuError(String),
    /// Invalid parameters.
    InvalidParams(String),
    /// Buffer size mismatch.
    BufferMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::NodeNotFound(id) => write!(f, "node {id} not found"),
            PipelineError::ComputeError(msg) => write!(f, "compute error: {msg}"),
            PipelineError::GpuError(msg) => write!(f, "GPU error: {msg}"),
            PipelineError::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
            PipelineError::BufferMismatch { expected, actual } => {
                write!(f, "buffer mismatch: expected {expected} floats, got {actual}")
            }
        }
    }
}

impl std::error::Error for PipelineError {}
