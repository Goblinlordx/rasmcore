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
    /// Bound at `@group(0) @binding(3+)` as `storage, read`.
    pub extra_buffers: Vec<Vec<u8>>,
    /// Mutable reduction buffers that persist across passes in a `gpu_shaders()` chain.
    ///
    /// Bound at `@group(0) @binding(3 + extra_buffers.len() + i)`.
    /// Access mode per pass: `read_write` for reduction passes, `read` for apply passes.
    ///
    /// **Executor contract**: buffers with the same `id` across passes in a chain
    /// are the **same GPU allocation**. Zero-initialized from `initial_data` before
    /// the first pass that declares them. Contents persist across all subsequent passes.
    pub reduction_buffers: Vec<ReductionBuffer>,
    /// If set, the executor checks this reduction buffer after the pass completes.
    /// If all bytes are zero, remaining passes in the chain are skipped (converged).
    /// The buffer is reset to zero before the NEXT pass starts.
    ///
    /// Used for iterative algorithms (e.g., Zhang-Suen thinning) where the shader
    /// writes a non-zero value to indicate "changed" and the host loops until no
    /// changes occur.
    pub convergence_check: Option<u32>,
    /// If set, the executor dispatches this shader `count` times, writing
    /// `i = 0, 1, 2, ..., count-1` as a little-endian u32 at byte offset
    /// `param_offset` in the params buffer before each dispatch.
    ///
    /// Single compile, multiple dispatches. Used for row-by-row wavefront
    /// algorithms (e.g., seam carving DP) where each iteration processes
    /// one row/column and depends on the previous.
    pub loop_dispatch: Option<LoopDispatch>,
}

/// Loop dispatch configuration — run a shader N times with incrementing index.
#[derive(Debug, Clone)]
pub struct LoopDispatch {
    /// Number of iterations (0..count exclusive).
    pub count: u32,
    /// Byte offset in the params buffer where the iteration index (u32) is written.
    pub param_offset: usize,
}

impl GpuShader {
    /// Create a GpuShader with no extra/reduction buffers.
    pub fn new(
        body: String,
        entry_point: &'static str,
        workgroup_size: [u32; 3],
        params: Vec<u8>,
    ) -> Self {
        Self {
            body,
            entry_point,
            workgroup_size,
            params,
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }

    /// Add extra (read-only) storage buffers.
    pub fn with_extra_buffers(mut self, bufs: Vec<Vec<u8>>) -> Self {
        self.extra_buffers = bufs;
        self
    }

    /// Add reduction (mutable, persistent) buffers.
    pub fn with_reduction_buffers(mut self, bufs: Vec<ReductionBuffer>) -> Self {
        self.reduction_buffers = bufs;
        self
    }

    /// Set convergence check: after this pass, if the reduction buffer with this
    /// ID is all zeros, skip remaining passes.
    pub fn with_convergence_check(mut self, buffer_id: u32) -> Self {
        self.convergence_check = Some(buffer_id);
        self
    }

    /// Set loop dispatch: run this shader `count` times, writing iteration
    /// index (0..count) as u32 at `param_offset` in the params buffer.
    pub fn with_loop_dispatch(mut self, count: u32, param_offset: usize) -> Self {
        self.loop_dispatch = Some(LoopDispatch { count, param_offset });
        self
    }
}

/// A mutable storage buffer that persists across passes in a multi-pass GPU chain.
///
/// Used for reduction results (partial sums, histograms, min/max).
/// The executor matches buffers by `id` — same ID = same GPU allocation.
#[derive(Debug, Clone)]
pub struct ReductionBuffer {
    /// Stable ID linking this buffer across passes. Pass 1 writes (read_write),
    /// pass 3 reads (read_only) — both reference the same `id`.
    pub id: u32,
    /// Initial data (determines allocation size). Zero-initialized on first use.
    /// Subsequent passes with the same `id` ignore this field.
    pub initial_data: Vec<u8>,
    /// Whether this pass needs read_write access (`true` for reduction passes)
    /// or read-only access (`false` for apply passes that just read the result).
    pub read_write: bool,
}

/// Advisory tile size hint from a node.
///
/// The demand strategy uses these to pick optimal tile sizes.
#[derive(Debug, Clone, Copy)]
pub struct TileHint {
    /// Minimum tile size below which this node wastes significant overlap.
    pub min_efficient_tile: u32,
    /// Kernel/tile overlap in pixels (0 for point ops).
    pub tile_overlap: u32,
}

/// ACES compliance level for a node.
///
/// Used by `Graph::validate_aces()` to check if a pipeline is
/// ACES-compliant before execution. Nodes that perform linear math
/// (point ops, convolutions, matrix multiplies) are inherently compliant.
/// Nodes with sRGB-specific perceptual assumptions are non-compliant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcesCompliance {
    /// Fully ACES-compliant. Operates in linear or ACES color spaces.
    /// Linear math: brightness, blur, resize, matrix ops, compositing.
    Compliant,
    /// Operates correctly in ACES Log spaces (ACEScct, ACEScc).
    /// Grading ops: curves, CDL, lift/gamma/gain.
    Log,
    /// Not ACES-compliant. Uses sRGB perceptual assumptions,
    /// hardcoded gamma curves, or display-referred math.
    NonCompliant,
    /// Compliance not yet audited or unknown.
    Unknown,
}

impl AcesCompliance {
    /// True if this node is safe to use in an ACES pipeline.
    pub fn is_aces_safe(self) -> bool {
        matches!(self, AcesCompliance::Compliant | AcesCompliance::Log)
    }
}

/// Upstream data provider — passed to `Node::compute` for demand-driven execution.
pub trait Upstream {
    /// Request pixel data for a region from an upstream node.
    fn request(&mut self, upstream_id: u32, rect: Rect) -> Result<Vec<f32>, PipelineError>;

    /// Query upstream node info (dimensions, color space).
    /// Useful for transforms that need to know source dimensions dynamically.
    fn info(&self, upstream_id: u32) -> Result<NodeInfo, PipelineError>;
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

    /// GPU compute shaders for multi-pass filters.
    ///
    /// Multi-pass filters (histogram→LUT→apply, separable blur H+V,
    /// retinex multi-scale, etc.) return multiple shaders that execute
    /// sequentially via ping-pong buffers. Output of shader[i] feeds
    /// into shader[i+1].
    ///
    /// Default: wraps the single `gpu_shader()` result.
    /// Override for multi-pass operations.
    fn gpu_shaders(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        self.gpu_shader(width, height).map(|s| vec![s])
    }

    /// IDs of upstream nodes this node depends on.
    fn upstream_ids(&self) -> Vec<u32>;

    /// Fusion capabilities of this node.
    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities::default()
    }

    /// Analytic expression for this node (point op fusion).
    ///
    /// If this node is a per-channel point operation expressible as an
    /// algebraic expression tree, return it. The fusion optimizer composes
    /// consecutive analytic expressions and constant-folds them.
    ///
    /// Returns [R, G, B] expressions — one per channel.
    /// Default: None (not an analytic point op).
    fn analytic_expression_per_channel(&self) -> Option<[crate::ops::PointOpExpr; 3]> {
        None
    }

    /// ACES compliance level of this node.
    ///
    /// Default: Unknown. Override to declare compliance status.
    /// Used by `Graph::validate_aces()` to check pipeline integrity.
    fn aces_compliance(&self) -> AcesCompliance {
        AcesCompliance::Unknown
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
    /// Neighborhood ops (blur, sharpen) should report their tile overlap
    /// so the strategy can pick tile sizes that minimize wasted overlap.
    fn tile_hint(&self) -> Option<TileHint> {
        None
    }

    /// Compute the input region needed to produce the given output region.
    ///
    /// Most ops can compute this statically from params alone:
    /// - Point ops: `Exact` — output unchanged
    /// - Spatial ops: `Exact` — output expanded by kernel radius
    /// - Transforms: `Exact` — inverse mapping bounding box
    ///
    /// Rare ops that depend on pixel data (displacement map, content-aware):
    /// - `FullImage` — the pipeline materializes the full upstream before this node
    ///
    /// `FullImage` nodes become **tile barriers** — tiling pauses, full image is
    /// fetched, then tiling resumes downstream. This is a last resort.
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
    }

    /// Build a 3D CLUT representing this node's color operation for fusion.
    ///
    /// If provided, the fusion optimizer can compose consecutive CLUT-capable
    /// nodes into a single 3D LUT pass. Override for color grading operations
    /// (curves, CDL, LGG, hue-vs-sat, tone mapping, etc.).
    /// Default: None (not CLUT-fusable).
    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        None
    }

    /// If this node is an LMT node, return the underlying Lmt value.
    /// Used by the fusion optimizer to flatten Chain LMTs into individual nodes.
    /// Default: None (not an LMT node).
    fn as_lmt(&self) -> Option<&crate::lmt::Lmt> {
        None
    }

    /// Run analysis on input pixels, producing a typed result.
    ///
    /// Analysis nodes override this to return `Some(result)`. The staged
    /// pipeline calls this via `graph.analyze_node()` after materializing
    /// the upstream pixels.
    ///
    /// Default: None (not an analysis node).
    fn analyze(
        &self,
        _input: &[f32],
        _width: u32,
        _height: u32,
    ) -> Option<Result<crate::staged::AnalysisResult, PipelineError>> {
        None
    }

    /// Whether this node is an analysis node (has an analyze() implementation).
    fn is_analysis_node(&self) -> bool {
        false
    }

    // ─── Cross-Node Analysis Buffer Protocol ────────────────────────────

    /// Analysis buffers this node produces (for cross-node GPU sharing).
    ///
    /// Override to declare reduction buffers that downstream nodes can consume.
    /// The graph walker uses these declarations to merge analysis + render
    /// shader chains into a single GPU submit with zero CPU readback.
    ///
    /// Default: empty (not an analysis producer).
    fn analysis_outputs(&self) -> &[crate::analysis_buffer::AnalysisBufferDecl] {
        &[]
    }

    /// Analysis buffers this node consumes from upstream analysis nodes.
    ///
    /// Override to declare which upstream analysis buffers this node reads.
    /// The logical IDs must match `AnalysisBufferDecl::logical_id` from an
    /// upstream producer. The negotiation step resolves logical → global IDs.
    ///
    /// Default: empty (not an analysis consumer).
    fn analysis_inputs(&self) -> &[crate::analysis_buffer::AnalysisBufferRef] {
        &[]
    }

    /// GPU shaders with cross-node analysis buffer context.
    ///
    /// When the graph walker detects analysis→render pairs, it calls this
    /// method instead of `gpu_shaders()`. The `mapping` provides resolved
    /// buffer IDs that are globally unique across the merged chain.
    ///
    /// Nodes that consume analysis buffers override this to substitute
    /// their internal logical buffer IDs with the resolved IDs from the mapping.
    ///
    /// Default: delegates to `gpu_shaders()` (ignores context).
    fn gpu_shaders_with_context(
        &self,
        width: u32,
        height: u32,
        _mapping: &crate::analysis_buffer::NodeBufferMapping,
    ) -> Option<Vec<GpuShader>> {
        self.gpu_shaders(width, height)
    }
}

/// How a node estimates its required input region.
///
/// The pipeline's tile scheduler uses this to pre-plan tile fetches
/// BEFORE any pixel execution. It must be computable from (output_rect,
/// params, upstream_dims) alone — no pixel data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputRectEstimate {
    /// Exact region needed — computable from params alone.
    /// This is the common case (99% of operations).
    Exact(Rect),

    /// Full upstream image needed — cannot estimate statically.
    /// This node becomes a tile barrier: the pipeline materializes
    /// the full upstream before executing this node, then resumes
    /// tiling downstream.
    ///
    /// Used by: displacement map, content-aware seam carving, mesh warp
    /// with data-driven control points.
    FullImage,

    /// Conservative upper bound — the node needs at most this much,
    /// but may request less during compute(). The pipeline pre-fetches
    /// the upper bound; the node clips internally.
    UpperBound(Rect),
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
