//! GPU offload types — shared between WASM and native pipelines.
//!
//! Filters that can run on GPU implement `GpuCapable` and provide
//! WGSL compute shader source + parameters. The pipeline graph walker
//! batches consecutive GPU-capable nodes and dispatches them to the
//! host's GPU runtime (WebGPU in browser, wgpu in CLI).

/// A single GPU compute operation (one shader dispatch).
#[derive(Debug, Clone)]
pub struct GpuOp {
    /// WGSL compute shader source code.
    pub shader: &'static str,
    /// Entry point function name (e.g., "main", "blur_h").
    pub entry_point: &'static str,
    /// Workgroup size used in the shader (x, y, z).
    /// Dispatch count = ceil(width/x, height/y, 1).
    pub workgroup_size: [u32; 3],
    /// Serialized uniform parameters (filter-specific, packed as bytes).
    /// Layout must match the WGSL uniform struct (little-endian, 4-byte aligned).
    pub params: Vec<u8>,
    /// Optional extra storage buffers (e.g., blur kernel weights, 3D LUT data).
    /// Each entry is a separate `storage<read>` buffer binding.
    pub extra_buffers: Vec<Vec<u8>>,
}

/// Trait for pipeline nodes that can execute on GPU.
///
/// Implemented alongside `ImageNode`. When GPU is available, the graph
/// walker calls `gpu_ops()` instead of `compute_region()`.
///
/// Return `None` if this node cannot run on GPU for its current config
/// (e.g., unsupported pixel format, extreme parameters). The node
/// becomes a batch boundary and falls back to CPU.
pub trait GpuCapable {
    /// Return GPU operations for this node's current configuration.
    ///
    /// A single node may produce multiple ops (e.g., separable blur
    /// returns horizontal + vertical passes). All ops within a node
    /// are dispatched sequentially with ping-pong buffers.
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>>;
}

/// Error from GPU execution.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU not available or initialization failed.
    NotAvailable(String),
    /// Shader compilation error.
    ShaderError(String),
    /// Execution error (dispatch, readback, etc.).
    ExecutionError(String),
    /// Buffer too large for GPU (exceeds max_buffer_size).
    BufferTooLarge { requested: usize, max: usize },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NotAvailable(msg) => write!(f, "GPU not available: {msg}"),
            GpuError::ShaderError(msg) => write!(f, "GPU shader error: {msg}"),
            GpuError::ExecutionError(msg) => write!(f, "GPU execution error: {msg}"),
            GpuError::BufferTooLarge { requested, max } => {
                write!(f, "GPU buffer too large: {requested} bytes (max {max})")
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU execution handler — provided by the host (browser WebGPU, CLI wgpu).
///
/// The pipeline calls this to dispatch a batch of GPU operations.
/// All ops in the batch are chained: output of op[i] = input of op[i+1].
/// Intermediate buffers stay in GPU memory (no round-trips).
pub trait GpuExecutor {
    /// Execute a batch of GPU operations on pixel data.
    ///
    /// - `ops`: Sequence of compute shader dispatches to chain.
    /// - `input`: RGBA8 pixel data (width * height * 4 bytes).
    /// - `width`, `height`: Image dimensions.
    ///
    /// Returns the output pixel data (same dimensions, RGBA8).
    fn execute(
        &self,
        ops: &[GpuOp],
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError>;

    /// Maximum buffer size in bytes that this GPU can handle.
    /// Used to decide between full-image and tiled GPU dispatch.
    /// Default: 256MB (covers up to ~8K images).
    fn max_buffer_size(&self) -> usize {
        256 * 1024 * 1024
    }
}

/// GPU dispatch configuration for the graph walker.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Maximum image size (in pixels) for full-image GPU dispatch.
    /// Above this, the pipeline tiles at GPU-friendly sizes.
    /// Default: 4096 * 4096 = 16M pixels (64MB at RGBA8).
    pub max_full_image_pixels: u64,
    /// Tile size for GPU dispatch when image exceeds max_full_image_pixels.
    /// Much larger than CPU tile size (4096 vs 512).
    pub gpu_tile_size: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_full_image_pixels: 4096 * 4096,
            gpu_tile_size: 4096,
        }
    }
}
