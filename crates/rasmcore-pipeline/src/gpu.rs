//! GPU offload types — shared between WASM and native pipelines.
//!
//! Filters that can run on GPU implement `GpuCapable` and provide
//! WGSL compute shader source + parameters. The pipeline graph walker
//! batches consecutive GPU-capable nodes and dispatches them to the
//! host's GPU runtime (WebGPU in browser, wgpu in CLI).

/// Buffer element format for GPU storage buffers.
///
/// Determines how pixel data is laid out in GPU memory:
/// - `U32Packed`: Current default — each pixel is a packed `u32` (RGBA8).
///   Shaders use `unpack()`/`pack()` to convert to/from `vec4<f32>`.
/// - `F32Vec4`: High-precision — each pixel is a `vec4<f32>` (16 bytes).
///   Shaders read/write `vec4<f32>` directly, no pack/unpack round-trip.
///
/// The executor uses this to determine buffer sizes and upload/download
/// format. Shaders must match the declared format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum BufferFormat {
    /// Packed u32 RGBA8 (4 bytes/pixel). Legacy mode.
    U32Packed,
    /// vec4<f32> RGBA (16 bytes/pixel). Default for f32 pipeline.
    #[default]
    F32Vec4,
}

impl BufferFormat {
    /// Bytes per pixel for this buffer format.
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            BufferFormat::U32Packed => 4,
            BufferFormat::F32Vec4 => 16,
        }
    }
}

/// A single GPU operation in an execution chain.
///
/// Operations are executed sequentially with ping-pong buffers. `Compute`
/// dispatches a shader; `Snapshot` saves the current buffer state so later
/// ops can reference it (e.g., high-pass needs the original after blur).
#[derive(Debug, Clone, serde::Serialize)]
pub enum GpuOp {
    /// Save a read-only copy of the current ping-pong read buffer.
    ///
    /// The snapshot is mapped to the specified binding index for all
    /// subsequent `Compute` ops. Multiple snapshots can coexist at
    /// different bindings.
    Snapshot {
        /// Binding index where this snapshot will be accessible.
        binding: u32,
    },
    /// Dispatch a compute shader.
    Compute {
        /// WGSL compute shader source code (owned, composed from shared fragments + filter body).
        shader: String,
        /// Entry point function name (e.g., "main", "blur_h").
        entry_point: &'static str,
        /// Workgroup size used in the shader (x, y, z).
        /// Dispatch count = ceil(width/x, height/y, 1).
        workgroup_size: [u32; 3],
        /// Serialized uniform parameters (filter-specific, packed as bytes).
        /// Layout must match the WGSL uniform struct (little-endian, 4-byte aligned).
        params: Vec<u8>,
        /// Optional extra storage buffers (e.g., blur kernel weights, 3D LUT data).
        /// Each entry is a separate `storage<read>` buffer binding.
        extra_buffers: Vec<Vec<u8>>,
        /// Buffer element format for input/output storage buffers.
        /// Determines buffer sizing (4 vs 16 bytes/pixel) and upload/download layout.
        /// Default: `U32Packed` (backward compatible).
        #[serde(default)]
        buffer_format: BufferFormat,
    },
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
    ///
    /// `buffer_format` indicates whether the executor expects u32-packed
    /// or f32 buffers. Nodes should select the matching shader variant.
    /// Default impl delegates to the legacy 2-arg signature for backward compat.
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        let _ = (width, height);
        None
    }

    /// Return GPU operations with explicit buffer format selection.
    ///
    /// Override this to provide f32 shader variants when `buffer_format`
    /// is `F32Vec4`. The default delegates to the 2-arg `gpu_ops()`.
    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: BufferFormat) -> Option<Vec<GpuOp>> {
        let _ = buffer_format;
        self.gpu_ops(width, height)
    }
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
    /// Other error (e.g., FFI callback failure).
    Other(String),
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
            GpuError::Other(msg) => write!(f, "GPU error: {msg}"),
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
    /// - `input`: Pixel data. For `U32Packed`: width*height*4 bytes (RGBA8).
    ///   For `F32Vec4`: width*height*16 bytes (4x f32 per pixel).
    /// - `width`, `height`: Image dimensions.
    /// - `buffer_format`: Element format for ping-pong buffers.
    ///
    /// Returns the output pixel data (same format as input).
    fn execute(
        &self,
        ops: &[GpuOp],
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        self.execute_with_format(ops, input, width, height, BufferFormat::F32Vec4)
    }

    /// Execute with explicit buffer format.
    fn execute_with_format(
        &self,
        ops: &[GpuOp],
        input: &[u8],
        width: u32,
        height: u32,
        buffer_format: BufferFormat,
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
