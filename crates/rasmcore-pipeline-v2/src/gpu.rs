//! GPU executor trait — f32-only, no format dispatch.
//!
//! The GPU executor receives f32 pixel data and WGSL shader source,
//! creates GPU buffers, dispatches compute, and reads back results.
//! Always `array<vec4<f32>>` — no U32Packed, no BufferFormat enum.

use crate::node::{GpuShader, PipelineError};

/// GPU execution error.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU not available or initialization failed.
    NotAvailable(String),
    /// Shader compilation error.
    ShaderError(String),
    /// Execution error (dispatch, readback, etc.).
    ExecutionError(String),
    /// Buffer too large for GPU.
    BufferTooLarge { requested: usize, max: usize },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NotAvailable(msg) => write!(f, "GPU not available: {msg}"),
            GpuError::ShaderError(msg) => write!(f, "shader error: {msg}"),
            GpuError::ExecutionError(msg) => write!(f, "execution error: {msg}"),
            GpuError::BufferTooLarge { requested, max } => {
                write!(f, "buffer too large: {requested} bytes (max {max})")
            }
        }
    }
}

impl std::error::Error for GpuError {}

impl From<GpuError> for PipelineError {
    fn from(e: GpuError) -> Self {
        PipelineError::GpuError(e.to_string())
    }
}

/// GPU executor — provided by the host (browser WebGPU, CLI wgpu).
///
/// The pipeline passes f32 pixel data and composed WGSL shaders.
/// The executor manages GPU resources, buffer allocation, and dispatch.
///
/// All pixel buffers are `array<vec4<f32>>` — no format variants.
///
/// # Multi-pass execution
///
/// When `ops` contains multiple shaders, the executor chains them via
/// ping-pong buffers: output of `ops[i]` becomes input of `ops[i+1]`.
///
/// # Reduction buffers
///
/// Shaders may declare `reduction_buffers` — mutable storage buffers that
/// persist across the entire chain. The executor must:
/// 1. Before pass 1: scan all shaders for unique reduction buffer `id`s.
///    Allocate one GPU buffer per unique ID, sized from `initial_data.len()`,
///    zero-initialized from `initial_data`.
/// 2. Per pass: bind each reduction buffer at
///    `@group(0) @binding(3 + extra_buffers.len() + i)`.
///    Use `read_write` or `read` access as declared by `ReductionBuffer::read_write`.
/// 3. **Do not re-initialize** between passes — contents persist.
pub trait GpuExecutor {
    /// Execute a sequence of GPU operations on f32 pixel data.
    ///
    /// - `ops`: Sequence of shader dispatches to chain (ping-pong).
    /// - `input`: f32 pixel data (width * height * 4 floats).
    /// - `width`, `height`: Tile dimensions.
    ///
    /// Returns output f32 pixel data (same dimensions).
    fn execute(
        &self,
        ops: &[GpuShader],
        input: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, GpuError>;

    /// Pre-compile shader sources without executing them.
    ///
    /// The executor should compile and cache each shader source so that
    /// subsequent `execute()` calls hit O(1) cache lookups.
    /// Default implementation is a no-op (execution will compile on first use).
    fn prepare(&self, _shader_sources: &[String]) {}

    /// Maximum buffer size in bytes this GPU can handle.
    fn max_buffer_size(&self) -> usize {
        256 * 1024 * 1024
    }
}
