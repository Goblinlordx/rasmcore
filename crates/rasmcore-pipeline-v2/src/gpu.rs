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
/// All buffers are `array<vec4<f32>>` — no format variants.
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

    /// Maximum buffer size in bytes this GPU can handle.
    fn max_buffer_size(&self) -> usize {
        256 * 1024 * 1024
    }
}
