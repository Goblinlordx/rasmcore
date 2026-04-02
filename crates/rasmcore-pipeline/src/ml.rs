//! ML inference offload types — shared between WASM and native pipelines.
//!
//! Nodes that can use ML inference implement `MlCapable` and provide
//! a model reference + input tensor. The pipeline graph walker dispatches
//! ML-capable nodes to the host's ML runtime (ONNX Runtime, CoreML, etc.).
//!
//! Unlike GPU (which has a CPU fallback for every operation), ML-only nodes
//! have NO CPU fallback. If the host has no ML runtime, ML nodes fail with
//! `MlError::NotAvailable`. The graph must be validated before execution
//! via `validate_ml_requirements()`.

/// Tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    Float32,
    Float16,
    Uint8,
    Int8,
}

impl TensorDtype {
    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            TensorDtype::Float32 => 4,
            TensorDtype::Float16 => 2,
            TensorDtype::Uint8 | TensorDtype::Int8 => 1,
        }
    }
}

/// Tensor shape and type descriptor.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Dimension sizes (e.g., [1, 3, 1024, 1024] for NCHW).
    pub shape: Vec<u32>,
    /// Element data type.
    pub dtype: TensorDtype,
}

impl TensorDesc {
    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }
}

/// Reference to a model by name and version.
/// The host resolves this to a local cached model file.
#[derive(Debug, Clone)]
pub struct ModelRef {
    /// Model identifier (e.g., "rmbg-1.4", "real-esrgan-x4plus").
    pub name: String,
    /// Version or variant (e.g., "1.0.0", "fp16", "int8").
    pub version: String,
}

/// ML inference operation emitted by a node.
#[derive(Debug, Clone)]
pub struct MlOp {
    /// Model to run inference with.
    pub model: ModelRef,
    /// Raw input tensor bytes.
    pub input: Vec<u8>,
    /// Input tensor descriptor.
    pub input_desc: TensorDesc,
    /// Expected output tensor descriptor.
    pub output_desc: TensorDesc,
    /// Key-value parameters (model-specific).
    pub params: Vec<(String, String)>,
}

/// Error from ML execution.
#[derive(Debug, Clone)]
pub enum MlError {
    /// Model not found in local cache or registry.
    ModelNotFound(String),
    /// Model failed to load or compile.
    ModelLoading(String),
    /// Inference runtime error.
    InferenceError(String),
    /// Input/output tensor shape mismatch.
    ShapeMismatch(String),
    /// No ML runtime available on this host.
    NotAvailable(String),
}

impl std::fmt::Display for MlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlError::ModelNotFound(msg) => write!(f, "ML model not found: {msg}"),
            MlError::ModelLoading(msg) => write!(f, "ML model loading error: {msg}"),
            MlError::InferenceError(msg) => write!(f, "ML inference error: {msg}"),
            MlError::ShapeMismatch(msg) => write!(f, "ML shape mismatch: {msg}"),
            MlError::NotAvailable(msg) => write!(f, "ML not available: {msg}"),
        }
    }
}

impl std::error::Error for MlError {}

/// Trait for pipeline nodes that can use ML inference.
///
/// Implemented alongside `ImageNode`. When ML runtime is available,
/// the graph walker calls `ml_op()` to get the inference operation,
/// dispatches it to the host executor, then calls `process_ml_output()`
/// to convert the raw tensor output back to pixel data.
///
/// Unlike GPU, there is NO automatic CPU fallback. ML-only nodes must
/// return `Err(NotSupported)` from `compute_region()` when ML is unavailable.
pub trait MlCapable {
    /// Return an ML inference operation for this node, or None if ML
    /// is not beneficial for this input (e.g., image too small).
    fn ml_op(&self, input: &[u8], width: u32, height: u32) -> Option<MlOp>;

    /// Process the ML inference output back into pixel format.
    ///
    /// Called after `ml_execute` returns raw tensor bytes. The implementation
    /// handles model-specific post-processing (sigmoid, resize mask, etc.).
    fn process_ml_output(
        &self,
        ml_output: &[u8],
        original_input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, MlError>;
}

/// Information about a model available on the host.
#[derive(Debug, Clone)]
pub struct MlCapabilityInfo {
    /// Model reference.
    pub model: ModelRef,
    /// Input tensor requirements.
    pub input_desc: TensorDesc,
    /// Output tensor format.
    pub output_desc: TensorDesc,
    /// Execution backend name (e.g., "coreml", "cuda", "cpu").
    pub backend: String,
    /// Estimated inference time in milliseconds.
    pub estimated_ms: u32,
}

/// Host-side ML inference executor.
///
/// The pipeline calls this to run inference on a model. The host manages
/// model storage, download, caching, and runtime selection.
pub trait MlExecutor {
    /// Execute an ML inference operation.
    ///
    /// Returns raw output tensor bytes matching `op.output_desc`.
    fn execute(&self, op: &MlOp) -> Result<Vec<u8>, MlError>;

    /// Check if a model is available locally (no download needed).
    fn has_model(&self, name: &str, version: &str) -> bool;

    /// List available models and their capabilities.
    fn capabilities(&self) -> Vec<MlCapabilityInfo>;

    /// Ensure a model is available (download if needed).
    /// Blocks until the model is ready for inference.
    fn ensure_model(&self, name: &str, version: &str) -> Result<(), MlError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_desc_byte_size() {
        let desc = TensorDesc {
            shape: vec![1, 3, 224, 224],
            dtype: TensorDtype::Float32,
        };
        assert_eq!(desc.num_elements(), 1 * 3 * 224 * 224);
        assert_eq!(desc.byte_size(), 1 * 3 * 224 * 224 * 4);
    }

    #[test]
    fn tensor_dtype_sizes() {
        assert_eq!(TensorDtype::Float32.element_size(), 4);
        assert_eq!(TensorDtype::Float16.element_size(), 2);
        assert_eq!(TensorDtype::Uint8.element_size(), 1);
        assert_eq!(TensorDtype::Int8.element_size(), 1);
    }

    #[test]
    fn ml_error_display() {
        let err = MlError::NotAvailable("no ONNX runtime".into());
        assert!(err.to_string().contains("ML not available"));
        assert!(err.to_string().contains("no ONNX runtime"));
    }

    #[test]
    fn ml_op_construction() {
        let op = MlOp {
            model: ModelRef {
                name: "rmbg-1.4".into(),
                version: "fp16".into(),
            },
            input: vec![0u8; 3 * 256 * 256 * 4],
            input_desc: TensorDesc {
                shape: vec![1, 3, 256, 256],
                dtype: TensorDtype::Float32,
            },
            output_desc: TensorDesc {
                shape: vec![1, 1, 256, 256],
                dtype: TensorDtype::Float32,
            },
            params: vec![("threshold".into(), "0.5".into())],
        };
        assert_eq!(op.model.name, "rmbg-1.4");
        assert_eq!(op.input_desc.byte_size(), 3 * 256 * 256 * 4);
        assert_eq!(op.output_desc.byte_size(), 1 * 256 * 256 * 4);
    }
}
