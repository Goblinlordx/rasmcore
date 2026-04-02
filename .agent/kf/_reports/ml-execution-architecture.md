# ML Execution System Architecture

**Date:** 2026-04-02
**Track:** ml-plugin-architecture_20260402061336Z
**Status:** Design complete, ready for implementation

## 1. Overview

ML execution in rasmcore mirrors the GPU execution pattern exactly:

| Aspect | GPU | ML |
|--------|-----|-----|
| **Node declares** | `GpuCapable` trait | `MlCapable` trait |
| **Node emits** | `Vec<GpuOp>` (shader + params) | `MlOp` (model ref + tensor) |
| **Host executes** | `GpuExecutor` (wgpu/WebGPU) | `MlExecutor` (ONNX Runtime) |
| **WIT interface** | `core/gpu.wit` | `core/ml.wit` |
| **Result** | Pixel buffer (same dimensions) | Tensor (may change dims) |
| **Fallback** | CPU path (always available) | None (ML-only features) |

Key difference: ML models are large (5-500MB) and fetched on demand as remote bundles, not compiled in. The host manages model storage, download, and caching.

## 2. WIT Interface (`core/ml.wit`)

```wit
interface ml {
    /// Reference to a model by name and version.
    /// The host resolves this to a local file path.
    record model-ref {
        name: string,        // e.g., "rmbg-1.4", "real-esrgan-x4plus"
        version: string,     // e.g., "1.0.0", "fp16"
    }

    /// Tensor descriptor for input/output specification.
    record tensor-desc {
        shape: list<u32>,    // e.g., [1, 3, 320, 320]
        dtype: tensor-dtype,
    }

    /// Supported tensor element types.
    enum tensor-dtype {
        float32,
        float16,
        uint8,
        int8,
    }

    /// ML inference operation emitted by a node.
    record ml-op {
        model: model-ref,
        input: list<u8>,             // Raw tensor bytes (layout per dtype)
        input-desc: tensor-desc,     // Input tensor shape + type
        output-desc: tensor-desc,    // Expected output shape + type
        params: list<tuple<string, string>>,  // Key-value config (e.g., threshold)
    }

    /// Error types for ML execution.
    variant ml-error {
        model-not-found(string),     // Model not in local cache
        model-loading(string),       // Model failed to load/compile
        inference-error(string),     // Runtime inference failure
        shape-mismatch(string),      // Input/output shape validation
        not-available(string),       // No ML runtime on this host
    }

    /// Execute an ML inference operation.
    ml-execute: func(op: ml-op) -> result<list<u8>, ml-error>;

    /// Information about a model available on this host.
    record ml-capability-info {
        model: model-ref,
        input-desc: tensor-desc,
        output-desc: tensor-desc,
        backend: string,             // e.g., "coreml", "cuda", "cpu"
        estimated-ms: u32,           // Estimated inference time
    }

    /// Query available ML models and their capabilities.
    ml-capabilities: func() -> list<ml-capability-info>;
}
```

## 3. Rust Traits

### MlCapable (Node-Side)

```rust
// In crates/rasmcore-pipeline/src/ml.rs

/// ML inference operation emitted by a node.
#[derive(Debug, Clone)]
pub struct MlOp {
    pub model_name: String,
    pub model_version: String,
    pub input: Vec<u8>,
    pub input_shape: Vec<u32>,
    pub input_dtype: TensorDtype,
    pub output_shape: Vec<u32>,
    pub output_dtype: TensorDtype,
    pub params: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    Float32,
    Float16,
    Uint8,
    Int8,
}

/// Trait for nodes that can use ML inference.
pub trait MlCapable {
    /// Return an ML operation for this node, or None if ML is not
    /// beneficial for this input (e.g., image too small).
    fn ml_op(&self, input: &[u8], width: u32, height: u32) -> Option<MlOp>;

    /// Process the ML inference output back into pixel format.
    /// Called after ml_execute returns raw tensor bytes.
    fn process_ml_output(
        &self,
        ml_output: &[u8],
        original_input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, MlError>;
}
```

**Design note:** Unlike `GpuCapable` which returns opaque pixel buffers, `MlCapable` has a `process_ml_output` step because ML models output tensors (masks, upscaled images) that need post-processing to become pipeline-compatible pixel buffers.

### MlExecutor (Host-Side)

```rust
/// Host-side ML inference executor.
pub trait MlExecutor {
    /// Execute an ML inference operation.
    fn execute(&self, op: &MlOp) -> Result<Vec<u8>, MlError>;

    /// Check if a model is available locally.
    fn has_model(&self, name: &str, version: &str) -> bool;

    /// List available models and their capabilities.
    fn capabilities(&self) -> Vec<MlCapabilityInfo>;

    /// Ensure a model is available (download if needed).
    /// Returns Ok when the model is ready for inference.
    fn ensure_model(&self, name: &str, version: &str) -> Result<(), MlError>;
}
```

## 4. Pipeline Integration

### Graph Dispatch (mirrors GPU dispatch in `request_region`)

```rust
// In NodeGraph::request_region(), after GPU dispatch, before CPU fallback:

// Try ML dispatch
let ml_dispatch = self.ml_executor.clone().and_then(|executor| {
    let ml_capable = self.ml_nodes.get(node_id as usize)?.as_ref()?;
    let upstream_id = self.nodes[node_id as usize].upstream_id()?;
    Some((executor, ml_capable, upstream_id))
});

if let Some((executor, ml_capable, upstream_id)) = ml_dispatch {
    let full_rect = Rect::new(0, 0, info.width, info.height);
    let input = self.request_region(upstream_id, full_rect)?;

    if let Some(ml_op) = ml_capable.ml_op(&input, info.width, info.height) {
        match executor.execute(&ml_op) {
            Ok(ml_output) => {
                let pixels = ml_capable.process_ml_output(
                    &ml_output, &input, info.width, info.height
                )?;
                // Crop if needed, accumulate, return
                self.accumulate_tile(node_id, request, &pixels, bpp);
                return Ok(pixels);
            }
            Err(_) => {
                // ML failed — fall through to CPU (if CPU path exists)
                // Note: unlike GPU, ML-only nodes have no CPU fallback
            }
        }
    }
}
```

### Registration (mirrors GPU registration)

```rust
impl NodeGraph {
    pub fn register_ml(&mut self, node_id: u32, ml: Box<dyn MlCapable>) { ... }
    pub fn set_ml_executor(&mut self, executor: Rc<dyn MlExecutor>) { ... }
    pub fn has_ml(&self, node_id: u32) -> bool { ... }

    /// Validate that all ML-required nodes have an executor available.
    /// Call before execution — unlike GPU (which silently falls back to CPU),
    /// ML-only nodes have NO CPU fallback and will fail at request_region.
    pub fn validate_ml_requirements(&self) -> Result<(), ValidationError> {
        let has_executor = self.ml_executor.is_some();
        for (id, ml_node) in self.ml_nodes.iter().enumerate() {
            if ml_node.is_some() && !has_executor {
                return Err(ValidationError {
                    message: format!(
                        "node {id} requires ML runtime but no MlExecutor is set. \
                         Call set_ml_executor() before execution."
                    ),
                    node_id: Some(id as u32),
                    upstream_id: None,
                });
            }
        }
        Ok(())
    }
}
```

**Critical design point:** `validate_ml_requirements()` must be called before pipeline
execution (e.g., in `sink::write()` or by the caller). GPU nodes silently fall back to CPU,
but ML nodes have no fallback — failing deep inside `request_region` with a confusing
`NotSupported` error is unacceptable. Fail fast at validation time with a clear message
about the missing executor.

The existing `validate()` method checks graph topology. `validate_ml_requirements()` is
a separate check because ML executor availability is a runtime concern (host may or may
not have ONNX Runtime installed), not a graph structure concern.

## 5. Remote Model Bundles

### Bundle Format

```
model-name-version/
  manifest.json       # Model metadata
  model.onnx          # ONNX model file
```

### Manifest Schema

```json
{
  "name": "rmbg-1.4",
  "version": "1.0.0",
  "format": "onnx",
  "sha256": "abc123...",
  "size_bytes": 176000000,
  "license": "Apache-2.0",
  "input": {
    "shape": [1, 3, 1024, 1024],
    "dtype": "float32",
    "preprocessing": "normalize_imagenet"
  },
  "output": {
    "shape": [1, 1, 1024, 1024],
    "dtype": "float32",
    "postprocessing": "sigmoid_to_mask"
  },
  "variants": {
    "fp16": { "file": "model_fp16.onnx", "sha256": "def456...", "size_bytes": 88000000 },
    "int8": { "file": "model_int8.onnx", "sha256": "ghi789...", "size_bytes": 44000000 }
  }
}
```

### Loading Protocol

1. **Discovery:** Node requests model by `(name, version)`
2. **Cache check:** Host checks `~/.rasmcore/models/{name}/{version}/`
3. **Download:** If missing, fetch from registry URL (configurable)
4. **Integrity:** Verify SHA-256 before loading
5. **Compile:** ONNX Runtime compiles/optimizes for local EP on first load
6. **Cache compiled:** ORT caches compiled model for subsequent loads

### Model Storage

```
~/.rasmcore/models/
  rmbg-1.4/
    1.0.0/
      manifest.json
      model.onnx          # fp32 (default)
      model_fp16.onnx      # fp16 variant
  real-esrgan-x4plus/
    1.0.0/
      manifest.json
      model.onnx
```

## 6. Host Executor Implementations

### Platform Matrix

| Platform | Primary EP | Fallback EP | Rust Crate |
|----------|-----------|-------------|------------|
| macOS (Apple Silicon) | CoreML | CPU | `ort` + `coreml` feature |
| macOS (Intel) | CPU | - | `ort` |
| Linux (NVIDIA) | CUDA | CPU | `ort` + `cuda` feature |
| Linux (AMD) | ROCm | CPU | `ort` + `rocm` feature |
| Linux (CPU) | CPU | - | `ort` |
| Windows | DirectML | CPU | `ort` + `directml` feature |
| Android (Qualcomm) | QNN | XNNPACK | `ort` + `qnn` feature |
| Android (other) | XNNPACK | CPU | `ort` + `xnnpack` feature |
| Browser | WebGPU (ORT-Web) | WASM (ORT-Web) | N/A (JS runtime) |

### Native Executor (Rust, via `ort` crate)

```rust
// In crates/rasmcore-cli/src/ml_executor.rs (future)

pub struct OrtMlExecutor {
    session_cache: RefCell<HashMap<String, ort::Session>>,
    model_dir: PathBuf,
}

impl MlExecutor for OrtMlExecutor {
    fn execute(&self, op: &MlOp) -> Result<Vec<u8>, MlError> {
        let session = self.get_or_load_session(&op.model_name, &op.model_version)?;
        let input_tensor = ndarray::ArrayD::from_shape_vec(
            op.input_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            bytemuck::cast_slice::<u8, f32>(&op.input).to_vec(),
        ).map_err(|e| MlError::ShapeMismatch(e.to_string()))?;

        let outputs = session.run(ort::inputs![input_tensor]?)
            .map_err(|e| MlError::InferenceError(e.to_string()))?;

        let output = outputs[0].try_extract_raw_tensor::<f32>()
            .map_err(|e| MlError::InferenceError(e.to_string()))?;

        Ok(bytemuck::cast_slice::<f32, u8>(output.1).to_vec())
    }
}
```

### Browser Executor (WASM host, via JS interop)

The WASM component calls `ml-execute` through the WIT interface. The JavaScript host implements it using `onnxruntime-web`:

```javascript
// Host-side JS (conceptual)
async function mlExecute(op) {
    const session = await ort.InferenceSession.create(
        modelUrl(op.model.name, op.model.version),
        { executionProviders: ['webgpu', 'wasm'] }
    );
    const input = new ort.Tensor(op.inputDesc.dtype, op.input, op.inputDesc.shape);
    const results = await session.run({ input });
    return results.output.data;
}
```

## 7. Concrete Example: Background Removal Node

```rust
pub struct BackgroundRemovalNode {
    upstream: u32,
    info: ImageInfo,
    model_version: String,  // "fp32", "fp16", "int8"
}

impl ImageNode for BackgroundRemovalNode {
    fn info(&self) -> ImageInfo {
        self.info.clone() // Same dimensions, RGBA8 output
    }

    fn compute_region(&self, request: Rect, upstream_fn: ...) -> Result<Vec<u8>> {
        // CPU fallback: not available for ML-only features
        Err(ImageError::NotSupported(
            "background removal requires ML runtime".into()
        ))
    }
}

impl MlCapable for BackgroundRemovalNode {
    fn ml_op(&self, input: &[u8], width: u32, height: u32) -> Option<MlOp> {
        // Preprocess: resize to model input size, normalize
        let resized = resize_to(input, width, height, 1024, 1024);
        let normalized = normalize_imagenet(&resized);

        Some(MlOp {
            model_name: "rmbg-1.4".into(),
            model_version: self.model_version.clone(),
            input: normalized,
            input_shape: vec![1, 3, 1024, 1024],
            input_dtype: TensorDtype::Float32,
            output_shape: vec![1, 1, 1024, 1024],
            output_dtype: TensorDtype::Float32,
            params: vec![],
        })
    }

    fn process_ml_output(
        &self, ml_output: &[u8], original: &[u8], width: u32, height: u32,
    ) -> Result<Vec<u8>> {
        // Postprocess: sigmoid -> resize mask to original size -> apply as alpha
        let mask = sigmoid_to_mask(ml_output, 1024, 1024);
        let resized_mask = resize_to_gray(&mask, 1024, 1024, width, height);
        apply_mask_as_alpha(original, &resized_mask, width, height)
    }
}
```

## 8. Security Considerations

| Risk | Mitigation |
|------|------------|
| **Model integrity** | SHA-256 verification before loading. Reject models that fail hash check. |
| **Resource limits** | Max model size configurable (default 500MB). Max inference time (timeout). Max memory per session. |
| **Sandboxing** | ONNX Runtime runs in host process, not in WASM sandbox. Models can only access I/O tensors, not filesystem. |
| **Supply chain** | Model registry must use HTTPS. Manifest includes license field for compliance. |
| **Denial of service** | Inference timeout (default 30s). Queue depth limit for concurrent requests. |

## 9. Implementation Tracks

This architecture enables these implementation tracks (in order):

1. **ml-execution-impl** (already created) — Add `MlOp`, `MlCapable`, `MlExecutor` traits + WIT interface + graph integration
2. **ml-ort-executor** — Native `OrtMlExecutor` using `ort` crate with platform-specific EPs
3. **ml-background-removal** — First concrete ML node (RMBG-1.4)
4. **ml-super-resolution** — Real-ESRGAN upscaling node
5. **ml-model-registry** — CDN/registry infrastructure for remote model bundles
6. **ml-browser-executor** — JavaScript host executor using onnxruntime-web

## 10. Key Design Decisions

1. **ONNX as universal model format** — All models distributed as ONNX. Host picks the best execution provider per platform. No model format fragmentation.

2. **Host-provided capability** — ML runs in the host, not in WASM. Same pattern as GPU. WASM component emits an operation descriptor; host decides how to execute.

3. **No CPU fallback for ML features** — Unlike GPU (where every filter has a CPU path), ML-only features (background removal, super-resolution) have no CPU alternative. The node's `compute_region` returns `NotSupported` if ML is unavailable.

4. **`process_ml_output` two-step** — ML models output tensors (masks, upscaled images) that need post-processing (sigmoid, resize, alpha application). This post-processing is node-specific, not model-specific, so it lives in the `MlCapable` impl.

5. **Remote bundles with local caching** — Models are too large to compile in. Downloaded on first use, cached locally, verified by SHA-256. Host manages storage.

6. **`ort` crate for all native backends** — Single Rust dependency covers CUDA, CoreML, DirectML, QNN, ROCm, CPU. Platform-specific EPs selected at runtime via Cargo features.
