# ML Pipeline Integration — Research Findings

## Date: 2026-04-07
## Status: Complete — architecture validated via PoC, implementation tracks created

---

## 1. Existing Infrastructure Audit

### wit/core/ml.wit
- Stub exists with ml-execute, ml-capabilities, model-ref, tensor-desc, ml-error
- tensor I/O uses `list<u8>` (correct for WIT boundary — bytes are the universal format)
- ml-op uses string key-value params → should use typed param-descriptor (reuse from pipeline.wit)
- ml-capability-info lacks: display-name, category, output-kind, input-spec, typed params
- NOT imported by pipeline world.wit (only exports pipeline-v2)

### Pipeline GPU architecture
- gpu-plan / gpu-stage / stage-input system supports multi-stage GPU chaining
- stage-input::prior-stage(string) allows GPU buffer reuse between stages
- inject-gpu-result takes list<f32> (forces CPU readback) — needs GPU buffer path for ML
- Host dispatches shaders via WebGPU/wgpu, guest produces plans

---

## 2. GPU-ML Interop Findings

### ONNX Runtime Web (validated via PoC)
- **WebGPU EP works** on macOS M4 Chrome — session creates in ~340ms for 64MB model
- ORT manages its own GPUDevice internally (bug #26107 blocks injecting custom device)
- `Tensor.fromGpuBuffer()` and `preferredOutputLocation: 'gpu-buffer'` exist but require ORT's device
- Practical: 1-copy between ORT device and pipeline device (host handles internally)

### WebNN (partially validated)
- `navigator.ml` available in Chrome behind `chrome://flags` → "Web Machine Learning Neural Network API"
- `createContext` requires options object: `{ deviceType: 'gpu' }` (not string argument)
- macOS M4: API present but GPU/NPU/CPU context creation status varies
- ORT can use WebNN as execution provider: `{ name: 'webnn', deviceType: 'gpu' }`
- NOT production-ready — flag-only, limited platform support

### Native (ort crate)
- `ort` 2.0.0-rc.12 on crates.io — modern Rust ONNX Runtime wrapper
- `load-dynamic` feature loads libonnxruntime at runtime (no static linking needed)
- macOS: `brew install onnxruntime` + `ORT_DYLIB_PATH` env
- Auto-detects CoreML EP (macOS), CUDA EP (NVIDIA), CPU fallback
- No direct wgpu↔CUDA buffer interop — CPU roundtrip required for native GPU path

### Zero-copy assessment
- **Web**: Best-effort via WebNN MLTensor exportableToGPU (spec'd, not production)
- **Practical web**: 1-copy between ORT WebGPU device and pipeline WebGPU device
- **Native**: CPU roundtrip (GPU→CPU→GPU) unless same CUDA context
- **Architecture**: Host handles copy transparently — pipeline doesn't care

---

## 3. Model Survey

### Super-resolution
| Model | Size | Speed (GPU) | ONNX Available | License |
|---|---|---|---|---|
| Real-ESRGAN x4plus | 64MB | ~200-500ms/tile | Yes (HuggingFace) | BSD-3 |
| Real-ESRGAN anime_6B | 17MB | ~100-300ms/tile | Yes | BSD-3 |
| ESPCN x4 | 100KB | ~5-20ms | Yes (ONNX Model Zoo) | MIT |
| SwinIR / Swin2SR | 48MB | ~1-3s | Partial | Apache 2.0 |

### Segmentation / Background removal
| Model | Size | Input | ONNX Available | License |
|---|---|---|---|---|
| RMBG-1.4 (BRIA) | 176MB | 1024x1024 resize | Yes (HuggingFace) | MIT |

### Depth estimation
| Model | Size | Input | ONNX Available | License |
|---|---|---|---|---|
| MiDaS v2.1 small | 50MB | 384x384 resize | Yes (HuggingFace) | MIT |

### Key finding: Qualcomm Real-ESRGAN ONNX is fixed 128x128 input (not dynamic)
- Requires tiling for any image larger than 128x128
- Tiling with overlap eliminates seam artifacts
- 256-512px tiles optimal for GPU utilization; 128px works but underutilizes

---

## 4. Architecture Design

### Host-as-capability-provider
- WASM guest never bundles/loads models
- Host declares capabilities via ml-capabilities() at runtime
- Guest calls ml-execute(tile_tensor) per tile
- Separate packages: @rasmcore/ml (browser), rasmcore-ml crate (native)
- Model definitions (metadata+URL) ship with package, weights downloaded on demand

### Model input spec (tileable vs full-image)
```
enum ml-tile-mode { tileable, full-image }

record ml-input-spec {
  tile-mode: ml-tile-mode,
  preferred-size: option<(u32, u32)>,  // tileable: optimal tile size
  min-size: option<(u32, u32)>,        // tileable: minimum usable tile
  target-size: option<(u32, u32)>,     // full-image: resize target
  overlap: u32,                         // pixels for seam blending
  padding: ml-padding-mode,             // mirror/zero/clamp
}
```

### Tiling ownership
- **Pipeline** (MlNode) handles tiling, overlap blending, stitching
- **Host** just runs single-tile inference via ml-execute
- **Concurrency** is host-determined (GPU memory, device class) — NOT in model spec
- **Variant selection** (preview vs quality) is application concern — NOT pipeline concern

### GPU-first ML
- ml-execute returns GPU buffer handle when GPU path active
- Output feeds directly into next GPU stage (zero-copy when possible)
- Host handles internal copy if inference GPU != pipeline GPU
- ML stage integrates into multi-gpu-plan as another stage-input variant

### SDK registration API
- Core SDK exposes registerMlProvider(provider: MlProvider)
- MlProvider: { execute(op), capabilities(), dispose() }
- @rasmcore/ml is separate npm package (not bundled)
- Individual models are tree-shakeable imports
- Native: rasmcore-ml crate with same MlProvider trait

---

## 5. Implementation Plan

### Track chain (in dependency order):
1. **ml-wit-interface** — Revise ml.wit with ml-tile-mode, ml-input-spec, ml-output-kind, param-descriptor reuse. Wire as pipeline import. Add apply-ml() to pipeline.wit.
2. **ml-graph-node** — MlNode implementing Node trait. Tensor conversion, tiling with overlap, GPU plan ml-stage integration.
3. **ml-sdk-host** — @rasmcore/ml browser package + rasmcore-ml native crate. Backend detection, model lifecycle, GPU buffer returns, 3 reference models.

### Reference models for initial release:
1. Real-ESRGAN x4plus (upscale, tileable, image output)
2. RMBG-1.4 (background removal, full-image, mask output)
3. MiDaS v2.1 small (depth estimation, full-image, mask output)

---

## 6. Open Questions (for implementation phase)

- Exact WebNN EP behavior on macOS with Real-ESRGAN (needs testing with correct model)
- ORT WebGPU shared device bug #26107 status — may be fixed by implementation time
- Optimal tile size for M-series GPUs (128 vs 256 vs 512)
- Whether RMBG-1.4 ONNX needs special preprocessing (normalization values)
- Progressive tile display (show tiles as they complete vs wait for all)
