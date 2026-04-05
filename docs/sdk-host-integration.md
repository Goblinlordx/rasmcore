# SDK Host Integration Guide

rasmcore ships as a WASM Component (WASI 0.2). The pipeline handles all image processing internally — filters, codecs, fusion, color conversion — but certain capabilities require host cooperation. This guide documents what works out of the box and what the host must provide.

## What Works Without Host Wiring

These features require zero host setup beyond loading the WASM module:

| Capability | Details |
|-----------|---------|
| All 70+ filters | CPU f32 execution, analytic fusion, CLUT fusion |
| All codecs | JPEG, PNG, WebP, TIFF, EXR, HDR, QOI, BMP, GIF, FITS, ICO, PNM, TGA, DDS |
| Color space conversion | sRGB, Linear, ACEScg, ACEScct, ACEScc, Display P3, Rec.709, Rec.2020 |
| View transforms | sRGB, Linear, Rec.709 output encoding |
| Demand-driven tiling | Bounded memory execution for large images |
| Layer cache | Content-addressed caching with optional quantization (Q16/Q8) |
| Pipeline tracing | Opt-in timing diagnostics for fusion, dispatch, encode |

### Minimal Example (Browser)

```typescript
import { pipelineV2 } from './rasmcore-v2-image.js';

const pipe = new pipelineV2.ImagePipelineV2();
const source = pipe.read(imageBytes, undefined);
const bright = pipe.applyFilter(source, 'brightness', serializeParams({ amount: 0.2 }));
const png = pipe.write(bright, 'png', 85);
// png is a Uint8Array — done. No GPU, no cache, no host wiring needed.
```

### Minimal Example (Native Rust via wasmtime)

```rust
let engine = Engine::default();
let component = Component::from_file(&engine, "rasmcore_v2_wasm.wasm")?;
let mut store = Store::new(&engine, ());
let (bindings, _) = ImageProcessor::instantiate(&mut store, &component, &linker)?;

let pipe = bindings.rasmcore_v2_image_pipeline_v2().call_constructor(&mut store)?;
let source = pipe.call_read(&mut store, &image_bytes, None)?;
let adjusted = pipe.call_apply_filter(&mut store, source, "exposure", &params)?;
let jpeg = pipe.call_write(&mut store, adjusted, "jpeg", Some(90))?;
```

---

## Optional: GPU Acceleration

GPU acceleration is the primary performance path. The pipeline generates WGSL shaders and returns them to the host — the host compiles and dispatches via WebGPU (browser) or wgpu (native).

### Architecture

```
Pipeline (WASM)                    Host (Browser/Native)
  |                                  |
  |-- render-gpu-plan(node) -------->|
  |   returns: GpuPlan {             |
  |     shaders: GpuShader[],        |
  |     input_pixels: f32[],         |-- compile shaders
  |     width, height                |-- create buffers
  |   }                              |-- dispatch compute
  |                                  |-- readback result
  |<-- inject-gpu-result(node, px) --|
  |                                  |
  |-- write(node, "png", 90) ------->| (uses cached GPU result)
```

The pipeline is GPU-agnostic. It splits work at the shader boundary:
- **Pipeline owns**: graph topology, fusion, shader generation, caching
- **Host owns**: GPU device, shader compilation, buffer management, dispatch

### GpuShader Contract

Each shader in the plan contains:

```typescript
interface GpuShader {
  source: string;        // Complete WGSL (io_f32 bindings + body)
  entryPoint: string;    // Compute entry point name
  workgroupX: number;    // Dispatch workgroup size
  workgroupY: number;
  workgroupZ: number;
  params: Uint8Array;    // Uniform buffer data (little-endian, 4-byte aligned)
  extraBuffers: Uint8Array[];  // Read-only storage buffers (LUTs, kernels)
}
```

**Binding layout** (all shaders follow this convention):
- `@group(0) @binding(0)`: input pixels (`storage, read`) — `array<vec4<f32>>`
- `@group(0) @binding(1)`: output pixels (`storage, read_write`) — `array<vec4<f32>>`
- `@group(0) @binding(2)`: params (`uniform`) — filter-specific layout
- `@group(0) @binding(3+)`: extra buffers (`storage, read`) — kernel weights, LUTs

**Execution**: chain shaders via ping-pong buffers. Output of shader[i] becomes input of shader[i+1].

### Browser Integration (WebGPU)

```typescript
import { pipelineV2 } from './rasmcore-v2-image.js';

const pipe = new pipelineV2.ImagePipelineV2();
const source = pipe.read(imageBytes, undefined);
const node = pipe.applyFilter(source, 'brightness', params);

// 1. Get GPU plan from pipeline
const plan = pipe.renderGpuPlan(node);
if (plan) {
  // 2. Dispatch via WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Upload input pixels
  const inputBuf = device.createBuffer({
    size: plan.inputPixels.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(plan.inputPixels));

  // Execute each shader in sequence (ping-pong)
  let readBuf = inputBuf;
  let writeBuf = device.createBuffer({ size: inputBuf.size, usage: inputBuf.usage });

  for (const shader of plan.shaders) {
    const module = device.createShaderModule({ code: shader.source });
    // ... create pipeline, bind group, dispatch, swap buffers
  }

  // 3. Readback and inject result
  const resultPixels = await readbackBuffer(device, readBuf);
  pipe.injectGpuResult(node, Array.from(resultPixels));
}

// 4. Encode (uses cached GPU result if available)
const png = pipe.write(node, 'png', undefined);
```

For a complete implementation, see `sdk/v2/lib/gpu-handler.ts` (GpuHandlerV2 class) which handles shader caching, buffer pooling, ping-pong execution, and canvas blit.

### Native Integration (wgpu)

```rust
use rasmcore_pipeline_v2::{GpuExecutor, GpuError, GpuShader};

pub struct WgpuExecutorV2 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_cache: HashMap<u64, wgpu::ShaderModule>,
    // ... buffer pools
}

impl GpuExecutor for WgpuExecutorV2 {
    fn execute(
        &self,
        ops: &[GpuShader],
        input: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, GpuError> {
        // 1. Upload input to GPU buffer
        // 2. For each shader: compile (cached), create bind group, dispatch
        // 3. Ping-pong read/write buffers between passes
        // 4. Readback final output
        Ok(output_pixels)
    }
}

// Wire into graph
let executor = WgpuExecutorV2::try_new()?;
graph.set_gpu_executor(Rc::new(executor));
// Now request_full() automatically uses GPU when possible
```

See `crates/rasmcore-cli/src/gpu_executor_v2.rs` for the full implementation.

### Shader Caching (Host Responsibility)

Shader compilation is expensive (10-50ms per shader). The host should cache compiled pipelines:

```typescript
// Browser: cache by shader source hash
const cache = new Map<string, GPUComputePipeline>();

function getOrCompile(device: GPUDevice, shader: GpuShader): GPUComputePipeline {
  const key = hashSource(shader.source);
  let pipeline = cache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code: shader.source });
    pipeline = device.createComputePipeline({ /* ... */ });
    cache.set(key, pipeline);
  }
  return pipeline;
}
```

```rust
// Native: cache by source content hash (FNV-1a)
let hash = content_hash(source.as_bytes());
let module = self.shader_cache
    .entry(hash)
    .or_insert_with(|| device.create_shader_module(/* ... */));
```

The pipeline's fusion optimizer composes multiple filters into single shaders, so the number of unique shaders is typically much smaller than the number of filters.

---

## Optional: Layer Cache

The layer cache is a content-addressed store for intermediate pipeline results. It persists across pipeline instances within the same WASM instance, enabling instant re-renders when only downstream params change.

### How It Works

Each node's output is keyed by a content hash encoding its full computation lineage:

```
hash = blake3(upstream_hash || operation_name || param_bytes)
```

When a pipeline runs, it checks the cache before computing each node. If the hash matches (same input + same params), the cached result is returned instantly.

### Initialization

```typescript
// Browser
const cache = new pipelineV2.LayerCache(256); // 256 MB capacity
const pipe = new pipelineV2.ImagePipelineV2();
pipe.setLayerCache(cache);
// All subsequent operations benefit from caching
```

The cache is a WIT resource — it lives in WASM linear memory. The host only needs to:
1. Create it with a capacity
2. Inject it into each pipeline instance via `setLayerCache()`
3. Optionally share it across multiple pipeline instances

### Cache Quality

The cache supports transparent quantization to reduce memory usage:

| Quality | Storage | Precision |
|---------|---------|-----------|
| Full | 16 bytes/pixel (f32) | Bit-exact |
| Q16 | 8 bytes/pixel (u16) | ~0.001 max error |
| Q8 | 4 bytes/pixel (u8) | ~0.005 max error |

Consumers always receive f32 — quantization is transparent on read. Set quality before storing:

```typescript
// Not yet exposed via WIT — internal pipeline setting
// Default is Full (f32). Q16 is recommended for preview pipelines.
```

### Memory Management

The cache auto-grows from the initial budget up to 1 GB (default max). Eviction is budget-based: when storing a new entry would exceed the budget, unreferenced entries are evicted first. The `store()` method handles this automatically.

---

## WIT Interface Reference

The complete contract is defined in `wit/v2/pipeline.wit`. Key resource methods:

### Discovery
- `list-operations() -> list<operation-info>` — All registered filters, codecs, transforms
- `find-operation(name) -> option<operation-info>` — Lookup by name

### Graph Construction
- `read(data, config?) -> node-id` — Decode image bytes
- `apply-filter(source, name, params) -> node-id` — Add filter node
- `apply-transform(source, name, params) -> node-id` — Add transform node
- `convert-color-space(source, target) -> node-id` — Color conversion
- `node-info(node) -> node-info` — Get dimensions and color space

### Execution
- `render(node) -> pixel-buffer` — CPU execution, returns f32 RGBA
- `write(node, format, quality?) -> buffer` — Execute and encode
- `render-gpu-plan(node) -> option<gpu-plan>` — Get GPU shader chain (host dispatches)
- `inject-gpu-result(node, pixels)` — Cache GPU result for subsequent write()

### Configuration
- `set-layer-cache(cache)` — Inject shared cache
- `set-demand-strategy(strategy)` — Tiling strategy
- `set-proxy-scale(scale)` — Spatial param auto-scaling
- `set-gpu-available(available)` — Signal GPU availability

### Diagnostics
- `set-tracing(enabled)` — Enable pipeline tracing
- `take-trace() -> list<trace-event>` — Collect timing events (fusion, dispatch, encode)
