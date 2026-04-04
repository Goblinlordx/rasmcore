# WebGPU Direct Display Surface for HDR Preview

**Track:** research-webgpu-display_20260404154202Z
**Type:** Research / Design
**Date:** 2026-04-05

---

## 1. Current Data Flow & Latency Analysis

### Current Pipeline (GPU path)

```
User adjusts filter
  │
  ▼
v2-preview-worker.ts: processChain()
  │ Build fluent pipeline, call renderGpuPlan()
  ▼
GpuHandlerV2.execute()
  │ Upload Float32Array → GPU storage buffer (writeBuffer)
  │ Dispatch N compute shaders (ping-pong buffers A ↔ B)
  │ Copy final buffer → staging buffer (COPY_DST | MAP_READ)
  │ staging.mapAsync(READ) — GPU→CPU readback [~2-8ms for 720p]
  ▼
Float32Array on CPU (in worker)
  │ raw.injectGpuResult(node, pixels) — cache in WASM pipeline
  │ pipe.write('png') — WASM PNG encode [~5-15ms for 720p]
  ▼
PNG bytes (Uint8Array)
  │ postMessage({ type: 'result', png: buf }, [buf]) — transfer to main thread
  ▼
Main thread (usePreviewWorker.ts)
  │ new Blob([png]) → URL.createObjectURL() → new Image()
  │ img.onload → canvas.getContext('2d').drawImage(img, ...) [~1-3ms]
  ▼
2D canvas displays the image
```

### Latency Breakdown (estimated, 720p preview)

| Step | Time | Notes |
|------|------|-------|
| GPU compute dispatch | 1-5ms | Shader-dependent, often <2ms |
| GPU→CPU readback (mapAsync) | 2-8ms | PCIe/unified memory transfer |
| injectGpuResult + PNG encode | 5-15ms | WASM, dominates total |
| postMessage transfer | <1ms | Transferable ArrayBuffer |
| Blob → Image decode → 2D draw | 2-5ms | Browser image pipeline |
| **Total** | **~10-34ms** | |

### Proposed Pipeline (direct display)

```
GpuHandlerV2.execute() — compute shaders finish
  │ Final result in GPU storage buffer (never leaves GPU)
  ▼
Render pass: fullscreen-quad shader
  │ Reads storage buffer, writes to canvas texture
  │ Canvas configured: rgba16float + toneMapping 'extended'
  ▼
WebGPU canvas displays HDR image — zero CPU round-trip
```

| Step | Time | Notes |
|------|------|-------|
| GPU compute dispatch | 1-5ms | Same as before |
| Render pass (blit) | <1ms | Trivial fullscreen quad |
| **Total** | **~1-6ms** | |

**Savings:** Eliminates mapAsync readback, PNG encode, postMessage, Blob/Image decode. ~5-28ms saved per frame. The encode path (export to file) remains unchanged — only the display path changes.

---

## 2. WebGPU Canvas Render Pass Feasibility

### Validation

The approach is feasible. WebGPU supports rendering compute output directly to a canvas texture in the same command encoder:

```typescript
// Same command encoder: compute dispatch + render pass
const encoder = device.createCommandEncoder();

// 1. Compute pass (existing)
const computePass = encoder.beginComputePass();
// ... dispatch shaders ...
computePass.end();

// 2. Render pass — blit compute output to canvas
const canvasTexture = context.getCurrentTexture();
const renderPass = encoder.beginRenderPass({
  colorAttachments: [{
    view: canvasTexture.createView(),
    loadOp: 'clear',
    storeOp: 'store',
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }],
});
renderPass.setPipeline(blitPipeline);
renderPass.setBindGroup(0, blitBindGroup); // storage buffer + uniforms
renderPass.draw(6); // fullscreen quad (2 triangles)
renderPass.end();

device.queue.submit([encoder.finish()]);
```

GPU ordering guarantees within a single command encoder ensure the compute pass completes before the render pass reads the buffer. No explicit synchronization needed.

### Key Constraint

The compute shaders currently run in a Web Worker (`v2-preview-worker.ts`). A WebGPU canvas context can only be created from a canvas element on the main thread (or an OffscreenCanvas transferred to a worker). Two options:

1. **OffscreenCanvas in worker** — Transfer an OffscreenCanvas to the worker via `canvas.transferControlToOffscreen()`. The worker configures the WebGPU context on it. Compute + render happen in the worker. Main thread just has the `<canvas>` element.

2. **Move render pass to main thread** — Worker returns a GPUBuffer handle (not possible cross-thread). Not viable — GPUDevice is not transferable.

**Decision: OffscreenCanvas in worker** is the correct approach. The GPUDevice lives in the worker, so the render pass must also live there. The main thread transfers an OffscreenCanvas to the worker at init time.

---

## 3. GPUCanvasConfiguration & Browser Support

### Canvas Configuration

```typescript
const context = offscreenCanvas.getContext('webgpu');
context.configure({
  device,
  format: 'rgba16float',        // 16-bit float per channel — carries HDR values
  alphaMode: 'premultiplied',
  toneMapping: { mode: 'extended' }, // HDR: allow values > 1.0
});
```

- `format: 'rgba16float'` — Required for HDR. `getPreferredCanvasFormat()` returns `bgra8unorm` (always), but `rgba16float` is a valid non-preferred format.
- `toneMapping.mode: 'extended'` — Values > 1.0 map to higher luminance on HDR displays. On SDR displays, behaves like `'standard'` (clamps to 1.0).
- `alphaMode: 'premultiplied'` — Standard for compositing with the page.

### Browser Support Matrix

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| WebGPU | 113+ | 113+ | Nightly (flag) | 18+ (partial) |
| `toneMapping` | 131+ | 131+ | Not yet | Not yet |
| `rgba16float` canvas | 131+ | 131+ | Not yet | Unclear |
| OffscreenCanvas + WebGPU | Yes | Yes | Not yet | Not yet |

### HDR Detection

```typescript
// CSS media query — works in all modern browsers
const isHDR = matchMedia('(dynamic-range: high)').matches;

// Choose tone mapping mode
const toneMapping = { mode: isHDR ? 'extended' : 'standard' };
```

CSS `@media (dynamic-range: high)` — Chrome 98+, Safari 15.4+, Firefox 100+.

---

## 4. Buffer Ownership Model

### Current Model

```
GpuHandlerV2.execute():
  bufA, bufB — created per execute(), destroyed at end
  staging — created for readback, destroyed after mapAsync
  → Returns Float32Array (CPU copy). All GPU buffers gone.
```

The caller gets a CPU-side copy. No GPU resources survive past `execute()`.

### Proposed Model

The display path needs the GPU buffer to survive so the render pass can read it. Two sub-modes:

#### A. Display Mode (preview — zero-copy)

```
GpuHandlerV2.executeAndDisplay():
  bufA, bufB — ping-pong as before
  After last compute pass: DON'T readback
  Instead: bind final buffer in render pass → canvas texture
  Destroy bufA/bufB AFTER render pass completes
  → Returns void. Display is a side-effect.
```

#### B. Export Mode (file output — unchanged)

```
GpuHandlerV2.execute():
  Same as current. mapAsync → Float32Array → returned to caller.
  Caller encodes to PNG/AVIF/etc via WASM.
```

### API Design

```typescript
class GpuHandlerV2 {
  // Existing — for export path
  async execute(ops, input, width, height): Promise<GpuResult>;

  // New — for display path
  async executeAndDisplay(ops, input, width, height): Promise<void>;

  // Configure display surface (called once at init)
  setDisplayCanvas(canvas: OffscreenCanvas): void;
}
```

The `setDisplayCanvas` method configures the WebGPU canvas context with `rgba16float` + tone mapping. Called once when the worker receives the OffscreenCanvas from the main thread.

### Lifetime Rules

1. **Compute buffers (bufA/bufB):** Created at start of `executeAndDisplay()`, destroyed after `queue.submit()` of the combined compute+render command buffer. The GPU schedules destruction after all commands complete.
2. **Canvas texture:** Acquired via `context.getCurrentTexture()` per frame. Ownership managed by the browser — released when the next frame begins.
3. **Blit pipeline & bind group layout:** Created once in `setDisplayCanvas()`, cached for reuse. Bind group recreated per frame (references the current compute output buffer).
4. **Uniform buffer (viewport):** Persistent, updated per frame via `writeBuffer()` for pan/zoom.

---

## 5. Fallback Strategy

### Detection Flow

```
Page load
  │
  ├─ WebGPU available? (navigator.gpu exists)
  │   ├─ Yes → Request adapter + device
  │   │   ├─ Success → Use WebGPU canvas (OffscreenCanvas path)
  │   │   └─ Failure → Fall back to 2D canvas
  │   └─ No → Fall back to 2D canvas
  │
  ▼
2D canvas fallback = current PNG round-trip path (unchanged)
```

### Dual-Canvas Architecture

**Recommendation: Single canvas element, context chosen at startup.**

A canvas element can only have one context type for its lifetime. Once `getContext('webgpu')` is called, `getContext('2d')` returns null (and vice versa). Options:

| Approach | Pros | Cons |
|----------|------|------|
| **A. Single canvas, choose at startup** | Simple DOM, no z-index issues | Must reload to switch context type |
| **B. Two canvas elements, toggle visibility** | Can switch at runtime | Complex z-index, duplicate pan/zoom, DOM bloat |
| **C. Always WebGPU canvas, 2D fallback uses readback** | Unified render path | WebGPU required even for fallback |

**Decision: Approach A** — Detect WebGPU support once at startup. Create either a WebGPU canvas (OffscreenCanvas transferred to worker) or a 2D canvas (current path). Runtime switching is unnecessary — WebGPU support doesn't change during a session.

### Implementation

```tsx
// Canvas.tsx — at mount time
const supportsWebGPU = 'gpu' in navigator;

if (supportsWebGPU) {
  const offscreen = canvasEl.transferControlToOffscreen();
  worker.postMessage({ type: 'init-display', canvas: offscreen }, [offscreen]);
  // Worker owns the canvas now — main thread cannot draw to it
} else {
  // Current 2D canvas path — worker sends PNG, main thread draws
}
```

**Important:** Once `transferControlToOffscreen()` is called, the main thread cannot use that canvas element for 2D drawing. The worker fully controls it. This is fine — the worker already does all the rendering work.

### Non-GPU Filter Chains

When a filter chain has no GPU-acceleratable operations, the pipeline runs entirely in WASM on the CPU. The worker still needs to display the result. Two options:

1. **Upload CPU result to GPU, render via blit** — `device.queue.writeBuffer()` with the f32 pixels, then run the blit render pass. Small overhead (~1ms) but keeps a unified display path.
2. **Readback + 2D canvas** — Would require a second canvas or switching contexts. Not viable.

**Decision: Option 1.** Always display via the WebGPU render pass, even for CPU-only results. The `writeBuffer` cost is negligible for preview-sized images.

---

## 6. Interaction with Pending Tracks

### gpu-shader-cache (gpu-shader-cache_20260404053001Z) — HIGH conflict

**What it does:** Caches `GPUComputePipeline` objects to avoid re-creating shader modules on repeated executions. Touches `gpu-handler-v2.ts`.

**Interaction points:**
- The display render pass needs its own `GPURenderPipeline` (not compute). The shader cache should be extended to cache this too, or the blit pipeline can be cached separately (it's a single pipeline, created once).
- If the shader cache introduces a `PipelineCache` abstraction, the blit pipeline should integrate as a consumer.

**Compatibility strategy:**
- The blit render pipeline is created once per `setDisplayCanvas()` call and never changes. It doesn't need the shader cache — it's a singleton.
- The shader cache track only touches compute pipeline caching. No conflict if the display renderer stores its own blit pipeline reference.
- **Sequencing: Either order works.** If gpu-shader-cache lands first, the display renderer simply doesn't use it (blit pipeline is a singleton). If display lands first, the shader cache doesn't need to know about render pipelines.

### pipeline-perf-tracing (pipeline-perf-tracing_20260404050900Z) — HIGH conflict

**What it does:** Adds performance tracing to the V2 pipeline and GPU dispatch. Touches `v2-preview-worker.ts`.

**Interaction points:**
- The display path adds a new code path in the worker (`executeAndDisplay` instead of `execute`). Tracing should instrument both paths.
- The worker message handler gains a new message type (`init-display`).

**Compatibility strategy:**
- Tracing should wrap both `execute()` and `executeAndDisplay()` calls.
- The tracing track can add timing markers around the new display path if it lands second, or the display track can add trace points if it lands second.
- **Sequencing: Either order works.** Both tracks modify different parts of the worker — tracing adds instrumentation wrappers, display adds a new code path.

**Risk mitigation:** Both tracks should be rebased against each other before merge. The conflict zone is the worker's `processChain()` function — both tracks modify the GPU dispatch section.

---

## 7. Fullscreen-Quad WGSL Shader Specification

### Shader: `display-blit.wgsl`

```wgsl
// Fullscreen-quad blit shader: storage buffer → canvas texture
//
// Reads f32 pixel data from the compute output storage buffer and writes
// to the canvas texture. Supports pan/zoom via viewport uniforms.

struct Viewport {
    // Canvas dimensions in pixels
    canvas_width: f32,
    canvas_height: f32,
    // Image dimensions in pixels
    image_width: f32,
    image_height: f32,
    // Pan offset in image pixels (center of viewport in image space)
    pan_x: f32,
    pan_y: f32,
    // Zoom factor (1.0 = fit, 2.0 = 200%, etc.)
    zoom: f32,
    // Tone mapping mode: 0 = standard (clamp), 1 = extended (pass-through)
    tone_mode: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> vp: Viewport;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle (3 vertices, no vertex buffer needed)
// Covers [-1,1] clip space with UV [0,1]
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    // 0 → (-1,-1), 1 → (3,-1), 2 → (-1,3) — oversized triangle, clipped to viewport
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5); // flip Y
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map UV (canvas space) → image pixel coordinate
    let canvas_px = in.uv * vec2<f32>(vp.canvas_width, vp.canvas_height);

    // Canvas center
    let center = vec2<f32>(vp.canvas_width, vp.canvas_height) * 0.5;

    // Image pixel coordinate (accounting for pan + zoom)
    let img_px = (canvas_px - center) / vp.zoom + vec2<f32>(vp.pan_x, vp.pan_y);

    // Bounds check
    let ix = i32(floor(img_px.x));
    let iy = i32(floor(img_px.y));
    let w = i32(vp.image_width);
    let h = i32(vp.image_height);

    if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
        // Outside image bounds — transparent (shows page background)
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Nearest-neighbor sampling from storage buffer
    let idx = iy * w + ix;
    var color = pixels[idx];

    // Tone mapping
    if (vp.tone_mode == 0u) {
        // Standard: clamp to [0, 1]
        color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    // Extended: pass through — canvas toneMapping handles display mapping

    // Premultiply alpha for canvas compositing
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
```

### Notes

- **Fullscreen triangle** (3 vertices, no index buffer) instead of quad (6 vertices). Single triangle oversized to cover the viewport, clipped by the rasterizer. More efficient — fewer vertices, no diagonal edge.
- **Nearest-neighbor sampling** for pixel-art fidelity at high zoom. Bilinear can be added later as a user preference via the viewport uniform.
- **Pan/zoom in shader** instead of CSS transforms. Provides sub-pixel precision and avoids CSS transform re-compositing. The `Viewport` uniform is updated per frame via `device.queue.writeBuffer()`.
- **Tone mapping is minimal** — the canvas's `toneMapping` configuration handles the actual SDR→HDR display mapping. The shader just needs to pass through linear values > 1.0 (extended mode) or clamp them (standard mode).

### Render Pipeline Configuration

```typescript
const blitPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: device.createShaderModule({ code: BLIT_SHADER }),
    entryPoint: 'vs_main',
  },
  fragment: {
    module: device.createShaderModule({ code: BLIT_SHADER }),
    entryPoint: 'fs_main',
    targets: [{ format: 'rgba16float' }],
  },
  primitive: { topology: 'triangle-list' },
});
```

---

## 8. Pan/Zoom Strategy

### Current: CSS Transforms

```tsx
// useCanvasTransform.ts
const style = {
  transform: `translate(${tx}px, ${ty}px) scale(${zoom})`,
  transformOrigin: '0 0',
};
```

This applies CSS transforms to the canvas element. Works well with 2D canvas.

### Proposed: Viewport Uniform in Shader

With WebGPU canvas, CSS transforms still work — the browser composites the canvas element like any other DOM element. However, shader-based pan/zoom has advantages:

| Aspect | CSS Transform | Shader Viewport |
|--------|--------------|-----------------|
| Sub-pixel precision | Browser-dependent | Full float precision |
| Resampling on zoom | Browser bilinear (blurry) | Nearest-neighbor (crisp pixels) |
| Performance | CSS compositor (fast) | GPU uniform update (fast) |
| Interaction with page | Normal DOM flow | Normal DOM flow |
| Implementation | No shader changes | Viewport uniform in blit shader |

**Decision: Shader-based viewport** for the WebGPU path. The `Viewport` uniform struct (defined in the WGSL spec above) receives pan/zoom state from `useCanvasTransform`. The canvas element itself stays at full viewport size with no CSS transforms.

The main thread sends viewport updates to the worker:
```typescript
worker.postMessage({
  type: 'viewport',
  panX, panY, zoom,
  canvasWidth, canvasHeight,
});
```

The worker updates the uniform buffer and requests a re-render (no recompute — just re-blit with new viewport).

---

## 9. Performance Estimate

### Current Path (720p preview, GPU-accelerated filter chain)

| Step | Time |
|------|------|
| GPU compute | ~3ms |
| mapAsync readback | ~4ms |
| WASM PNG encode | ~10ms |
| postMessage + Blob + Image + 2D draw | ~3ms |
| **Total** | **~20ms (50 fps)** |

### Proposed Path (same workload)

| Step | Time |
|------|------|
| GPU compute | ~3ms |
| Render pass (blit) | ~0.5ms |
| **Total** | **~3.5ms (285 fps)** |

**Speedup: ~5.7x** for display latency. The encode path for file export is unaffected.

For viewport-only updates (pan/zoom without recompute), the proposed path only runs the blit shader: **<1ms per frame**, enabling smooth 60fps+ pan/zoom with no recomputation.

---

## 10. Architecture Decision: Extend GpuHandlerV2 vs Separate DisplayRenderer

### Option A: Extend GpuHandlerV2

Add `setDisplayCanvas()` and `executeAndDisplay()` to the existing class. Pros: single GPUDevice, shared buffer management. Cons: mixes compute orchestration with display concerns.

### Option B: Separate DisplayRenderer

New class that takes a GPUDevice and OffscreenCanvas. GpuHandlerV2 returns a GPUBuffer reference; DisplayRenderer consumes it. Pros: clean separation. Cons: buffer ownership gets complicated (who destroys it?).

### Decision: Option A — Extend GpuHandlerV2

Reasoning:
1. The display render pass must run in the **same command encoder** as the compute pass for correct ordering. A separate class would need to coordinate command encoder sharing.
2. The GpuHandlerV2 already owns the GPUDevice. Adding a canvas context is a minor extension.
3. Buffer lifetime is simple: create at start of `executeAndDisplay()`, used by both compute and render passes in the same command buffer, destroyed after submit.
4. The blit pipeline is a singleton (~20 lines of setup code). Not enough complexity to warrant a separate class.

The class grows by ~60 lines (canvas config, blit pipeline setup, viewport uniform, render pass in `executeAndDisplay`). This is proportional to the feature's complexity.

---

## 11. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Canvas context location | OffscreenCanvas in worker | GPUDevice lives in worker; can't transfer GPUDevice cross-thread |
| Canvas format | `rgba16float` | Required for HDR values > 1.0 |
| Tone mapping | `extended` on HDR, `standard` on SDR | Detected via `(dynamic-range: high)` media query |
| Fallback | Single canvas, context chosen at startup | WebGPU support is static per session |
| CPU-only results | Upload to GPU, display via blit | Unified display path, negligible overhead |
| Architecture | Extend GpuHandlerV2 | Same command encoder for compute + render |
| Pan/zoom | Shader viewport uniform | Sub-pixel precision, crisp pixels at zoom |
| Blit shader | Fullscreen triangle, nearest-neighbor | Efficient, pixel-perfect |
| Buffer model | Create/destroy per executeAndDisplay() | Simple lifetime, no leaks |
| Shader cache interaction | Blit pipeline is singleton, separate from compute cache | No conflict with gpu-shader-cache track |
| Track sequencing | Independent of gpu-shader-cache and pipeline-perf-tracing | Either order works; rebase resolves conflicts in processChain() |

---

## 12. Implementation Sketch

### Files to Create/Modify

| File | Change |
|------|--------|
| `web-ui/src/gpu-handler-v2.ts` | Add `setDisplayCanvas()`, `executeAndDisplay()`, blit pipeline, viewport uniform |
| `web-ui/src/shaders/display-blit.wgsl` | New file — fullscreen-quad blit shader |
| `web-ui/src/v2-preview-worker.ts` | Handle `init-display` and `viewport` messages; call `executeAndDisplay()` for preview |
| `web-ui/src/components/Canvas.tsx` | Detect WebGPU, transfer OffscreenCanvas at mount |
| `web-ui/src/hooks/usePreviewWorker.ts` | Send OffscreenCanvas to worker; forward viewport state |
| `web-ui/src/hooks/useCanvasTransform.ts` | Emit viewport updates to worker (pan/zoom changes) |

### Migration Path

1. Feature-flagged: `USE_WEBGPU_DISPLAY` env var or runtime detection
2. WebGPU path and 2D path coexist — detection at startup chooses one
3. No WIT or WASM pipeline changes required
4. Export path (write to file) completely unchanged
