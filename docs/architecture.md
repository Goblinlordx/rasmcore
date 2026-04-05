# rasmcore Architecture

## Overview

rasmcore is a modular image and media processing engine built in pure Rust, compiled to WebAssembly Component Model modules. It is designed from the ground up for the WASM ecosystem — not a port of an existing C/C++ library.

This document describes the key architectural decisions and the V2 pipeline design.

---

## V2 Pipeline — f32-Native Processing

rasmcore V2 uses a **pull-based, demand-driven pipeline** where all pixel data flows as `Rgba32f` (4x f32 per pixel). This eliminates the format dispatch complexity of V1.

```
pipeline.read(data)           // Source node — decode + promote to Rgba32f
  → pipeline.resize(960, 540) // Transform node — f32 in, f32 out
  → pipeline.blur(2.0)        // Filter node — f32 in, f32 out
  → pipeline.writeJpeg(cfg)   // Sink — demote to Rgba8, encode
```

### Promote / Demote Pattern

1. **Ingest**: Decoded pixels (any format) are immediately promoted to `Rgba32f`
2. **Processing**: All nodes receive and return `Rgba32f` — no format branching
3. **Output**: At the encode boundary, pixels are demoted to the target format (Rgba8, Rgb16, etc.)

This means filter implementations never handle format dispatch. They always see f32 values in [0.0, 1.0].

### GPU-Primary Dispatch

The V2 pipeline tries **GPU first, CPU fallback**:

1. Check if the filter implements `GpuFilter`
2. If yes, dispatch to WebGPU compute shader (WGSL)
3. If no GPU, or if GPU unavailable, fall back to `CpuFilter::compute()`

GPU buffers are always `F32Vec4` (vec4<f32> per pixel) — no u32 packing.

---

## Demand-Driven Tile Pipeline

Each node requests **dynamically-sized rectangular regions** from its upstream. A blur node with radius 5 requests its upstream region expanded by 5 pixels on each side. A resize node maps output coordinates back to source coordinates.

### Spatial Cache

A pipeline-level spatial cache ensures overlapping pixel regions are computed once and reused. When adjacent output tiles need overlapping upstream data, the cache serves it from memory.

### Layer Cache with Quantization

The `LayerCache` persists results across pipeline lifetimes using content hashes (blake3). It supports opt-in quantization for memory-constrained environments:

| Quality | Bytes/pixel | Memory (4000x3000, 10 layers) |
|---------|------------|-------------------------------|
| Full (f32) | 16 | ~1.9 GB |
| Q16 (u16) | 8 | ~960 MB |
| Q8 (u8) | 4 | ~480 MB |

Quantization is transparent: `store()` quantizes, `get()` promotes back to f32.

---

## ACES Color Pipeline

rasmcore V2 includes Academy Color Encoding System (ACES) support via standard pipeline nodes.

### Working Spaces

| Space | Encoding | Use Case |
|-------|----------|----------|
| **Linear sRGB** | Linear (gamma 1.0) | Default. Blur, composite, resize — physically correct |
| **ACEScct** | Logarithmic + toe | Grading. Brightness, contrast, curves — perceptually uniform |
| **ACEScg** | Linear (AP1 primaries) | CG rendering, VFX compositing |

### IDT / ODT Nodes

ACES color management is implemented as regular pipeline filter nodes, not special pipeline behavior:

- **`aces_idt`** — Input Device Transform. Converts from source color space (sRGB, Rec.709, etc.) to ACEScct. Added after `read()`.
- **`aces_odt`** — Output Display Transform. Converts from ACEScct to output color space. Added before `write()` or display.

The application/UI layer decides when to insert these nodes. The pipeline engine stays simple.

### Why ACEScct for Grading

Point-op adjustments (brightness, contrast, levels, curves) in linear light produce perceptually non-uniform results — dark areas are affected much more than light areas. This is because human vision is logarithmic, not linear.

In ACEScct (logarithmic encoding), a simple `pixel + offset` IS perceptually uniform — it matches how DaVinci Resolve and professional grading tools behave. All existing point-op filters work correctly in ACEScct with zero formula changes.

Spatial operations (blur, sharpen) need linear light for physical correctness. The app layer can wrap them with `aces_cct_to_cg` / `aces_cg_to_cct` conversion nodes.

### Color Space Tracking

Each node carries `NodeInfo.color_space` metadata. The IDT/ODT nodes update this field so downstream nodes know the current working space.

---

## Filter Architecture

### Registration and Codegen

Filters are defined using `#[derive(Filter)]` and implement the `CpuFilter` trait. The codegen pipeline (invoked by `build.rs`) parses filter source files and generates:

- Pipeline node wrappers
- WIT interface declarations
- CLI dispatch tables
- Parameter manifest (JSON)
- SDK methods

### Trait Hierarchy

| Trait | Purpose | Pipeline Effect |
|-------|---------|-----------------|
| `CpuFilter` | CPU compute (required) | Base execution path |
| `GpuFilter` | GPU compute shader | GPU-primary dispatch |
| `PointOp` | Per-channel 1D LUT | LUT fusion (consecutive ops merged) |
| `ColorOp` | 3D color LUT | CLUT fusion |
| `AnalyticOp` | Expression tree IR | Kernel fusion + WGSL codegen |

### Selective Adjustments

The mask system enables Lightroom-style selective editing:

```
source → fork ─┬─→ [adjustment stack] ──┬─→ masked_blend → output
                │                        │
                └─→ [passthrough] ───────┘
                          ↑
                    mask generator
```

Mask generators (gradient, radial, luminance range, color range, brush path) produce grayscale masks. The `masked_blend` compositor applies: `output = adjusted * mask + original * (1 - mask)`.

---

## How rasmcore Differs from libvips

libvips pioneered the demand-driven pipeline model in C. rasmcore takes the same conceptual approach but differs:

- **Designed for WASM**, not ported to it (pure Rust, no Emscripten)
- **Component Model native** — WIT-defined typed APIs, auto-generated SDKs for any language
- **No SharedArrayBuffer** — works with standard WASM, no special browser headers
- **Spatial cache** vs line cache — handles both sequential and random access patterns
- **f32-native** — no format dispatch overhead, GPU-first processing

## How rasmcore Differs from ImageMagick

- **Stateless components** vs monolithic binary
- **Typed interfaces** via WIT vs string command parsing
- **Pure Rust** vs C/C++ with FFI (no memory safety vulnerabilities from C parsers)
- **Pipeline execution** vs sequential (no full-image materializations between steps)
- **GPU acceleration** via WebGPU compute shaders

---

## Dual-Level API

**Level 1 — Pipeline resource.** A `pipeline` WIT resource that owns a node graph. The host builds a chain via method calls, execution happens inside the component on `write()`. Only the final encoded output crosses the WASM boundary.

**Level 2 — GPU plan dispatch.** The pipeline emits execution plans (`gpu-plan`) containing WGSL shader source and input pixels. The host executes these plans on actual GPU hardware and decides where the output goes — canvas, file, video encoder, or nowhere. The WASM component never sees the output.

**Level 3 (future) — Stateless compute kernels.** Individual functions that take pixels in and return pixels out. A host can dispatch these to multiple WASM instances for parallelism.

---

## GPU Display Surface

The GPU display surface eliminates the traditional CPU round-trip for real-time preview rendering. Instead of encoding pixels to PNG and drawing to a 2D canvas, the host renders GPU compute output directly to a WebGPU canvas.

### Three Rendering Paths

**Path A — GPU compute + direct blit (fastest, primary):**

```
WASM: renderGpuPlan(sinkNode)  → returns shaders + input pixels (a recipe, not output)
Host: upload input → GPU storage buffer
      execute compute shaders (ping-pong buffers A↔B)
      blit render pass: read final buffer → fullscreen triangle → canvas texture
      single command buffer submission (compute + render together)
      → pixels never leave GPU
```

**Path B — CPU compute + GPU upload (fallback for non-GPU filters):**

```
WASM: render(sinkNode)  → returns f32 pixel buffer
Host: upload f32 array → GPU storage buffer
      blit render pass → canvas texture
      → one CPU→GPU copy
```

**Path C — PNG encode (legacy, no WebGPU):**

```
WASM: pipe.write('png')  → returns encoded PNG bytes
Host: Blob → Image → 2D canvas.drawImage()
      → two copies + encode/decode overhead
```

### How It Works

The WASM component is a **planner**, not an executor. When the host calls `renderGpuPlan()`, the pipeline walks its node graph, collects the chain of fusable GPU shaders, and returns a plan:

```wit
record gpu-plan {
    shaders: list<gpu-shader>,    // WGSL source + workgroup config
    input-pixels: pixel-buffer,   // f32 data entering the shader chain
    width: u32,
    height: u32,
}
```

The host takes this plan and executes it however it wants. In the browser, `GpuHandlerV2.executeAndDisplay()` runs all compute shaders and appends a blit render pass in a single command submission. The blit shader (`display-blit.wgsl`) reads from the same storage buffer the compute shaders wrote to — no copy, no transfer.

The pipeline never writes to a GPU buffer. The compute shaders already write to storage buffers (that's how compute shaders work). The blit shader just reads from that same buffer and writes to the canvas texture.

### Canvas Configuration

The display canvas uses `rgba16float` format with configurable tone mapping:

- **Standard mode**: clamps to [0, 1] — SDR display
- **Extended mode**: passes through values > 1.0 — HDR display on capable monitors

Pan/zoom is handled in the blit shader via a `Viewport` uniform buffer (canvas dimensions, image dimensions, pan offset, zoom factor). CSS transforms are not used — the shader handles coordinate mapping directly.

### Flicker Prevention

Setting `canvas.width` or `canvas.height` clears the canvas (browser spec). During rapid slider changes, this caused visible flashes. Fix: `resizeDisplay()` skips if dimensions haven't changed.

### Host Portability

The `gpu-plan` is pure data — WGSL strings, f32 arrays, workgroup dimensions. Any host that can parse WGSL and dispatch compute shaders can use it:

| Host | GPU Backend | Notes |
|------|------------|-------|
| Browser (WebGPU) | Vulkan/Metal/DX12 via browser | Current implementation |
| wgpu (Rust native) | Vulkan/Metal/DX12 direct | Same WGSL shaders |
| Metal (Swift) | WGSL→MSL via naga | Shader logic unchanged |
| Vulkan (C++) | WGSL→SPIR-V via naga/tint | Shader logic unchanged |

---

## Data Transfer Efficiency

### Current: Two Copies at Boundary

```
WASM linear memory → JS heap (jco copies list<f32>) → GPU buffer (writeBuffer)
```

For preview resolution (720px): ~8 MB — negligible.
For 4K: ~126 MB — noticeable but happens once per pipeline execution.

After upload, the entire filter chain + display stays GPU-side with zero copies.

### Future: Shared Memory (wasi-threads / WASI 0.3)

With `SharedArrayBuffer`, the host could read directly from WASM linear memory:

```
WASM linear memory ←→ Float32Array view (zero copy) → GPU buffer (one copy)
```

One copy instead of two. The remaining CPU→GPU copy is unavoidable without `wasi-gfx`.

### Future: Host-Side Codec Offload

The architecture naturally supports moving decode/encode out of WASM entirely:

```
// Instead of WASM decoding:
Host decodes (hardware decoder) → GPU buffer → compute shaders → display

// Instead of WASM encoding:
Compute output → host reads GPU buffer → host encodes (hardware encoder)
```

The pipeline becomes a pure computation graph planner. The host owns all I/O. The WASM component emits execution plans and never touches pixel bytes.

This is enabled by `read-pixels` (host provides pre-decoded pixels) and host-side access to the gpu-plan output buffer. No special WASM features needed — just a WIT interface extension.

---

## SIMD Strategy

rasmcore uses SIMD on all platforms from a single codebase:

- **x86/x86_64:** SSE4.1, AVX2 (runtime detection)
- **ARM/AArch64:** NEON (runtime detection)
- **WASM:** SIMD128 (compile-time via `-C target-feature=+simd128`)

One source, one binary, optimal everywhere.

---

## Hexagonal Architecture

- **Domain logic** has zero WIT dependencies — fully testable with `cargo test`
- **WIT adapters** are thin translation layers (no business logic)
- **Domain errors** are explicit enums, translated to WIT errors at the boundary

---

## Crate Structure

```
rasmcore-pipeline/          — Shared pipeline primitives (cache, rect, layer cache)
rasmcore-pipeline-v2/       — V2 pipeline (f32-native graph, GPU executor, ACES, fusion)
rasmcore-image/             — Image processing domain (filters, codecs, pipeline nodes)
rasmcore-macros/            — Proc macros (#[derive(Filter)], #[register_generator], etc.)
rasmcore-codegen/           — Build-time codegen (WIT, adapters, dispatch, manifest)
rasmcore-gpu-shaders/       — WGSL shader composition helpers
rasmcore-ffi/               — C FFI bindings for native hosts
rasmcore-codecs-v2/         — V2 unified codec system
```

---

## Testing Strategy

1. **Domain unit tests** — TDD at function boundaries (`cargo test`)
2. **Parity tests** — Compare against reference implementations (ImageMagick, vips, OpenCV, numpy)
3. **GPU parity tests** — Verify GPU output matches CPU within f32 tolerance
4. **Cross-language SDK tests** — Validate TypeScript, Python, Go SDKs via Component Model
