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

rasmcore V2 includes Academy Color Encoding System (ACES) support:

- **Color space tracking**: Per-node color space metadata (ACEScg, ACEScc, sRGB, etc.)
- **Auto-conversion**: The graph inserts color space conversion nodes automatically
- **RRT + ODT**: Reference Rendering Transform and Output Device Transforms
- **Compliance audit**: Nodes can be tagged with compliance level (Compliant, Log, NonCompliant, Unknown)

The ACES pipeline operates in linear light (ACEScg) for processing, with IDTs (Input Device Transforms) at ingest and RRT+ODT at output.

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

**Level 2 — Stateless compute kernels.** Individual functions that take pixels in and return pixels out. A host can dispatch these to multiple WASM instances for parallelism.

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
