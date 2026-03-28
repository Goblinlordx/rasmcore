# rasmcore Architecture

## Overview

rasmcore is a modular image and media processing engine built in pure Rust, compiled to WebAssembly Component Model modules. It is designed from the ground up for the WASM ecosystem — not a port of an existing C/C++ library.

This document describes the key architectural decisions and how they differ from existing solutions.

---

## Demand-Driven Tile Pipeline

rasmcore uses a **pull-based, demand-driven pipeline** inspired by libvips. Operations are not executed immediately — they form a directed acyclic graph (DAG) of nodes. Execution is triggered by a `write()` call at the end of the chain, which pulls pixel data backward through the graph.

```
pipeline.read(data)           // Source node — parses headers, no pixel decode
  → pipeline.resize(960, 540) // Transform node — records parameters
  → pipeline.blur(2.0)        // Filter node — records parameters
  → pipeline.writeJpeg(cfg)   // Sink — drives execution, pulls tiles backward
```

Each node requests **dynamically-sized rectangular regions** from its upstream. A blur node with radius 5 requests its upstream region expanded by 5 pixels on each side. A resize node maps output coordinates back to source coordinates at the appropriate scale. There is no fixed tile grid — the write sink's output format determines the initial chunk geometry, and each upstream node expands the request by its kernel's overlap requirements.

### Spatial Cache with Reference-Counted Borrowing

A pipeline-level **spatial cache** ensures that overlapping pixel regions are computed once and reused. When adjacent output chunks both need overlapping upstream data, the cache serves the overlapping portion from memory instead of recomputing it.

The cache uses reference counting: nodes acquire regions (incrementing the count) and release them when done (decrementing). When the count reaches zero, the region is eligible for reclamation but stays cached for potential reuse. A memory budget controls eviction.

For operations that need only a sliding window of data (blur, sharpen, convolution), only 2-3 upstream regions are ever live simultaneously — regardless of image size.

### Sub-Region Reuse

When a node requests a region that partially overlaps an already-cached region, the cache computes only the **missing sub-regions** and assembles the result from cached + newly computed pixels. This eliminates redundant computation at region boundaries without requiring fixed tile alignment.

---

## How rasmcore Differs from libvips

libvips pioneered the demand-driven pipeline model in C. rasmcore takes the same conceptual approach but differs in several ways:

**Designed for WASM, not ported to it.** libvips is a C library compiled to WASM via Emscripten (as wasm-vips). This brings the entire C runtime, threading model, and memory allocator into the WASM binary (~15 MB). rasmcore is pure Rust targeting the WASM Component Model directly, producing small, composable modules.

**Component Model native.** rasmcore uses WIT (WebAssembly Interface Types) to define its API. This means any language with a Component Model binding — Rust, TypeScript, Python, Go, C# — gets a typed SDK automatically generated from the same interface definition. libvips and ImageMagick are C libraries with hand-written bindings per language.

**No SharedArrayBuffer requirement.** wasm-vips requires SharedArrayBuffer and Cross-Origin headers (COOP/COEP) in browsers for its threading model. rasmcore works with standard WASM — no special headers, no browser restrictions.

**Spatial cache vs line cache.** libvips uses a fixed-height line cache per operation. rasmcore uses a spatial cache with dynamically-sized regions and sub-region reuse, which handles both sequential (scanline) and random (tiled, rotated) access patterns with a single mechanism.

**Portable parallelism.** rasmcore's architecture separates the pipeline orchestrator from compute kernels. The orchestrator (graph + cache + dispatch) is pure Rust that compiles identically to both native and WASM. Today, the host can drive parallelism by dispatching stateless compute kernels to multiple WASM instances with a shared native cache. When wasi-threads stabilizes, the same orchestrator code moves inside the component with zero changes.

---

## How rasmcore Differs from ImageMagick

ImageMagick is a comprehensive, decades-old image processing suite written in C. Its architecture reflects a different era and set of constraints.

**Stateless components vs monolithic binary.** ImageMagick is a single large binary (~15 MB in its WASM form, magick-wasm). rasmcore is composed of small, independently versioned WASM components that can be mixed, matched, and composed using standard Component Model tooling.

**Typed interfaces vs string commands.** ImageMagick's API is command-oriented (strings like `-resize 960x540!`). rasmcore uses WIT-defined typed interfaces with per-format configuration records. A JPEG encode config is a typed struct with `quality: u8`, not a string parameter.

**Pure Rust vs C/C++ with FFI.** ImageMagick delegates to numerous C libraries (libjpeg, libpng, libtiff, etc.) via FFI. rasmcore uses pure Rust implementations, eliminating an entire class of memory safety vulnerabilities inherent to C parsers processing untrusted input.

**Pipeline vs sequential execution.** ImageMagick processes operations sequentially, materializing the full image in memory between each step. rasmcore's demand-driven pipeline avoids intermediate materializations — pixels flow through the node graph from source to sink without full-image copies at each stage.

---

## Dual-Level API

rasmcore exposes two API levels from the same component:

**Level 1 — Pipeline resource.** A `pipeline` WIT resource that owns a node graph internally. The host builds a chain of operations via method calls (`read`, `resize`, `blur`, `write`), and execution happens inside the component on `write()`. Only the final encoded output crosses the WASM boundary. This minimizes host-guest data transfer.

**Level 2 — Stateless compute kernels.** Individual functions (`apply_blur`, `apply_resize`) that take pixels in and return pixels out. These are pure functions with no state. A host that wants to drive parallelism can dispatch these to multiple WASM instances, managing the graph and cache in native host memory.

The same component exposes both levels. Simple consumers use Level 1. Performance-sensitive hosts use Level 2 with their own scheduling.

---

## SIMD Strategy

rasmcore uses SIMD acceleration on all platforms from a single codebase:

- **x86/x86_64:** SSE4.1, AVX2 (runtime detection)
- **ARM/AArch64:** NEON (runtime detection)
- **WASM:** SIMD128 (compile-time via `-C target-feature=+simd128`)

A single `cargo component build` produces one `.wasm` file with SIMD128 instructions. Runtimes that support SIMD128 (wasmtime, V8, SpiderMonkey, JavaScriptCore — all major engines) execute the SIMD path. There are no separate builds for SIMD vs non-SIMD.

For native builds, the same code auto-detects the CPU's SIMD capabilities at runtime and selects the optimal kernel. One source, one binary, optimal everywhere.

Relaxed SIMD (fused multiply-add, relaxed swizzle) is available in wasmtime and modern browsers, providing additional acceleration for convolution and color-space conversion kernels.

---

## Per-Format Codec Modules

Each image format (JPEG, PNG, WebP, etc.) is its own Rust module with:

- A precisely defined **typed configuration record** (not string parameters)
- Standards-aligned behavior referencing the relevant specification (ITU-T T.81 for JPEG, ISO/IEC 15948 for PNG, etc.)
- Both read (decode) and write (encode) capabilities
- Independent testability at the domain boundary

This modular structure means adding a new format is self-contained work that doesn't touch other formats. It also enables the sidecar pattern for patent-encumbered codecs — formats like HEVC can live in separate repositories and be composed at deployment time.

---

## Read/Write Public API

The public API uses `read` and `write` terminology instead of `decode` and `encode`. Users think in terms of "read a JPEG" and "write a PNG", not codec internals. The domain layer retains `decode`/`encode` since those are technically precise at the implementation level. The rename happens at the WIT interface boundary.

---

## Hexagonal Architecture

All rasmcore modules follow a hexagonal (ports/adapters) architecture:

- **Domain logic** has zero WIT dependencies. It defines its own types, error enums, and function signatures. It is fully exercisable via native Rust unit tests without any WASM runtime.
- **WIT adapters** are thin translation layers that convert between WIT-generated types and domain types. They contain no business logic.
- **Domain-defined errors** are explicit enums per module. The adapter layer translates them to WIT error types at the boundary. Domain errors never leak through the interface untranslated.

This separation means the domain logic can be tested with `cargo test` at native speed, while the WASM integration tests validate the full stack through an actual runtime.

---

## Shared Pipeline Crate

The pipeline primitives (spatial cache, rectangle geometry, overlap types) live in a shared `rasmcore-pipeline` workspace crate. The image processing module depends on it. Future modules (video, data) will share the same pipeline infrastructure, enabling a consistent processing model across domains.

This also enables component composition: a video component can import the image component's pipeline interface for per-frame operations, with the pipeline primitives shared at the Rust source level and the component interfaces composed at the WASM level.

---

## Testing Strategy

rasmcore uses a three-tier testing hierarchy:

1. **Domain unit tests** — TDD at domain function boundaries. These are the primary test suite, run with `cargo test` at native speed.

2. **Parity tests** — Compare rasmcore output against reference implementations (ImageMagick for image processing). These validate correctness across the full stack including the WASM adapter layer. Pixel-exact for lossless operations, PSNR/MAE thresholds for lossy.

3. **Cross-language SDK tests** — Validate that generated SDKs (TypeScript via jco, Rust via wasmtime) produce correct results. These prove the Component Model interface works end-to-end.

Performance benchmarks run at all three tiers: native domain functions, WASM component via wasmtime, and comparison against external tools.
