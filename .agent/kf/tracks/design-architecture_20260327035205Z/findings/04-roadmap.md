# Implementation Roadmap

## Date: 2026-03-27
## Project: rasmcore

---

## MVP Definition

The minimum viable product is **rasmcore-image** — a complete image processing WASM component.

**Why image first:**
1. Lowest complexity — no stateful codecs, no container formats, no streaming
2. Highest standalone value — image processing is universally needed
3. Proves the architecture — WIT interfaces, cargo-component build, jco transpilation
4. Pure Rust crates are mature (`image`, `photon-rs`)
5. Quick to build parity tests against (libpng, libjpeg are well-understood)

---

## Phased Roadmap

### Phase 0: Foundation (Current)
**Status: COMPLETE**

- [x] Research all technology domains
- [x] Design WIT interfaces
- [x] Define module architecture
- [x] Make key architectural decisions
- [x] Define testing strategy

### Phase 1: Scaffold & First Component
**Target: rasmcore-image MVP**

1. **Project scaffold**
   - Set up Cargo workspace with cargo-component
   - Create `wit/` directory with all interface definitions
   - Set up CI (build, validate, test)
   - Configure wasm-tools for validation

2. **rasmcore:core implementation**
   - Generate Rust types from `rasmcore:core/types` WIT
   - Implement error types
   - This is a shared dependency, not a standalone component

3. **rasmcore-image implementation**
   - Image decoder: PNG, JPEG, WebP, GIF, BMP, TIFF, AVIF, QOI, ICO
   - Image encoder: PNG, JPEG, WebP
   - Transform: resize, crop, rotate, flip, format conversion
   - Filters: blur, sharpen, brightness, contrast, grayscale

4. **Testing infrastructure**
   - Parity test framework (compare output vs native tools)
   - Benchmark framework (WASM vs native)
   - Test fixtures (sample images in each format)

5. **First host integration**
   - jco transpile to npm package
   - wasmtime Rust host example
   - Verify end-to-end: Rust → WASM → jco → TypeScript

### Phase 2: Codec Framework & AV1
**Target: codec plugin architecture + AV1**

1. **rasmcore:codec interfaces live**
   - Implement encoder/decoder resource pattern
   - Plugin discovery via `supported-codecs()`

2. **rasmcore-codec-av1**
   - AV1 encoder (wrapping rav1e)
   - AV1 decoder (wrapping rav1d — needs WASM feasibility verification)
   - Parity tests: bitstream comparison vs native rav1e/rav1d

3. **Component composition validation**
   - Verify `wasm-tools compose` works with codec plugins
   - Document composition workflow

### Phase 3: Video & Audio Processing
**Target: container handling + audio**

1. **rasmcore-video**
   - Container demuxer: MP4, MKV/WebM
   - Container muxer: MP4, MKV/WebM
   - Integration with codec plugins

2. **rasmcore-audio**
   - Audio transforms: resample, format convert, channel mix, gain, trim

3. **Additional open codecs**
   - FLAC (encode/decode)
   - Opus (decode initially, encode later)
   - VP9 (pure Rust reimplementation — longer effort)

### Phase 4: Data Processing
**Target: format conversion + tabular ops**

1. **rasmcore-data**
   - Format converter: JSON, CSV, MessagePack, CBOR
   - Parquet read/write (via parquet-wasm approach)
   - Tabular operations: select, filter, sort, head

### Phase 5: GPU Acceleration
**Target: WebGPU compute for image processing**

1. **GPU-accelerated image filters**
   - Blur, sharpen, color grade via wgpu compute shaders
   - Optional — falls back to CPU if GPU unavailable
   - Browser WebGPU (now) + wasi-gfx (when available)

### Phase 6: Non-Free Codecs
**Target: patent-encumbered codec reimplementations**

1. **Separate repository setup**
   - rasmcore/non-free repo with patent notices
   - Same CI/build system as core

2. **H.264 decoder** (pure Rust, from ITU-T H.264 spec + x264/FFmpeg reference)
3. **H.264 encoder** (pure Rust)
4. **AAC decoder** (pure Rust)
5. **H.265 decoder** (pure Rust — most complex, later priority)

### Phase 7: Distribution & SDK
**Target: easy consumption**

1. **OCI registry publishing**
   - Publish all components to ghcr.io/rasmcore/
   - Publish WIT interfaces as packages

2. **TypeScript SDK**
   - npm packages with ergonomic wrappers around jco output
   - Documentation and examples

3. **License investigation and public release decision**

---

## Pre-Release Milestones

| Milestone | Components | What It Proves |
|-----------|------------|----------------|
| M1: First Component | rasmcore-image | Architecture works end-to-end |
| M2: Plugin System | rasmcore-image + rasmcore-codec-av1 | Composition model works |
| M3: Video Pipeline | video + codec + container | Full media pipeline |
| M4: Data Processing | rasmcore-data | Non-media domain works |
| M5: GPU | image + GPU filters | WebGPU acceleration works |
| M6: Non-Free Codecs | H.264 plugin | Patent separation model works |

---

## What's NOT in the Roadmap (Explicitly Out of Scope)

- Native library distribution (Go CGO, Python PyO3) — WASM only
- Crypto operations — no need, WASI-crypto too early
- Streaming/live video — batch processing first
- VR/AR/3D rendering — media processing only
- Machine learning inference — different domain
- Full FFmpeg feature parity — targeted functionality, not completeness
