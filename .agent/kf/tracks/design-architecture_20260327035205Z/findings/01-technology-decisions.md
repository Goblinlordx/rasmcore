# Technology Decision Matrix

## Date: 2026-03-27
## Project: rasmcore

---

## Confirmed Capabilities

| Capability | Status | Evidence | Confidence |
|------------|--------|----------|------------|
| Rust → WASM component compilation | GO | wasm32-wasip2 Tier 2 in Rust 1.82+, cargo-component works | HIGH |
| Custom WIT interfaces | GO | cargo-component required, works with custom WIT | HIGH |
| Component composition (plugins) | GO | wasm-tools compose, WAC — build-time and runtime linking | HIGH |
| Streaming I/O for large files | GO | wasi:io input-stream/output-stream, chunked reads | HIGH |
| Rust host (wasmtime) | GO | Full Component Model, production-ready | HIGH |
| TypeScript host (jco 1.0) | GO | Stable, transpiles to ES modules with TS types | HIGH |
| Browser GPU compute (WebGPU) | GO | All major browsers, wgpu works from WASM | HIGH |
| Image processing (pure Rust) | GO | `image` + `photon-rs`, all formats, WASM-ready | HIGH |
| AV1 encode/decode (pure Rust) | GO | rav1e (encode) + rav1d (decode, ISRG-backed) | HIGH |
| Data format handling | GO | Serde ecosystem, Polars, parquet-wasm — all WASM-ready | HIGH |
| OCI-based component distribution | GO | wkg tool, ghcr.io, namespace routing | HIGH |

## Known Gaps

| Gap | Severity | Mitigation |
|-----|----------|------------|
| Go host — no Component Model | Informational | Consumer problem, not ours (WASM-only strategy) |
| Python host — guest only | Informational | Consumer problem, not ours |
| VP9/H.264/H.265 — no pure Rust | Expected | Pure Rust reimplementation from specs (long-term) |
| Server-side GPU (wasi-gfx) | Low | Phase 2 proposal, browser GPU works now, server 2027+ |
| cargo-component API instability | Medium | Pin versions, test upgrades in CI |
| WASI 0.3 not finalized | Low | Build for 0.2 now, design for 0.3 migration |
| WASI-crypto | Deferred | Not needed for media/data processing |
| License | Deferred | TBD before any public release |

## Known Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Video encoding slow on WASM | Medium | Certain | Document as known limitation; WASM SIMD helps |
| Component boundary copy overhead | Medium | Certain | Coarse-grained APIs, stream-based I/O |
| cargo-component breaking changes | Medium | Likely | Version pinning, integration tests |
| Codec reimplementation effort | High | Certain | Phase over multiple releases, AV1 first |
| WASI 0.3 timeline slip | Low | Possible | 0.2 is sufficient for MVP |

---

## Key Architectural Decisions

### 1. WASI Target: 0.2 now, design for 0.3

- Build all components targeting `wasm32-wasip2`
- Use `input-stream`/`output-stream` for I/O (maps to 0.3 `stream<T>`)
- Avoid complex pollable patterns (replaced by native async in 0.3)
- Migrate to 0.3 when stable (expected H2 2026)

### 2. Strictly WASM — no native library fallbacks

- All modules are WASM components with WIT interfaces
- Go/Python integration gaps are consumer-side problems
- One compilation target, one distribution model

### 3. Pure Rust codec reimplementations

- No C library wrapping or wasi-sdk compilation
- FFmpeg source and ITU-T specs as algorithm reference
- Patent-encumbered codecs in separate non-free repositories

### 4. Parity + benchmark testing for every module

- Every module has parity tests against native counterparts
- Benchmark harness comparing WASM vs native execution

### 5. GPU via wgpu — browser now, server later

- Image processing modules optionally use WebGPU compute shaders
- Same wgpu code works native and WASM
- Server-side GPU deferred until wasi-gfx matures

### 6. Build-time composition for plugins

- `wasm-tools compose` as primary plugin mechanism
- Runtime loading as advanced option for host applications
- WIT interface matching is the plugin contract

### 7. OCI registries for distribution

- Components published to OCI registries via `wkg`
- Namespace: `rasmcore:*` packages
- Signing and verification on publish/fetch

### 8. License TBD — project private until substantial coverage
