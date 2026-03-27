# Synthesis & Recommendation

## Research Date: 2026-03-27

---

## Executive Summary

The WASM Component Model and WASI ecosystem is **mature enough to build on** for Rust and TypeScript targets. WASI 0.2 is stable, tooling exists, and the path to 0.3 (native async with `stream<T>`) is clear. However, **Go host support is a critical gap** that requires a strategic workaround.

---

## Go/No-Go Assessment

### GO — Confirmed Capabilities

| Capability | Status | Confidence |
|------------|--------|------------|
| Rust → WASM component compilation | Stable (wasm32-wasip2 + cargo-component) | HIGH |
| Custom WIT interface definitions | Supported via cargo-component | HIGH |
| Component composition (plugin linking) | Supported via wasm-tools compose | HIGH |
| Streaming I/O for large files | Available via wasi:io (0.2), improved in 0.3 | HIGH |
| Rust host (wasmtime) | Production-ready | HIGH |
| TypeScript/JS host (jco 1.0) | Production-ready | HIGH |
| Cross-component type safety via WIT | Mature | HIGH |
| Module validation and inspection | Stable via wasm-tools | HIGH |

### CONDITIONAL GO — Requires Mitigation

| Capability | Issue | Mitigation |
|------------|-------|------------|
| Go host runtime | No Component Model support in any Go runtime | Provide native Rust lib via C API/CGO for Go consumers |
| Python host runtime | Guest-only today | Use wasmtime-py or provide native Python bindings (PyO3) |
| cargo-component stability | API explicitly "not stable" | Pin versions, test upgrades in CI |
| WASI 0.3 async streams | In preview, not finalized | Build for 0.2 now, design for 0.3 migration |

### RISK — Known Challenges

| Risk | Severity | Mitigation |
|------|----------|------------|
| Go Component Model gap | HIGH | Dual distribution: WASM components + native C API for Go |
| cargo-component breaking changes | MEDIUM | Version pinning, integration tests |
| WASI 0.3 timeline slip | LOW | 0.2 is sufficient for MVP, 0.3 is an enhancement |
| WIT interface design lock-in | MEDIUM | Start with narrow interfaces, expand iteratively |

---

## Recommended Architecture

### Primary Distribution: WASM Components

All core modules are built in Rust and distributed as WASM components with WIT interfaces. This serves:
- **Rust hosts** via wasmtime embedding
- **TypeScript/Node.js** via jco transpilation to npm packages
- **Browser** via jco transpilation to ES modules
- **Any future WASM Component Model host**

### Secondary Distribution: Native Libraries

For languages without Component Model host support:
- **Go:** Expose Rust modules via C API (using `cbindgen`) + Go bindings via CGO
- **Python:** Expose via PyO3/maturin for native Python packages
- This is a pragmatic bridging strategy until Go/Python hosts mature

### Build Targets

| Target | Purpose |
|--------|---------|
| `wasm32-wasip2` | Primary WASM component output |
| Native (x86_64, aarch64) | C API shared library for Go/Python native bindings |

---

## WASI Version Strategy

### Phase 1 (Now): Target WASI 0.2

- Stable, well-supported
- Streaming via `wasi:io` input-stream/output-stream is functional
- All tooling works today

### Phase 2 (H2 2026): Migrate to WASI 0.3

- Native `stream<T>` and `future<T>` types
- Zero-copy stream forwarding
- More ergonomic async I/O
- Better suited for our media streaming use cases

### Design Principle: Forward-Compatible

When designing WIT interfaces now, avoid patterns that would be hard to migrate:
- Use `input-stream`/`output-stream` for I/O (maps directly to 0.3 `stream<u8>`)
- Avoid complex polling patterns (will be replaced by native async)
- Keep interfaces coarse-grained (buffer-oriented, not byte-at-a-time)

---

## Streaming/Large File Feasibility

### Current State (WASI 0.2)

The `wasi:io` streams interface provides everything we need for chunked processing:

```
Host opens file → provides input-stream → WASM reads chunks → processes → writes to output-stream
```

- Non-blocking reads with `read(len)` — returns available bytes
- Blocking reads with `blocking-read(len)` — waits for data
- Stream-to-stream splice for zero-copy forwarding
- Module never sees file path (capability-based security)

### Chunk Size Considerations

- `blocking-write-and-flush`: ~4096 byte limit per call
- For media processing: use non-blocking `write` + manual `flush` for larger chunks
- Recommended: 64KB read chunks, batch writes with explicit flush control

### Assessment

**FEASIBLE.** The streaming model works for large files today. The 0.3 upgrade will make it more ergonomic but is not blocking.

---

## Tooling Recommendations

| Purpose | Recommended Tool | Alternative |
|---------|-----------------|-------------|
| Rust component build | cargo-component | `cargo build --target wasm32-wasip2` (WASI-only) |
| WIT binding generation | wit-bindgen (Rust) | Automatic via cargo-component |
| Component composition | wasm-tools compose | — |
| Component validation | wasm-tools validate | — |
| TS/JS consumption | jco transpile | — |
| Go consumption | cbindgen + CGO | wasmtime CLI subprocess |
| Python consumption | PyO3/maturin | componentize-py (guest) |
| CI validation | wasm-tools validate + host test suites | — |

---

## Overall Verdict

### **GO — with conditions**

The WASM Component Model is ready for razm/core. The Rust → WASM pipeline is mature. TypeScript hosting is production-ready. The Go gap is real but can be bridged with native library distribution.

**Key condition:** Accept that the MVP will provide first-class WASM support for Rust and TypeScript hosts, with Go and Python support via native library bindings as a parallel workstream.

---

## Next Steps (for Architecture Track)

1. Design WIT interfaces for core types (buffer, frame, stream, error)
2. Design domain interfaces (image-processor, video-encoder, data-transformer)
3. Decide on WASM component + native library dual-distribution strategy
4. Prototype: Build a minimal image-resize component, host in wasmtime and jco
5. Prototype: Build C API for the same module, consume from Go
