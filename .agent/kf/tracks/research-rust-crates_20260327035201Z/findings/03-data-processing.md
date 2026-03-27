# Data Processing Crate Evaluation

## Research Date: 2026-03-27

---

## 1. Polars — DataFrame Library

**Repository:** [pola-rs/polars](https://github.com/pola-rs/polars)
**Status:** Mature, very active, widely adopted
**Pure Rust:** Yes (built on Apache Arrow)

### WASM Support

- **Can compile to WASM** — proven in browser-based DataFrame analytics
- Existing projects: [polars-wasm](https://github.com/llalma/polars-wasm), [polars-wasm-mwe](https://github.com/rohit-ptl/polars-wasm-mwe)
- Use case: Client-side data exploration (load Parquet, filter, group, plot — no server)

### Feature Flags for WASM

| Feature | Default | WASM Compatible | Notes |
|---------|---------|----------------|-------|
| Core DataFrame ops | Yes | Yes | Filter, select, group, join |
| `simd` | Optional | Partial | Requires nightly, WASM-SIMD available |
| `performant` | Optional | Unknown | May use platform-specific opts |
| `bigidx` | Optional | Yes | For >2^32 rows |
| `rayon` (threading) | Default | NO | Must disable for WASM |
| `object` (generic ChunkedArray) | Optional | Yes | |

### Configuration for WASM

```toml
[dependencies]
polars = { version = "0.46", default-features = false, features = [
    "lazy", "csv", "parquet", "json", "dtype-full"
] }
```

### Limitations on WASM

- No multithreading (rayon disabled)
- Memory constrained by WASM linear memory (typically 4GB max)
- Some IO features (reading from URLs, async IO) may not work without WASI
- Performance is single-threaded — fine for moderate datasets, not for large-scale analytics

### Assessment

**STRONG CANDIDATE with caveats.** Polars on WASM is proven for browser analytics and moderate data processing. However, for large-scale data processing (millions of rows, complex aggregations), the single-threaded WASM constraint is significant. Best suited for:
- Format conversion (CSV → Parquet, JSON → Arrow)
- Data filtering and transformation
- Client-side analytics dashboards
- Not: production ETL pipelines (use native for that)

---

## 2. Arrow-rs — Apache Arrow for Rust

**Repository:** [apache/arrow-rs](https://github.com/apache/arrow-rs)
**Status:** Mature, official Apache project
**Pure Rust:** Yes

### WASM Compilation

- Core `arrow` crate compiles to `wasm32-unknown-unknown` and `wasm32-wasi`
- Requires `default-features = false` with explicit feature selection
- Example: `arrow = { version = "55", default-features = false, features = ["csv", "ipc"] }`

### Known WASM Issues

| Component | WASM Status | Issue |
|-----------|-------------|-------|
| `arrow` (core) | Works | Must disable default features |
| `parquet` | **Does NOT compile** to wasm32-unknown-unknown | [Known issue](https://github.com/apache/arrow-rs/issues/4776) |
| `object-store` (HTTP) | **Does NOT compile** to WASM | HTTP client deps incompatible |
| SIMD features | Partial | packed_simd issues on WASM |

### Assessment

**USEFUL as foundation, not as primary API.** Arrow-rs provides the columnar memory format that Polars builds on. Direct Arrow-rs usage is lower-level — good for custom data processing pipelines but more complex than Polars for end users.

**Key limitation:** The Parquet crate doesn't compile to `wasm32-unknown-unknown`. However, `parquet-wasm` (see below) works around this.

---

## 3. Data Format Crates — WASM Readiness Survey

### Serialization Framework: Serde

The Serde ecosystem is the backbone. All format crates below use Serde for (de)serialization.

| Crate | Format | Pure Rust | WASM | Notes |
|-------|--------|-----------|------|-------|
| `serde_json` | JSON | Yes | Yes | Standard JSON. WASM-specific: `serde-json-wasm` for smaller size |
| `csv` | CSV | Yes | Yes | Full CSV parser with Serde integration |
| `rmp-serde` | MessagePack | Yes | Yes | Complete pure-Rust MessagePack |
| `serde_cbor` | CBOR | Yes | Yes | Compact binary format |
| `serde_yaml` | YAML | Yes | Yes | YAML support |
| `toml` | TOML | Yes | Yes | Config format |
| `bincode` | Binary | Yes | Yes | Fast binary serialization |
| `postcard` | Binary (no_std) | Yes | Yes | Designed for embedded/WASM |
| `ciborium` | CBOR | Yes | Yes | Modern CBOR implementation |

### Parquet

| Crate | WASM Target | Status |
|-------|-------------|--------|
| `parquet` (arrow-rs) | wasm32-unknown-unknown | **NO** — does not compile |
| `parquet` (arrow-rs) | wasm32-wasi | Partial (SIMD issues) |
| `parquet-wasm` | wasm32-unknown-unknown | **YES** — purpose-built for WASM |

**parquet-wasm** ([kylebarron/parquet-wasm](https://github.com/kylebarron/parquet-wasm)):
- Rust-based WASM bindings for Apache Parquet
- Read and write Parquet to/from Apache Arrow
- Sync and async API
- 1.2 MB brotli-compressed (full), 456 KB (read-only, no compression)
- Available on npm
- Supports remote file reading (range requests)

### Assessment

**All common data formats have pure-Rust WASM-ready implementations.** The Serde ecosystem is comprehensive. Parquet is the only format with a compilation gap, and `parquet-wasm` fills it.

---

## 4. Specialized Data Crates

| Crate | Purpose | Pure Rust | WASM | Notes |
|-------|---------|-----------|------|-------|
| `flatbuffers` | FlatBuffers | Yes | Yes | Zero-copy deserialization |
| `prost` | Protocol Buffers | Yes | Yes | gRPC-compatible protobuf |
| `capnp` | Cap'n Proto | Yes | Yes | Zero-copy serialization |
| `avro-rs` | Apache Avro | Yes | Likely | Schema-based serialization |
| `rosbag` | ROS bag files | Partial | Unknown | May need investigation for razm/core |

### ROS Bag Files

If rosbag conversion is a target use case, the `rosbag` crate exists but WASM readiness is unclear. This would need specific feasibility testing.

---

## Synthesis: Crate Evaluation Matrix

### Image Processing

| Crate | WASM Ready | Pure Rust | Maturity | Recommendation |
|-------|-----------|-----------|----------|----------------|
| `image` | Yes (config needed) | Yes | High | **PRIMARY** — format I/O |
| `photon-rs` | Yes (WASM-first) | Yes | Medium | **PRIMARY** — processing/effects |
| OxiMedia | Claims yes | Yes | **UNVERIFIED** | **AVOID** — quality concerns |
| `resvg` | Yes | Yes | High | **OPTIONAL** — SVG support |
| `tiny-skia` | Yes | Yes | High | **OPTIONAL** — 2D rendering |

### Video Processing

| Crate | WASM Ready | Pure Rust | Maturity | Recommendation |
|-------|-----------|-----------|----------|----------------|
| `rav1e` | Yes (slow) | Mostly | High | **PRIMARY** — AV1 encode |
| `rav1d` | Needs test | Yes | Medium-High | **PRIMARY** — AV1 decode |
| `videocall-codecs` | Yes | Unknown | Low-Medium | **EVALUATE** — VP8/VP9 |
| libvpx (via wasi-sdk) | Possible | No (C) | High | **PLUGIN** — VP9 sidecar |
| openh264 (via wasi-sdk) | Possible | No (C) | High | **PLUGIN** — H.264 sidecar |

### Data Processing

| Crate | WASM Ready | Pure Rust | Maturity | Recommendation |
|-------|-----------|-----------|----------|----------------|
| `polars` | Yes (config needed) | Yes | High | **PRIMARY** — DataFrame ops |
| `arrow-rs` | Partial | Yes | High | **FOUNDATION** — columnar format |
| `parquet-wasm` | Yes | Yes (Rust→WASM) | Medium | **PRIMARY** — Parquet I/O |
| `serde_json` | Yes | Yes | High | **PRIMARY** — JSON |
| `csv` | Yes | Yes | High | **PRIMARY** — CSV |
| `rmp-serde` | Yes | Yes | High | **PRIMARY** — MessagePack |

---

## Critical Gaps Identified

| Gap | Severity | Mitigation |
|-----|----------|------------|
| No pure-Rust VP9 encoder | Medium | Prioritize AV1; VP9 via plugin |
| No pure-Rust H.264/H.265 | Expected | Plugin/sidecar model (openh264 for H.264) |
| Video encoding slow on WASM | High | Document as known limitation; recommend native for production encoding |
| Arrow parquet crate WASM issue | Medium | Use `parquet-wasm` instead |
| rav1d WASM compilation unverified | Medium | Needs feasibility test |
| ROS bag WASM support unclear | Low | Needs investigation if rosbag is in scope |

---

## Sources

- [Polars — GitHub](https://github.com/pola-rs/polars)
- [polars-wasm — GitHub](https://github.com/llalma/polars-wasm)
- [arrow-rs — GitHub](https://github.com/apache/arrow-rs)
- [parquet-wasm — GitHub](https://github.com/kylebarron/parquet-wasm)
- [serde.rs](https://serde.rs/)
- [serde-json-wasm — crates.io](https://crates.io/crates/serde-json-wasm)
- [rmp-serde (MessagePack)](https://github.com/3Hren/msgpack-rust)
- [Photon — GitHub](https://github.com/silvia-odwyer/photon)
