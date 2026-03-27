# Video Processing Crate Evaluation

## Research Date: 2026-03-27

---

## 1. rav1e — AV1 Encoder

**Repository:** [xiph/rav1e](https://github.com/xiph/rav1e)
**Status:** Active, well-established
**Pure Rust:** Mostly — has optional x86_64 NASM assembly for SIMD optimizations

### Overview

- AV1 encoder by Xiph.org (same org as Opus, Vorbis, FLAC)
- "Designed to eventually cover all use cases"
- Best suited where libaom (reference encoder) is too slow
- Requires Rust 1.74.0+

### WASM Support

- Basic `wasm32-wasi` support added in v0.3.0
- Dedicated WASM bindings: [rav1e_js](https://github.com/rust-av/rav1e_js)
- Self-described as "the slowest and most dangerous AV1 encoder of the web" (tongue-in-cheek)
- Plans for WASM-SIMD and multi-threading improvements
- NASM assembly optimizations NOT available on WASM (pure Rust fallback)

### Performance Considerations

- Native: Competitive, especially with `RUSTFLAGS="-C target-cpu=native"` (11-13% boost with AVX2)
- WASM: Significantly slower than native due to no SIMD assembly, no multithreading
- AV1 encoding is inherently CPU-intensive — WASM overhead is substantial

### Assessment

**USABLE but slow on WASM.** rav1e works for AV1 encoding from WASM but performance will be a major concern. For production video encoding, native execution is strongly preferred. WASM path is viable for:
- Light encoding tasks (thumbnails, preview generation)
- Environments where security sandboxing justifies the performance cost
- Browser-based encoding of short clips

---

## 2. rav1d — AV1 Decoder (Pure Rust)

**Repository:** [memorysafety/rav1d](https://github.com/memorysafety/rav1d)
**Status:** Active, backed by Prossimo/ISRG (Internet Security Research Group)
**Pure Rust:** Yes — Rust port of dav1d

### Overview

- Pure Rust port of dav1d (the fastest C AV1 decoder)
- Funded by memorysafety.org initiative
- v1.1.0 released May 2025, synced with dav1d v1.5.1
- Fully functional — works in Chromium
- Drop-in C API replacement for libdav1d

### Performance

- Currently ~5% slower than C dav1d
- Active optimization efforts (community contest)
- Goal: match or exceed dav1d performance

### WASM Support

- Not explicitly documented for WASM targets
- As a pure Rust project, should compile to WASM with proper configuration
- May need `default-features = false` to disable platform-specific SIMD
- **Needs feasibility testing** for wasm32-wasip2

### Assessment

**HIGH VALUE for razm/core.** A security-focused pure-Rust AV1 decoder backed by a credible organization (ISRG/memorysafety.org). 5% slower than C is acceptable. WASM compilation needs verification but is architecturally feasible.

---

## 3. VP9 / VP8 — Limited Pure Rust Options

### vpx-rs (C wrapper)

- Rust wrapper around libvpx (C library)
- **NOT pure Rust** — requires libvpx system dependency
- Not suitable for our WASM-only approach

### videocall-codecs

- Cross-platform VP8/VP9 library with WASM support
- Updated January 2026
- Decoder runs in Web Workers for browser use
- **Most promising VP9 option for WASM**

### SkeevingQuack/rust-vp9

- Learning project — VP9 decoder in Rust
- Not production-ready

### Assessment

**GAP: No mature pure-Rust VP9 encoder/decoder.** Options are:
1. Use `videocall-codecs` (needs maturity evaluation)
2. Focus on AV1 instead (better codec, better Rust support)
3. Accept VP9 as a plugin target (compile libvpx to WASM via wasi-sdk)
4. Wait for ecosystem to mature

**Recommendation:** Prioritize AV1 over VP9 for the MVP. AV1 has better Rust tooling and is the superior codec technically.

---

## 4. Open Codec Landscape Survey

### Encoders Available in Pure Rust

| Codec | Crate | Status | WASM | Notes |
|-------|-------|--------|------|-------|
| AV1 | `rav1e` | Stable | Yes (slow) | Primary video codec choice |
| FLAC | `claxon` / `flac` | Stable | Yes | Audio, lossless |
| Opus | `opus-rs` (C wrapper) | Stable | Needs wasi-sdk | Not pure Rust |
| Vorbis | `lewton` (decoder) | Stable | Yes | Decoder only |

### Decoders Available in Pure Rust

| Codec | Crate | Status | WASM | Notes |
|-------|-------|--------|------|-------|
| AV1 | `rav1d` | Active | Needs test | Best-in-class pure Rust |
| VP8 | `videocall-codecs` | Active | Yes | Needs maturity check |
| VP9 | `videocall-codecs` | Active | Yes | Needs maturity check |
| H.264 | None pure Rust | — | — | Patent-encumbered + no pure Rust impl |
| H.265 | None pure Rust | — | — | Patent-encumbered + no pure Rust impl |
| Theora | `theora` | Minimal | Unknown | Legacy codec |
| MP3 | `minimp3` / `symphonia` | Stable | Yes | Decoder, patents expired |

### Container Format Support

| Format | Crate | Status | WASM | Notes |
|--------|-------|--------|------|-------|
| MP4/M4A | `mp4parse` | Stable | Yes | Mozilla-backed |
| MKV/WebM | `matroska` | Active | Yes | Pure Rust |
| OGG | `ogg` | Stable | Yes | Pure Rust |
| FLV | Limited | — | — | Gap |
| MPEG-TS | Limited | — | — | Gap |

---

## 5. Patent-Encumbered Codec Status

These codecs have NO pure-Rust implementations and are patent-encumbered — they are candidates for the sidecar plugin model:

| Codec | Status | Plugin Strategy |
|-------|--------|-----------------|
| H.264/AVC | No pure Rust | Compile openh264 to WASM via wasi-sdk (Cisco's BSD-licensed impl) |
| H.265/HEVC | No pure Rust | Compile x265 to WASM (GPL) or kvazaar (LGPL) |
| H.266/VVC | No pure Rust | Very limited tooling |
| AAC | No pure Rust | Compile fdk-aac to WASM |
| AC-3/E-AC-3 | No pure Rust | Compile a]52dec to WASM |

**Key insight:** openh264 (Cisco) is BSD-licensed and Cisco pays the MPEG-LA royalties, making it the most practical H.264 path for a sidecar plugin.

---

## Recommendations for razm/core Video Module

### MVP Codec Stack

| Role | Choice | Rationale |
|------|--------|-----------|
| Video Encoder | `rav1e` (AV1) | Best pure-Rust encoder, WASM support exists |
| Video Decoder | `rav1d` (AV1) | Pure Rust, backed by ISRG, near-native perf |
| Audio | `symphonia` | Pure Rust multi-format audio decoder |
| Container | `mp4parse` + `matroska` | MP4 and MKV/WebM support |

### Plugin Candidates (Post-MVP)

| Codec | Source | Distribution |
|-------|--------|-------------|
| H.264 | openh264 → WASM | Sidecar plugin (BSD license) |
| H.265 | x265 → WASM | Sidecar plugin (GPL, license concern) |
| VP9 | libvpx → WASM | Sidecar plugin (BSD) |

### Critical Limitation

**Video encoding in WASM is slow.** For production transcoding workloads, native execution should be preferred. The WASM path is best for:
- Sandboxed/secure environments
- Browser-based processing
- Light operations (decode, frame extraction, thumbnail generation)
- Proof-of-concept and portability

---

## Sources

- [rav1e — GitHub](https://github.com/xiph/rav1e)
- [rav1e_js — GitHub](https://github.com/rust-av/rav1e_js)
- [rav1d — GitHub](https://github.com/memorysafety/rav1d)
- [rav1d — memorysafety.org](https://www.memorysafety.org/initiative/av1/)
- [videocall-codecs — crates.io](https://crates.io/crates/videocall-codecs)
- [Av1an — GitHub](https://github.com/rust-av/Av1an)
