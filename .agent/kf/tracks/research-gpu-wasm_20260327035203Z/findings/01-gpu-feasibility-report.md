# GPU Acceleration from WASM — Feasibility Report

## Research Date: 2026-03-27

---

## 1. wgpu Crate — WASM & GPU Status

**Repository:** [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu)
**Status:** Mature, actively maintained
**MSRV:** Rust 1.87+

### Platform Matrix

| Platform | Backend | Status |
|----------|---------|--------|
| Windows | Vulkan, D3D12 | Stable |
| Linux | Vulkan | Stable |
| macOS | Metal | Stable |
| iOS | Metal | Stable |
| Android | Vulkan | Stable |
| **Browser (WASM)** | **WebGPU** | **Stable** |
| Browser (WASM) | WebGL2 | Stable (fallback) |

### Compute Shader Support

- **Full compute pipeline support** — create compute shaders, dispatch workgroups, read back results
- **WGSL** (WebGPU Shading Language) is the primary shader language
- SPIR-V and GLSL also supported (auto-converted to WGSL on WebGPU targets)
- Workgroup-based synchronization
- Buffer read/write, texture sampling from compute shaders

### WASM-Specific Behavior

- WebGPU backend enabled by default on WASM targets
- WebGL2 available as fallback (but NO compute shader support on WebGL2)
- Same Rust code compiles to native (Vulkan/Metal/D3D12) and WASM (WebGPU)
- **No special feature flags needed** — WASM support is built-in

### Assessment

**STRONG.** wgpu provides a unified GPU API that works identically on native and WASM. Compute shaders work on both paths. This is exactly what rasmcore needs for GPU-accelerated processing.

---

## 2. WebGPU Browser Support

### Current Status (as of early 2026)

| Browser | Status | Compute Shaders | Notes |
|---------|--------|-----------------|-------|
| Chrome | Stable (v113+) | Yes | Full support since 2023 |
| Edge | Stable | Yes | Chromium-based, matches Chrome |
| Firefox | Stable (v141+ Windows, v145+ macOS) | Yes | Linux/Android in progress for 2026 |
| Safari | Stable (v26.0) | Yes | macOS Tahoe 26, iOS 26, iPadOS 26 |
| Chrome Android | Stable (v121+) | Yes | Qualcomm/ARM GPUs, Android 12+ |

**Bottom line:** WebGPU with compute shaders ships in ALL major desktop browsers as of late 2025. Mobile coverage is expanding. This is production-ready for browser-based GPU compute.

---

## 3. WASI-GFX — Server-Side GPU Access

### Proposal: wasi-gfx

**Status:** Phase 2 (WASI proposal process)
**Champions:** Mendy Berger, Sean Isom
**Repository:** [WebAssembly/wasi-gfx](https://github.com/WebAssembly/wasi-gfx)

### What It Provides

Four WIT interface packages:

| Package | Purpose |
|---------|---------|
| `wasi:webgpu` | GPU interaction (mirrors WebGPU API) |
| `wasi:frame-buffer` | CPU-based graphics rendering |
| `wasi:surface` | Display surface + input events |
| `wasi:graphics-context` | Links graphics APIs to windowing |

### Runtime Support

- Runtimes like Wasmtime, WasmEdge, and wasmCloud can map `wasi:webgpu` calls to native GPU APIs (Metal, Vulkan, D3D12)
- `wasi-gfx-runtime` exists as a reference implementation
- Wraps wgpu, Bevy, webgpu.h backends

### Compute-Specific Status

The `wasi:webgpu` interface mirrors the full WebGPU API, which **includes compute pipelines**. A WASM component using `wasi:webgpu` can:
- Create compute shader modules (WGSL)
- Allocate GPU buffers
- Dispatch compute workgroups
- Read back results

This is the same API surface as browser WebGPU, just via WASI host bindings instead of JavaScript.

### Maturity Assessment

| Aspect | Status |
|--------|--------|
| Specification | Phase 2 — not yet standardized |
| Reference runtime | Exists (wasi-gfx-runtime) |
| Wasmtime native support | Not yet in mainline |
| Production readiness | **NOT YET** — experimental |

---

## 4. The Two GPU Paths

### Path A: Browser (WASM + WebGPU) — READY NOW

```
Rust code → wgpu API → compile to WASM → browser loads → WebGPU executes on GPU
```

- Works today in all major browsers
- Full compute shader support
- Same Rust/wgpu code as native
- **Production-ready**

### Path B: Server-Side (WASI + wasi:webgpu) — EXPERIMENTAL

```
Rust code → wgpu API → compile to WASM component → WASI runtime → wasi:webgpu → native GPU
```

- wasi-gfx is Phase 2
- Reference runtime exists but not in mainline wasmtime
- Compute shaders theoretically supported
- **Not production-ready yet**

### The Gap

The browser path works because browsers already implement WebGPU natively. The server-side path requires WASI runtimes to implement the `wasi:webgpu` interface, which is still being standardized.

**For rasmcore:** Browser GPU acceleration is available now. Server-side GPU acceleration is 12-18 months away from production readiness.

---

## 5. Performance Characteristics

### When GPU Beats CPU (from WASM)

| Workload | WASM SIMD (CPU) | WebGPU (GPU) | Winner |
|----------|-----------------|--------------|--------|
| Small image filters (<1MP) | ~35ms | ~20ms + dispatch overhead | **CPU** (dispatch overhead dominates) |
| Large image processing (>4MP) | ~150ms | ~25ms | **GPU** (10x+ faster) |
| Video frame effects (1080p) | ~45ms/frame | ~5ms/frame | **GPU** (massive win) |
| Particle/physics simulation (100K) | ~50ms | ~2ms | **GPU** (25x faster) |
| ML inference (large model) | 2-5 tok/s | 25-40 tok/s | **GPU** (10x faster) |
| ML inference (small model) | ~10ms | ~20ms | **CPU** (dispatch overhead) |
| Data parsing/serialization | Fast | N/A | **CPU** (not parallelizable) |
| Codec encode/decode (sequential) | Moderate | Complex | **CPU** (sequential algorithms) |

### Key Insight: GPU Dispatch Overhead

GPU dispatch from WASM has overhead:
1. Buffer upload (CPU → GPU)
2. Shader dispatch
3. Buffer readback (GPU → CPU)

For small workloads, this overhead exceeds the computation itself. **GPU is only worth it for parallel-heavy, large-data operations.**

### WASM SIMD as Baseline

- WASM SIMD achieves 90-95% of native C++ performance
- 128-bit SIMD registers (vs 256-bit AVX2 native — ~2x gap for SIMD-heavy code)
- Widely supported in all browsers
- **No GPU dispatch overhead** — always available

### Recommended Strategy

| Operation Type | Recommended Path | Rationale |
|----------------|-----------------|-----------|
| Image filters (large) | GPU (WebGPU) | Highly parallel, benefits from GPU |
| Image format decode/encode | CPU (WASM SIMD) | Sequential, I/O-bound |
| Video frame effects | GPU (WebGPU) | Per-pixel parallel operations |
| Video codec encode/decode | CPU (WASM SIMD) | Complex sequential algorithms |
| Data transformation | CPU (WASM) | Not parallelizable |
| Matrix operations (ML) | GPU (WebGPU) | Massively parallel |
| Color space conversion | GPU (WebGPU) | Per-pixel parallel |

---

## 6. Media Processing Operations — GPU Suitability

### Image Processing — HIGH GPU VALUE

| Operation | GPU Suitable | Notes |
|-----------|-------------|-------|
| Resize (bilinear/bicubic) | Yes | Per-pixel computation, highly parallel |
| Color space conversion | Yes | Per-pixel, embarrassingly parallel |
| Convolution filters (blur, sharpen) | Yes | Kernel-based, GPU-native |
| Edge detection (Sobel, Canny) | Yes | Kernel-based |
| Histogram equalization | Partial | Reduction operation, needs careful design |
| Format decode (PNG, JPEG) | No | Sequential, entropy-coded |
| Format encode (PNG, JPEG) | Partial | Some stages parallelizable |

### Video Processing — MIXED GPU VALUE

| Operation | GPU Suitable | Notes |
|-----------|-------------|-------|
| Frame-level effects (color grade) | Yes | Per-pixel per-frame |
| Motion estimation | Partial | Search algorithms, some parallelism |
| AV1/H.264 encode | Mostly No | Complex sequential dependencies |
| AV1/H.264 decode | Mostly No | Entropy decoding is sequential |
| Transcoding pipeline | Hybrid | Decode on CPU, effects on GPU, encode on CPU |
| Scaling/letterboxing | Yes | Per-pixel |

### Data Processing — LOW GPU VALUE

| Operation | GPU Suitable | Notes |
|-----------|-------------|-------|
| CSV/JSON parsing | No | Sequential text parsing |
| Aggregation (sum, count) | Partial | Reduction, but data transfer overhead |
| Sort | Partial | GPU sort exists but complex |
| Filter/select | Partial | Depends on selectivity |
| Join | No | Complex, not well-suited |

---

## 7. Recommendations for rasmcore

### Tier 1: Implement Now (Browser GPU)

- Design image processing interfaces to optionally accept GPU device handles
- Implement GPU-accelerated image filters (blur, sharpen, color grade, resize) using wgpu compute shaders
- These work in browsers today via WebGPU and natively via Vulkan/Metal

### Tier 2: Design For (Server-Side GPU)

- Use `wasi:webgpu` interface patterns when designing WIT interfaces
- Don't block on wasi-gfx standardization — it's 12-18 months from production
- When it arrives, rasmcore GPU modules should "just work" because they use the same wgpu API

### Tier 3: Skip for Now

- GPU-accelerated video codec encode/decode — the algorithms are fundamentally sequential
- GPU for data processing — the data transfer overhead negates benefits for typical workloads

### Architecture Pattern

```wit
// rasmcore:image interface — GPU-aware design
interface processor {
    use rasmcore:core/types.{buffer, image-info};

    // CPU path — always available
    resize: func(input: buffer, info: image-info, width: u32, height: u32) -> result<buffer, error>;

    // GPU path — available when WebGPU/wasi:webgpu is present
    // The host decides whether to provide GPU access
    resize-gpu: func(input: buffer, info: image-info, width: u32, height: u32, device: borrow<gpu-device>) -> result<buffer, error>;
}
```

Alternative: Use feature detection at the world level:

```wit
world image-processor-gpu {
    include image-processor;
    import wasi:webgpu/gpu@1.0.0;
}
```

### WASM SIMD Baseline

Even without GPU, rasmcore modules should use WASM SIMD for CPU-side acceleration:
- 128-bit SIMD is universally supported
- Rust's `std::simd` or `packed_simd` compiles to WASM SIMD automatically
- 90-95% native performance for parallelizable operations

---

## 8. Timeline Estimates

| Capability | Availability | Confidence |
|------------|-------------|------------|
| Browser GPU compute (WebGPU) | **NOW** | HIGH |
| wgpu in WASM (browser) | **NOW** | HIGH |
| WASM SIMD in all browsers | **NOW** | HIGH |
| wasi-gfx Phase 3 (standard) | Mid-Late 2026 | MEDIUM |
| wasi-gfx in Wasmtime mainline | Late 2026 - Early 2027 | LOW-MEDIUM |
| Production server-side GPU | 2027+ | LOW |

---

## Sources

- [wgpu — GitHub](https://github.com/gfx-rs/wgpu)
- [wgpu documentation](https://docs.rs/wgpu/)
- [wasi-gfx — GitHub](https://github.com/WebAssembly/wasi-gfx)
- [wasi-gfx-runtime — GitHub](https://github.com/wasi-gfx/wasi-gfx-runtime)
- [WebGPU supported in all major browsers](https://web.dev/blog/webgpu-supported-major-browsers)
- [WebGPU hits critical mass](https://www.webgpu.com/news/webgpu-hits-critical-mass-all-major-browsers/)
- [WebGPU vs WASM benchmarks](https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/)
- [wgpu compute image filters](https://blog.redwarp.app/image-filters/)
- [Browser video editing WebGPU+WASM](https://byteiota.com/browser-video-editing-webgpu-wasm-performance/)
- [WASM SIMD 2025](https://dev.to/dataformathub/rust-webassembly-2025-why-wasmgc-and-simd-change-everything-3ldh)
- [WebGPU compute exploration](https://github.com/scttfrdmn/webgpu-compute-exploration)
