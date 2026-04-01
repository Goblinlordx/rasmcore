# GPU Offload Architecture Design

**Date:** 2026-04-02
**Status:** Design complete — ready for implementation planning
**PoC Results:** 36-145x speedup at 1024x1024 (blur, spin blur, spherize)

---

## 1. Executive Summary

The rasmcore pipeline can offload expensive image operations to GPU via
host-injected WebGPU compute shaders. The WASM component emits WGSL shader
source + parameters; the host runs them on whatever GPU runtime is available
(browser WebGPU, Python wgpu-py, Go wgpu-native, Rust wgpu).

**Key design principle:** The WASM component is platform-agnostic. It produces
WGSL + params. The host decides how to execute them.

---

## 2. GpuCapable Trait

### Definition

```rust
/// A GPU operation — one compute shader dispatch.
pub struct GpuOp {
    /// WGSL compute shader source code.
    pub shader: &'static str,
    /// Entry point function name (e.g., "main", "blur_h").
    pub entry_point: &'static str,
    /// Workgroup dispatch hint: (x, y, z) workgroup counts are computed
    /// at runtime as ceil(width/hint.0, height/hint.1, 1).
    pub workgroup_size: [u32; 3],
    /// Serialized uniform params (filter-specific, packed as bytes).
    pub params: Vec<u8>,
}

/// Trait for filters that can run on GPU.
///
/// Implemented alongside the CPU path. When GPU is available, the pipeline
/// calls gpu_ops() instead of compute_region().
pub trait GpuCapable {
    /// Return the GPU operations for this filter with current config.
    /// Returns None if the filter cannot run on GPU for this specific
    /// configuration (e.g., unsupported pixel format, extreme parameters).
    fn gpu_ops(&self) -> Option<Vec<GpuOp>>;
}
```

### How Filters Opt In

Filters implement `GpuCapable` alongside `ImageNode`. The WGSL shader is a
`const &str` embedded in the filter module:

```rust
// In filters/spatial/blur.rs:
const BLUR_H_WGSL: &str = include_str!("shaders/blur_h.wgsl");
const BLUR_V_WGSL: &str = include_str!("shaders/blur_v.wgsl");

impl GpuCapable for BlurNode {
    fn gpu_ops(&self) -> Option<Vec<GpuOp>> {
        let radius = self.config.radius;
        let sigma = radius / 3.0;
        let kernel = gaussian_kernel(radius as u32, sigma);
        
        // Separable: horizontal pass then vertical pass
        Some(vec![
            GpuOp {
                shader: BLUR_H_WGSL,
                entry_point: "blur_h",
                workgroup_size: [256, 1, 1],
                params: pack_blur_params(radius, sigma, &kernel),
            },
            GpuOp {
                shader: BLUR_V_WGSL,
                entry_point: "blur_v",
                workgroup_size: [1, 256, 1],
                params: pack_blur_params(radius, sigma, &kernel),
            },
        ])
    }
}
```

### Param Serialization

Each filter serializes its config to a byte buffer matching the WGSL uniform
struct layout. This is filter-specific — the trait returns opaque bytes, and
the WGSL shader knows the layout.

Convention: params are packed as little-endian, 4-byte aligned, matching
WGSL `uniform` struct rules (std140/std430 layout).

---

## 3. WIT Import Protocol

### Definition

```wit
interface gpu {
    /// A single GPU compute operation.
    record gpu-op {
        /// WGSL compute shader source code.
        shader: string,
        /// Entry point function name.
        entry-point: string,
        /// Workgroup size hint (x, y, z).
        workgroup-x: u32,
        workgroup-y: u32,
        workgroup-z: u32,
        /// Serialized uniform parameters (filter-specific).
        params: buffer,
    }

    /// Error from GPU execution.
    variant gpu-error {
        /// GPU not available or initialization failed.
        not-available(string),
        /// Shader compilation error.
        shader-error(string),
        /// Execution error.
        execution-error(string),
    }

    /// Execute a batch of GPU operations on pixel data.
    ///
    /// The host chains operations: output of op[i] = input of op[i+1].
    /// All intermediate buffers stay in GPU memory (no round-trips).
    /// Only the final output is read back to CPU.
    ///
    /// Input: RGBA8 pixel data, width x height x 4 bytes.
    /// Output: RGBA8 pixel data, same dimensions.
    gpu-execute: func(
        ops: list<gpu-op>,
        input: buffer,
        width: u32,
        height: u32,
    ) -> result<buffer, gpu-error>;
}
```

### Optional Import

The `gpu` interface is an **optional** WIT import. If the host doesn't
provide it, the pipeline falls back to CPU execution (current behavior).

Detection at runtime:

```rust
fn has_gpu() -> bool {
    // Check if the gpu-execute import is available
    // (WIT optional imports return None/trap if absent)
    cfg!(feature = "gpu-offload") // or runtime feature flag
}
```

### Platform Implementations

| Platform | GPU Runtime | How host provides gpu-execute |
|----------|------------|-------------------------------|
| Browser  | WebGPU     | JS host calls `navigator.gpu` APIs |
| Python   | wgpu-py    | Python host wraps wgpu-py compute |
| Go       | wgpu-native | Go host wraps wgpu-native C bindings |
| Rust     | wgpu       | Rust host wraps wgpu directly |
| No GPU   | (none)     | Import absent — CPU fallback |

---

## 4. Batch Detection Algorithm

### Graph Walker Modification

During `execute()`, the graph walker currently processes nodes one at a time:

```
for each node in topological order:
    pixels = node.compute_region(request, upstream_fn)
```

With GPU offload, it scans ahead for consecutive GPU-capable nodes:

```
for each node in topological order:
    if node implements GpuCapable AND gpu_available:
        batch = collect consecutive GPU-capable nodes starting here
        gpu_ops = flatten all GpuOps from batch
        output = gpu_execute(gpu_ops, input_pixels, width, height)
        skip ahead past the batch
    else:
        pixels = node.compute_region(request, upstream_fn)
```

### Batch Collection Rules

1. **Start:** First GPU-capable node after a CPU node (or source)
2. **Continue:** Each subsequent node that implements GpuCapable
3. **Stop:** First node that does NOT implement GpuCapable, OR end of chain
4. **Multiple batches:** Allowed. CPU nodes between GPU batches cause a
   GPU→CPU→GPU round-trip (readback + re-upload)
5. **Single-node batches:** Allowed. Even one GPU op can be faster than CPU
   if the op is expensive enough (e.g., bilateral filter)

### Buffer Flow

```
Source (CPU) → [GPU batch 1: blur, sepia, contrast] → CPU node: convolve → [GPU batch 2: spherize] → Write (CPU)
     ↓                    ↓                ↓              ↓                     ↓                    ↓
  read()        upload → 3 shaders    readback      compute_region()     upload → 1 shader     readback → writePng
                 (all in GPU mem)                                         (GPU mem)
```

---

## 5. Buffer Management

### CPU→GPU Transfer

The last CPU node's output (raw RGBA8 pixels) is passed as the `input`
buffer to `gpu_execute()`. The host uploads this to a GPU storage buffer
via `queue.writeBuffer()` (browser) or equivalent.

### GPU Internal Buffers

Between ops within a batch, data stays in GPU memory. The host creates
two storage buffers (ping-pong) and alternates input/output roles:

```
Op 1: read from buf_a, write to buf_b
Op 2: read from buf_b, write to buf_a
Op 3: read from buf_a, write to buf_b
...
```

### GPU→CPU Transfer

After the last op in a batch, the host reads back from the output buffer
via `mapAsync()` (browser). This is the most expensive part (~1.5ms for
1024x1024 RGBA8). The result is returned as the `buffer` in the WIT result.

### SharedArrayBuffer Optimization (Future)

If the WASM linear memory is backed by a SharedArrayBuffer (requires
COOP/COEP headers, already set), the host could potentially map the GPU
buffer directly into WASM memory space, avoiding the copy. This is a
future optimization — the initial implementation uses explicit copies.

---

## 6. CLUT Fusion Interaction

### Decision Logic

The pipeline currently has `fuse_color_ops()` which composes consecutive
color operations (sepia, hue_rotate, contrast, etc.) into a single 3D LUT.
This is O(1) per pixel on CPU — very fast.

With GPU available, the decision is:

1. **If the color ops are BETWEEN two GPU spatial ops:** Keep them as
   individual GPU shaders. They're trivially parallel and staying on GPU
   avoids two transfers. A color op shader runs in ~0.1ms on GPU.

2. **If the color ops are standalone (no adjacent GPU ops):** Use CLUT
   fusion on CPU. The LUT lookup is ~0.3ms for 1024x1024 — faster than
   GPU transfer overhead (~1.5ms upload + 1.5ms readback = 3ms).

3. **If the ENTIRE chain is color ops only:** CPU CLUT fusion wins.
   No point going to GPU for pure point operations with no spatial ops.

Implementation: `fuse_color_ops()` runs BEFORE batch detection. Fused
CLUT nodes do NOT implement GpuCapable (they're already optimized).
Individual unfused color nodes DO implement GpuCapable (for batching
between spatial ops).

---

## 7. Multi-Pass Operations

### Separable Filters

Gaussian blur is separable: horizontal pass then vertical pass. This is
expressed as two GpuOps in the same filter's `gpu_ops()`:

```rust
fn gpu_ops(&self) -> Option<Vec<GpuOp>> {
    Some(vec![
        GpuOp { shader: BLUR_H_WGSL, entry_point: "blur_h", ... },
        GpuOp { shader: BLUR_V_WGSL, entry_point: "blur_v", ... },
    ])
}
```

The host executes them sequentially with ping-pong buffers. Both passes
stay in GPU memory — no intermediate readback.

### Multi-Pass with Different Dimensions

Some ops change dimensions (e.g., resize). These CANNOT be part of a GPU
batch because the buffer size changes. They act as batch boundaries.

---

## 8. Fallback Strategy

### No GPU Available

If the host doesn't provide the `gpu-execute` import, the pipeline
runs entirely on CPU — current behavior, no changes needed.

### GPU Execution Fails

If `gpu_execute()` returns a `gpu-error`, the pipeline falls back to
CPU execution for that batch:

```rust
match gpu_execute(ops, input, w, h) {
    Ok(output) => output,
    Err(_) => {
        // Fallback: execute each node on CPU
        for node in batch {
            pixels = node.compute_region(request, upstream_fn);
        }
        pixels
    }
}
```

### Per-Node Fallback

If a specific filter's `gpu_ops()` returns `None` (e.g., unsupported
config), it becomes a batch boundary. The GPU batch ends, the node
runs on CPU, and a new GPU batch may start after it.

---

## 9. Shader Embedding Strategy

### File Organization

```
crates/rasmcore-image/src/domain/filters/
├── spatial/
│   ├── blur.rs                    (CPU implementation)
│   ├── shaders/
│   │   ├── blur_h.wgsl            (GPU horizontal pass)
│   │   └── blur_v.wgsl            (GPU vertical pass)
│   ├── spin_blur.rs
│   ├── shaders/
│   │   └── spin_blur.wgsl
...
```

Shaders are loaded via `include_str!("shaders/blur_h.wgsl")` — embedded
in the WASM binary at compile time.

### Compile-Time Validation (Future)

Use `naga` crate in build.rs to parse and validate WGSL shaders at
compile time. This catches shader syntax errors before deployment.

### Param Serialization Convention

Each filter defines a `pack_params()` function that serializes its
config to bytes matching the WGSL uniform struct:

```rust
fn pack_blur_params(radius: f32, sigma: f32, kernel: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(width as u32).to_le_bytes());  // offset 0
    buf.extend_from_slice(&(height as u32).to_le_bytes()); // offset 4
    buf.extend_from_slice(&(radius as u32).to_le_bytes()); // offset 8
    buf.extend_from_slice(&sigma.to_le_bytes());           // offset 12
    // kernel weights follow as a separate storage buffer
    buf
}
```

---

## 10. GPU Capability Catalog

### Tier 1: GPU-Easy (36-145x speedup, trivial shaders)

These have direct WGSL translations with massive speedups:

| Category | Filters | Shader Pattern |
|----------|---------|---------------|
| Spatial  | blur, box_blur, gaussian_blur_cv, sharpen, median | Neighborhood sampling with shared memory |
| Spatial  | spin_blur, motion_blur, zoom_blur, tilt_shift | Per-pixel multi-tap sampling |
| Spatial  | bilateral, guided_filter, nlm_denoise | Expensive neighborhood with range weighting |
| Distortion | spherize, swirl, barrel, ripple, wave, polar, depolar | Per-pixel inverse coord transform + bilinear |
| Color | sepia, hue_rotate, saturate, contrast, brightness, gamma, exposure | Per-pixel matrix/math (trivial) |
| Color | color_balance, vibrance, white_balance, channel_mixer | Per-pixel channel math |
| Effect | oil_paint, pixelate, halftone | Per-pixel with neighborhood or quantization |

**Count: ~45 filters** — covers the most impactful ops.

### Tier 2: GPU-Possible (moderate benefit, complex shaders)

| Category | Filters | Challenge |
|----------|---------|-----------|
| Enhancement | clahe | Per-tile histogram — needs workgroup reduction |
| Edge | sobel, canny, laplacian | Gradient convolution — straightforward |
| Morphology | erode, dilate, open, close | Neighborhood min/max — similar to median |
| Threshold | adaptive_threshold | Per-block statistics |
| Draw | draw_rect, draw_circle, draw_line | Conditional per-pixel — low compute |

**Count: ~25 filters** — moderate effort, moderate benefit.

### Tier 3: CPU-Only (GPU not beneficial or impractical)

| Category | Filters | Reason |
|----------|---------|--------|
| Grading | curves_*, hue_vs_sat, gradient_map | LUT-based, already O(1) via CLUT fusion |
| Tool | seam_carve, smart_crop | Sequential/global algorithms |
| Generator | noise generators | Random number generation on GPU is tricky |
| Advanced | connected_components, hough_lines | Global analysis, not parallelizable |
| Draw | draw_text, draw_text_ttf | Font rendering, not compute-bound |

**Count: ~40 filters** — keep on CPU.

### Priority Order for Shader Implementation

1. **blur** (r=20 showed 36x speedup, most commonly used)
2. **spherize, swirl, barrel** (distortion showed 145x, common in creative work)
3. **spin_blur, motion_blur, zoom_blur** (58x speedup)
4. **bilateral, nlm_denoise** (very expensive on CPU, huge GPU win)
5. **Point ops: sepia, contrast, brightness, hue_rotate** (keep data on GPU between spatial ops)
6. **sharpen, median** (common post-processing)
7. **Remaining spatial + distortion ops**

---

## 11. Implementation Plan — Follow-Up Tracks

### Track 1: GPU Trait + WIT Import (Foundation)
- Define `GpuCapable` trait and `GpuOp` struct in rasmcore-pipeline
- Define WIT `gpu` interface with `gpu-execute` import
- Add optional GPU feature flag to rasmcore-image
- Implement batch detection in graph walker (with CPU fallback)
- No shaders yet — infrastructure only

### Track 2: Browser Host (WebGPU Handler)
- JS-side `gpu-execute` implementation using WebGPU
- Ping-pong buffer management for multi-op batches
- Pipeline/shader caching (compile once, reuse)
- Integration with web-ui worker (pass GPU handler to WASM)

### Track 3: Shader Implementations — Spatial Ops
- WGSL shaders for: blur, sharpen, bilateral, spin_blur, motion_blur,
  zoom_blur, tilt_shift, median, box_blur, gaussian_blur_cv
- GpuCapable impl for each filter node
- Param pack functions matching WGSL uniform layouts
- ~10 shaders, highest impact

### Track 4: Shader Implementations — Distortion + Point Ops
- WGSL shaders for: spherize, swirl, barrel, ripple, wave, polar, depolar
- WGSL shaders for: sepia, hue_rotate, contrast, brightness, gamma,
  saturate, color_balance, vibrance, exposure
- ~18 shaders, enables full GPU chains

### Track 5 (Future): Non-Browser Hosts
- Python wgpu-py host implementation
- Rust wgpu host implementation
- Go wgpu-native host implementation

### Track 6 (Future): Advanced Optimizations
- SharedArrayBuffer zero-copy transfer
- Compile-time WGSL validation via naga
- GPU timestamp queries for per-op profiling
- Adaptive GPU/CPU decision based on image size

---

## 12. Tiling Interaction

### Current Tiling Architecture

The pipeline uses demand-driven tiling: the write sink requests tiles,
and each node computes only the requested region (with overlap expansion
for spatial ops). This is efficient for CPU because:
- Small tiles fit in L1/L2 cache
- Overlap expansion is modest (blur r=20 adds ~60px border)
- Sequential tile processing avoids full-image memory allocation

### GPU and Tiling — Key Insight

**The current small tiles may be causing much of the WASM slowdown.**
Each tile dispatch has overhead: function call, cache miss on first access,
overlap region re-computation. For spatial ops, the overlap is computed
redundantly between adjacent tiles.

**GPU wants LARGE tiles or full-image dispatch:**
- GPU shader dispatch has fixed overhead (~0.1ms)
- GPU parallelism scales with data size — more pixels = better utilization
- Transfer overhead (upload + readback) is per-dispatch, not per-pixel
- Small tiles mean many dispatches with high per-dispatch overhead

### Recommended Strategy

When GPU is available:

1. **Full-image dispatch for GPU batches.** Don't tile GPU ops. Pass the
   entire image (or very large tiles, e.g., 4096x4096) as a single
   dispatch. The GPU handles the parallelism internally.

2. **Tile size adaptation.** If the pipeline must tile (e.g., 16K image
   that exceeds GPU buffer limits), use much larger tiles for GPU:
   - CPU tile size: 256x256 or 512x512 (cache-friendly)
   - GPU tile size: 4096x4096 or full image (minimize dispatch count)

3. **Overlap is free on GPU.** For spatial ops, the GPU shader reads
   from the full input buffer — no separate overlap region computation.
   The `expand_uniform()` and `crop_to_request()` dance in the CPU
   pipeline becomes unnecessary when the shader reads directly.

4. **Skip the tile walker for GPU batches.** Instead of the write sink
   requesting small tiles backward through the graph, GPU batches
   process the full image forward. The write sink receives the complete
   result and tiles it for output format encoding (JPEG MCU rows, etc.)

### Performance Implication

The PoC showed 140ms for blur on CPU — but this includes tiled execution
overhead. With full-image GPU dispatch at 3.9ms, the tiling overhead is
likely a significant portion of the CPU time. A future optimization:
even WITHOUT GPU, processing larger tiles or full-image on CPU might
improve WASM performance significantly.

---

## 13. Open Questions

1. **Buffer format:** Should GPU ops always use RGBA8, or support other
   formats (RGB8, Gray8, F32)? RGBA8 is simplest and matches WebGPU
   storage buffer alignment. Format conversion can happen at batch
   boundaries.

2. **Dimension changes:** Ops like resize change dimensions. Should these
   be GPU-capable with a different buffer size, or always be batch
   boundaries? Initially: batch boundaries (simplest).

3. **Kernel/LUT data:** Blur kernels and LUT tables need to be passed as
   additional storage buffers. The `params` field in GpuOp is for uniform
   data only. Consider adding an `extra_buffers: Vec<Vec<u8>>` field.

4. **Shader deduplication:** Multiple blur nodes with different radii use
   the same shader but different params. The host should cache compiled
   pipelines by shader source hash, not by params.
