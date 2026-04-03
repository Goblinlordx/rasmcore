# File Organization

## One filter per file

Every filter MUST be in its own file under `src/domain/filters/<category>/<name>.rs`.

Each filter file contains:
- The `#[derive(Filter)]` struct + `impl CpuFilter` (V2 filters)
- The `#[register_mapper/compositor/generator]` annotation (V1 non-filter types)
- The `ConfigParams` struct (if any) + its impl blocks
- Private helper functions used ONLY by this filter
- `use crate::domain::filters::common::*;` for shared types and helpers

## Shared code

Shared helpers, enums, constants, and structs live in `filters/common.rs`.
Individual filter files MUST NOT call other filter functions directly — use
shared helpers from `common.rs` or domain modules (`point_ops`, `color_grading`, etc.).

## When adding a new filter

1. Create `filters/<category>/<name>.rs`
2. Add `mod <name>; pub use <name>::*;` to `filters/<category>/mod.rs`
3. Add ConfigParams struct + registration annotation + function
4. `use crate::domain::filters::common::*;` for shared imports
5. **Add GPU support** (see GPU section below)
6. Add tests including GPU parity validation
7. Verify: `cargo build && cargo component build`

## GPU support (REQUIRED for new filters)

Every new filter SHOULD implement `GpuFilter` when the operation is
expressible as a compute shader. This is the default expectation — only
skip GPU if the algorithm is fundamentally unsuitable (e.g., recursive
flood fill, iterative content-aware seam carving).

### How to add GPU support

1. **Write a WGSL compute shader** in `src/shaders/<name>_f32.wgsl`
   - Input/output are `array<vec4<f32>>` (F32Vec4 format)
   - Params via `var<uniform>` struct (little-endian, 4-byte aligned)
   - First two fields MUST be `width: u32, height: u32`
   - Workgroup size: `@workgroup_size(16, 16, 1)` for 2D image ops
   - Use `shaders::with_sampling_f32()` for distortion (bilinear sample)
   - Use `shaders::with_io_f32()` for per-pixel ops (load/store)

2. **Implement `GpuFilter` on the filter struct**:
   ```rust
   impl GpuFilter for MyFilterParams {
       fn gpu_ops(&self, w: u32, h: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
           self.gpu_ops_with_format(w, h, rasmcore_pipeline::gpu::BufferFormat::F32Vec4)
       }

       fn gpu_ops_with_format(&self, w: u32, h: u32, fmt: BufferFormat)
           -> Option<Vec<GpuOp>>
       {
           static SHADER: LazyLock<String> = LazyLock::new(|| {
               rasmcore_gpu_shaders::with_io_f32(include_str!("../../../shaders/my_filter_f32.wgsl"))
           });
           let mut params = Vec::with_capacity(16);
           params.extend_from_slice(&w.to_le_bytes());
           params.extend_from_slice(&h.to_le_bytes());
           // ... add filter-specific params
           Some(vec![GpuOp::Compute {
               shader: SHADER.clone(),
               entry_point: "main",
               workgroup_size: [16, 16, 1],
               params,
               extra_buffers: vec![],
               buffer_format: BufferFormat::F32Vec4,
           }])
       }
   }
   ```

3. **Add GPU parity test** — verify GPU output matches CPU within f32 tolerance.

### When to skip GPU

Return `None` from `gpu_ops()` if:
- The pixel format is unsupported (non-RGBA8)
- The algorithm has true sequential dependencies that can't be parallelized

The pipeline falls back to CPU automatically when `gpu_ops()` returns `None`.

## Numeric precision — f32 vs f64

**Default: f32.** All pixel processing, filter kernels, and GPU shaders use f32.
WGSL does not support f64, so any GPU-capable code path must be f32.

**Use f64 ONLY for:**
- **Color science math** — sRGB↔linear transfer functions, CIE Lab/Luv/OKLab
  conversions, Bradford chromatic adaptation, delta-E calculations. These involve
  chained operations on small differences between large numbers where f32
  accumulates visible error (~1e-7 per op vs ~1e-16 for f64).
- **Geometric matrix operations** — homography solve (8×8 inversion), perspective
  warp matrix computation. Small determinants lose significance in f32.
- **Statistical accumulation over large images** — variance, covariance, histogram
  moments. Naive sum-of-squares in f32 hits catastrophic cancellation at >1M pixels.

**Never f64 for:**
- Per-pixel filter operations (blur, sharpen, color adjust) — f32 is exact for 8/16-bit pixels
- GPU shader code — WGSL has no f64 support
- Kernel weights, interpolation coefficients — f32 precision (24-bit mantissa) far
  exceeds 8-bit (256 values) or 16-bit (65536 values) pixel depth

**Pattern:** Color science helpers in `rasmcore-kernel::color_spaces` use f64 internally
and convert to/from f32 at the pixel boundary. Filter code stays f32 throughout.

## Agent concurrency for bulk operations

When a track involves moving/splitting many files, use subagents with
`isolation: "worktree"` to handle batches in parallel. Parent agent reviews
each subagent's diff before merging.
