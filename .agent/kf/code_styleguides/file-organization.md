# File Organization

## One filter per file

Every filter MUST be in its own file under `src/domain/filters/<category>/<name>.rs`.

Each filter file contains:
- The `#[register_filter/mapper/compositor/generator]` annotation + pub fn
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

Every new filter SHOULD implement `GpuCapable` when the operation is
expressible as a compute shader. This is the default expectation — only
skip GPU if the algorithm is fundamentally unsuitable (e.g., recursive
flood fill, iterative content-aware seam carving).

### How to add GPU support

1. **Write a WGSL compute shader** in `src/shaders/<name>.wgsl`
   - Follow existing shader patterns (box_blur.wgsl, skeletonize.wgsl)
   - Input/output are `array<u32>` (packed RGBA8)
   - Params via `var<uniform>` struct (little-endian, 4-byte aligned)
   - Workgroup size: `@workgroup_size(16, 16, 1)` for 2D image ops
   - Extra data (kernels, LUTs) via additional `storage<read>` bindings

2. **Implement `GpuCapable` on ConfigParams**:
   ```rust
   const MY_FILTER_WGSL: &str = include_str!("../../../shaders/my_filter.wgsl");

   impl rasmcore_pipeline::GpuCapable for MyFilterParams {
       fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
           let mut params = Vec::with_capacity(16);
           params.extend_from_slice(&width.to_le_bytes());
           params.extend_from_slice(&height.to_le_bytes());
           // ... add filter-specific params
           Some(vec![rasmcore_pipeline::GpuOp {
               shader: MY_FILTER_WGSL,
               entry_point: "main",
               workgroup_size: [16, 16, 1],
               params,
               extra_buffers: Vec::new(),
           }])
       }
   }
   ```

3. **Add GPU parity test** — verify GPU ops are generated and validate
   CPU output matches expected behavior:
   ```rust
   #[test]
   fn gpu_ops_generated() {
       let params = MyFilterParams { ... };
       use rasmcore_pipeline::GpuCapable;
       let ops = params.gpu_ops(100, 100).unwrap();
       assert!(!ops.is_empty());
   }
   ```

### When to skip GPU

Return `None` from `gpu_ops()` if:
- The pixel format is unsupported (non-RGBA8)
- Parameters are extreme (would exceed GPU buffer limits)
- The algorithm requires CPU-only features (random file I/O, etc.)

The pipeline falls back to CPU automatically when `gpu_ops()` returns `None`.

## Agent concurrency for bulk operations

When a track involves moving/splitting many files, use subagents with
`isolation: "worktree"` to handle batches in parallel. Parent agent reviews
each subagent's diff before merging.
