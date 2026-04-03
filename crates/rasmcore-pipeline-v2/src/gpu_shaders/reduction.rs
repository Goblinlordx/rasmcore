//! Reusable 3-pass zero-atomic GPU reduction.
//!
//! Provides workgroup-local tree reduction → global reduce → apply pattern
//! for computing image-wide statistics (channel sums, min/max, histograms)
//! without atomic operations or CPU-GPU roundtrips.
//!
//! # 3-Pass Pattern
//!
//! ```text
//! Pass 1 (local reduce):
//!   Each workgroup tree-reduces its pixels in shared memory.
//!   Thread 0 writes the workgroup's partial result to reduction_buffer[wg_id].
//!   All threads pass pixels through (output = input).
//!
//! Pass 2 (global reduce):
//!   Workgroup 0 reads all partial results, tree-reduces to final.
//!   Writes final result to reduction_buffer[0].
//!   All workgroups pass pixels through.
//!
//! Pass 3 (apply — filter-specific):
//!   Each thread reads reduction_buffer[0] and transforms its pixel.
//!   Caller provides this shader; the module provides a WGSL reader snippet.
//! ```

use crate::node::{GpuShader, ReductionBuffer};

/// What kind of reduction to perform.
#[derive(Debug, Clone, Copy)]
pub enum ReductionKind {
    /// Sum RGB channels + pixel count → `vec4<f32>(sum_r, sum_g, sum_b, count)`.
    ChannelSum,
    /// Per-channel min and max → 2 × `vec4<f32>`:
    /// `[0] = (min_r, min_g, min_b, _)`, `[1] = (max_r, max_g, max_b, _)`.
    ChannelMinMax,
}

/// Configuration for a GPU reduction.
pub struct GpuReduction {
    pub kind: ReductionKind,
    pub workgroup_size: u32,
}

/// Output of `build_passes()` — the two reduction shaders + buffer metadata.
pub struct ReductionPasses {
    pub pass1: GpuShader,
    pub pass2: GpuShader,
    pub buffer_id: u32,
    pub buffer_size: usize,
    pub num_workgroups: u32,
}

impl GpuReduction {
    /// Create a ChannelSum reduction with the given workgroup size.
    pub fn channel_sum(workgroup_size: u32) -> Self {
        Self {
            kind: ReductionKind::ChannelSum,
            workgroup_size,
        }
    }

    /// Create a ChannelMinMax reduction with the given workgroup size.
    pub fn channel_min_max(workgroup_size: u32) -> Self {
        Self {
            kind: ReductionKind::ChannelMinMax,
            workgroup_size,
        }
    }

    /// Generate passes 1 and 2 (local reduce + global reduce).
    ///
    /// The caller appends their own pass 3 (apply) using [`read_buffer()`]
    /// to get a read-only `ReductionBuffer` with the same ID.
    pub fn build_passes(&self, width: u32, height: u32) -> ReductionPasses {
        let total = width * height;
        let wg = self.workgroup_size;
        let num_wg = total.div_ceil(wg);

        match self.kind {
            ReductionKind::ChannelSum => {
                self.build_channel_sum_passes(width, height, total, num_wg)
            }
            ReductionKind::ChannelMinMax => {
                self.build_channel_min_max_passes(width, height, total, num_wg)
            }
        }
    }

    /// WGSL snippet for pass 3 to declare and read the reduction result.
    ///
    /// Declares the reduction buffer binding. The caller's shader uses the
    /// provided helper function to access the result.
    pub fn result_reader_wgsl(&self, binding: u32) -> String {
        match self.kind {
            ReductionKind::ChannelSum => format!(
                "@group(0) @binding({binding}) var<storage, read> _reduction: array<vec4<f32>>;\n\
                 fn reduction_channel_sums() -> vec4<f32> {{ return _reduction[0]; }}\n"
            ),
            ReductionKind::ChannelMinMax => format!(
                "@group(0) @binding({binding}) var<storage, read> _reduction: array<vec4<f32>>;\n\
                 fn reduction_min() -> vec3<f32> {{ return _reduction[0].xyz; }}\n\
                 fn reduction_max() -> vec3<f32> {{ return _reduction[1].xyz; }}\n"
            ),
        }
    }

    /// Build a read-only `ReductionBuffer` for pass 3 (apply).
    pub fn read_buffer(&self, passes: &ReductionPasses) -> ReductionBuffer {
        ReductionBuffer {
            id: passes.buffer_id,
            initial_data: vec![], // reuse existing allocation
            read_write: false,
        }
    }

    // ── ChannelSum implementation ──────────────────────────────────────────

    fn build_channel_sum_passes(
        &self,
        width: u32,
        height: u32,
        total: u32,
        num_wg: u32,
    ) -> ReductionPasses {
        let wg = self.workgroup_size;
        // Buffer: num_wg × vec4<f32> (16 bytes each)
        let buf_size = num_wg as usize * 16;
        let buf_id = 0;

        let pass1_wgsl = generate_channel_sum_local(wg);
        let pass2_wgsl = generate_channel_sum_global(wg);

        let mut params1 = Vec::with_capacity(16);
        params1.extend_from_slice(&width.to_le_bytes());
        params1.extend_from_slice(&height.to_le_bytes());
        params1.extend_from_slice(&total.to_le_bytes());
        params1.extend_from_slice(&num_wg.to_le_bytes());

        let pass1 = GpuShader {
            body: pass1_wgsl,
            entry_point: "main",
            workgroup_size: [wg, 1, 1],
            params: params1.clone(),
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: buf_id,
                initial_data: vec![0u8; buf_size],
                read_write: true,
            }],
        };

        let pass2 = GpuShader {
            body: pass2_wgsl,
            entry_point: "main",
            workgroup_size: [wg, 1, 1],
            params: params1,
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: buf_id,
                initial_data: vec![],
                read_write: true,
            }],
        };

        ReductionPasses {
            pass1,
            pass2,
            buffer_id: buf_id,
            buffer_size: buf_size,
            num_workgroups: num_wg,
        }
    }

    // ── ChannelMinMax implementation ───────────────────────────────────────

    fn build_channel_min_max_passes(
        &self,
        width: u32,
        height: u32,
        total: u32,
        num_wg: u32,
    ) -> ReductionPasses {
        let wg = self.workgroup_size;
        // Buffer: num_wg × 2 × vec4<f32> (32 bytes per workgroup: min + max)
        let buf_size = num_wg as usize * 32;
        let buf_id = 1; // different from ChannelSum so both can coexist

        let pass1_wgsl = generate_channel_min_max_local(wg);
        let pass2_wgsl = generate_channel_min_max_global(wg);

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&total.to_le_bytes());
        params.extend_from_slice(&num_wg.to_le_bytes());

        let pass1 = GpuShader {
            body: pass1_wgsl,
            entry_point: "main",
            workgroup_size: [wg, 1, 1],
            params: params.clone(),
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: buf_id,
                initial_data: vec![0u8; buf_size],
                read_write: true,
            }],
        };

        let pass2 = GpuShader {
            body: pass2_wgsl,
            entry_point: "main",
            workgroup_size: [wg, 1, 1],
            params,
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: buf_id,
                initial_data: vec![],
                read_write: true,
            }],
        };

        ReductionPasses {
            pass1,
            pass2,
            buffer_id: buf_id,
            buffer_size: buf_size,
            num_workgroups: num_wg,
        }
    }
}

// ─── WGSL Generation ───────────────────────────────────────────────────────

fn generate_channel_sum_local(wg: u32) -> String {
    format!(
        r#"struct Params {{
  width: u32,
  height: u32,
  total_pixels: u32,
  num_workgroups: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> shared: array<vec4<f32>, {wg}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {{
  let idx = gid.x;
  let local_id = lid.x;

  // Load pixel or identity
  var val = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (idx < params.total_pixels) {{
    let pixel = input[idx];
    output[idx] = pixel; // passthrough
    val = vec4<f32>(pixel.x, pixel.y, pixel.z, 1.0);
  }}

  shared[local_id] = val;
  workgroupBarrier();

  // Tree reduction
  for (var stride = {half}u; stride > 0u; stride >>= 1u) {{
    if (local_id < stride) {{
      shared[local_id] += shared[local_id + stride];
    }}
    workgroupBarrier();
  }}

  if (local_id == 0u) {{
    partials[wid.x] = shared[0];
  }}
}}"#,
        wg = wg,
        half = wg / 2,
    )
}

fn generate_channel_sum_global(wg: u32) -> String {
    format!(
        r#"struct Params {{
  width: u32,
  height: u32,
  total_pixels: u32,
  num_workgroups: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> shared: array<vec4<f32>, {wg}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {{
  let local_id = lid.x;

  // Workgroup 0: reduce all partial sums
  if (wid.x == 0u) {{
    // Each thread accumulates ceil(num_workgroups / {wg}) partials
    var acc = vec4<f32>(0.0);
    let chunk = (params.num_workgroups + {wg_minus_1}u) / {wg}u;
    let start = local_id * chunk;
    let end = min(start + chunk, params.num_workgroups);
    for (var i = start; i < end; i = i + 1u) {{
      acc += partials[i];
    }}
    shared[local_id] = acc;
    workgroupBarrier();

    // Tree reduction
    for (var stride = {half}u; stride > 0u; stride >>= 1u) {{
      if (local_id < stride) {{
        shared[local_id] += shared[local_id + stride];
      }}
      workgroupBarrier();
    }}

    if (local_id == 0u) {{
      partials[0] = shared[0];
    }}
  }}

  // All workgroups: passthrough pixels
  let idx = gid.x;
  if (idx < params.total_pixels) {{
    output[idx] = input[idx];
  }}
}}"#,
        wg = wg,
        wg_minus_1 = wg - 1,
        half = wg / 2,
    )
}

fn generate_channel_min_max_local(wg: u32) -> String {
    format!(
        r#"struct Params {{
  width: u32,
  height: u32,
  total_pixels: u32,
  num_workgroups: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> shared_min: array<vec4<f32>, {wg}>;
var<workgroup> shared_max: array<vec4<f32>, {wg}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {{
  let idx = gid.x;
  let local_id = lid.x;

  var vmin = vec4<f32>(1e30, 1e30, 1e30, 0.0);
  var vmax = vec4<f32>(-1e30, -1e30, -1e30, 0.0);
  if (idx < params.total_pixels) {{
    let pixel = input[idx];
    output[idx] = pixel;
    vmin = vec4<f32>(pixel.x, pixel.y, pixel.z, 0.0);
    vmax = vmin;
  }}

  shared_min[local_id] = vmin;
  shared_max[local_id] = vmax;
  workgroupBarrier();

  for (var stride = {half}u; stride > 0u; stride >>= 1u) {{
    if (local_id < stride) {{
      shared_min[local_id] = min(shared_min[local_id], shared_min[local_id + stride]);
      shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
    }}
    workgroupBarrier();
  }}

  if (local_id == 0u) {{
    partials[wid.x * 2u] = shared_min[0];
    partials[wid.x * 2u + 1u] = shared_max[0];
  }}
}}"#,
        wg = wg,
        half = wg / 2,
    )
}

fn generate_channel_min_max_global(wg: u32) -> String {
    format!(
        r#"struct Params {{
  width: u32,
  height: u32,
  total_pixels: u32,
  num_workgroups: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> shared_min: array<vec4<f32>, {wg}>;
var<workgroup> shared_max: array<vec4<f32>, {wg}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {{
  let local_id = lid.x;

  if (wid.x == 0u) {{
    var acc_min = vec4<f32>(1e30, 1e30, 1e30, 0.0);
    var acc_max = vec4<f32>(-1e30, -1e30, -1e30, 0.0);
    let chunk = (params.num_workgroups + {wg_minus_1}u) / {wg}u;
    let start = local_id * chunk;
    let end = min(start + chunk, params.num_workgroups);
    for (var i = start; i < end; i = i + 1u) {{
      acc_min = min(acc_min, partials[i * 2u]);
      acc_max = max(acc_max, partials[i * 2u + 1u]);
    }}
    shared_min[local_id] = acc_min;
    shared_max[local_id] = acc_max;
    workgroupBarrier();

    for (var stride = {half}u; stride > 0u; stride >>= 1u) {{
      if (local_id < stride) {{
        shared_min[local_id] = min(shared_min[local_id], shared_min[local_id + stride]);
        shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
      }}
      workgroupBarrier();
    }}

    if (local_id == 0u) {{
      partials[0] = shared_min[0];
      partials[1] = shared_max[0];
    }}
  }}

  let idx = gid.x;
  if (idx < params.total_pixels) {{
    output[idx] = input[idx];
  }}
}}"#,
        wg = wg,
        wg_minus_1 = wg - 1,
        half = wg / 2,
    )
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_sum_generates_3_element_chain() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(1920, 1080);
        assert_eq!(passes.num_workgroups, (1920 * 1080 + 255) / 256);
        assert_eq!(passes.buffer_size, passes.num_workgroups as usize * 16);
        assert_eq!(passes.buffer_id, 0);
    }

    #[test]
    fn channel_sum_pass1_wgsl_valid() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(100, 100);
        let body = &passes.pass1.body;
        assert!(body.contains("var<workgroup> shared: array<vec4<f32>, 256>"));
        assert!(body.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(body.contains("partials[wid.x]"));
        assert!(body.contains("for (var stride = 128u;"));
        assert!(body.contains("workgroupBarrier()"));
        assert!(!body.contains("atomic"), "Should be zero-atomic");
    }

    #[test]
    fn channel_sum_pass2_wgsl_valid() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(100, 100);
        let body = &passes.pass2.body;
        assert!(body.contains("if (wid.x == 0u)"), "Reduction gated to workgroup 0");
        assert!(body.contains("partials[0] = shared[0]"), "Writes final result");
        assert!(body.contains("output[idx] = input[idx]"), "Passthrough pixels");
    }

    #[test]
    fn channel_sum_result_reader() {
        let r = GpuReduction::channel_sum(256);
        let wgsl = r.result_reader_wgsl(3);
        assert!(wgsl.contains("@group(0) @binding(3)"));
        assert!(wgsl.contains("fn reduction_channel_sums()"));
    }

    #[test]
    fn channel_sum_read_buffer() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(100, 100);
        let rb = r.read_buffer(&passes);
        assert_eq!(rb.id, passes.buffer_id);
        assert!(!rb.read_write, "Apply pass should be read-only");
    }

    #[test]
    fn channel_min_max_generates_passes() {
        let r = GpuReduction::channel_min_max(256);
        let passes = r.build_passes(1920, 1080);
        // 2 × vec4 per workgroup (min + max)
        assert_eq!(passes.buffer_size, passes.num_workgroups as usize * 32);
        assert_eq!(passes.buffer_id, 1);
    }

    #[test]
    fn channel_min_max_pass1_has_min_max_shared() {
        let r = GpuReduction::channel_min_max(256);
        let passes = r.build_passes(100, 100);
        let body = &passes.pass1.body;
        assert!(body.contains("shared_min"));
        assert!(body.contains("shared_max"));
        assert!(body.contains("1e30"), "Identity for min should be large");
        assert!(body.contains("-1e30"), "Identity for max should be small");
    }

    #[test]
    fn channel_min_max_result_reader() {
        let r = GpuReduction::channel_min_max(256);
        let wgsl = r.result_reader_wgsl(4);
        assert!(wgsl.contains("@group(0) @binding(4)"));
        assert!(wgsl.contains("fn reduction_min()"));
        assert!(wgsl.contains("fn reduction_max()"));
    }

    #[test]
    fn reduction_buffers_have_correct_access() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(100, 100);
        // Pass 1 and 2: read_write
        assert!(passes.pass1.reduction_buffers[0].read_write);
        assert!(passes.pass2.reduction_buffers[0].read_write);
        // Read buffer for pass 3: read-only
        let rb = r.read_buffer(&passes);
        assert!(!rb.read_write);
    }

    #[test]
    fn buffer_ids_differ_between_kinds() {
        let sum = GpuReduction::channel_sum(256);
        let mm = GpuReduction::channel_min_max(256);
        let sp = sum.build_passes(100, 100);
        let mp = mm.build_passes(100, 100);
        assert_ne!(sp.buffer_id, mp.buffer_id);
    }

    #[test]
    fn large_image_workgroup_count() {
        // 8K image: 7680 × 4320 = 33.2M pixels
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(7680, 4320);
        let num_wg = passes.num_workgroups;
        // Each pass 2 thread handles ceil(num_wg/256) partials
        let per_thread = (num_wg + 255) / 256;
        // Must fit: 256 threads × per_thread ≈ num_wg
        assert!(per_thread * 256 >= num_wg);
        assert!(per_thread < 1000, "Per-thread load reasonable: {per_thread}");
    }

    #[test]
    fn params_layout_correct() {
        let r = GpuReduction::channel_sum(256);
        let passes = r.build_passes(1920, 1080);
        let params = &passes.pass1.params;
        assert_eq!(params.len(), 16, "4 × u32 = 16 bytes");
        let width = u32::from_le_bytes(params[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(params[4..8].try_into().unwrap());
        let total = u32::from_le_bytes(params[8..12].try_into().unwrap());
        let num_wg = u32::from_le_bytes(params[12..16].try_into().unwrap());
        assert_eq!(width, 1920);
        assert_eq!(height, 1080);
        assert_eq!(total, 1920 * 1080);
        assert_eq!(num_wg, (1920 * 1080 + 255) / 256);
    }
}
