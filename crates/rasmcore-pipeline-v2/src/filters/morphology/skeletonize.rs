//! Skeletonize morphology filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::gpu_params_wh;
use super::gpu_params_push_u32;
use crate::node::ReductionBuffer;

// Skeletonize
// ═══════════════════════════════════════════════════════════════════════════

/// Zhang-Suen thinning shader — f32 RGBA variant.
///
/// ## Convergence detection — future optimization
///
/// Currently uses a single `atomic<u32>` change counter. All threads that
/// delete a pixel atomicAdd to the same address, which serializes under
/// contention. At 4K (~32K workgroups), this adds ~5-10% overhead per pass.
///
/// Optimization path: **per-workgroup flags** with host-side reduction.
/// 1. Allocate `change_flags[num_workgroups]` buffer (one u32 per workgroup)
/// 2. Each workgroup uses `var<workgroup> wg_changed: u32` (shared memory)
/// 3. Any thread that deletes sets `wg_changed = 1u` (no atomic — any write wins)
/// 4. After `workgroupBarrier()`, thread (0,0) writes `change_flags[wg_id] = wg_changed`
/// 5. Host checks if any element is non-zero (scan 128KB for 4K image — trivial)
///
/// This eliminates all atomic contention. Estimated improvement: ~5-10% per pass,
/// compounding over 50-100 iterations → ~5-10ms saved on a ~100ms 4K skeletonize.
/// Operates on luminance: pixel "on" if luma > threshold.
/// Uses atomic change counter in reduction buffer for convergence detection.
/// `sub_iteration` param selects step 1 (0) or step 2 (1).
const ZHANG_SUEN_WGSL: &str = r#"
struct Params { width: u32, height: u32, threshold: f32, sub_iteration: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> change_count: array<atomic<u32>>;

fn is_on(x: i32, y: i32) -> u32 {
  if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) { return 0u; }
  let px = input[u32(x) + u32(y) * params.width];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  return select(0u, 1u, luma > params.threshold);
}

fn count_neighbors(x: i32, y: i32) -> u32 {
  return is_on(x, y-1) + is_on(x+1, y-1) + is_on(x+1, y) + is_on(x+1, y+1)
       + is_on(x, y+1) + is_on(x-1, y+1) + is_on(x-1, y) + is_on(x-1, y-1);
}

fn count_transitions(x: i32, y: i32) -> u32 {
  let p2 = is_on(x, y-1); let p3 = is_on(x+1, y-1); let p4 = is_on(x+1, y);
  let p5 = is_on(x+1, y+1); let p6 = is_on(x, y+1); let p7 = is_on(x-1, y+1);
  let p8 = is_on(x-1, y); let p9 = is_on(x-1, y-1);
  var t = 0u;
  t += select(0u, 1u, p2 == 0u && p3 == 1u);
  t += select(0u, 1u, p3 == 0u && p4 == 1u);
  t += select(0u, 1u, p4 == 0u && p5 == 1u);
  t += select(0u, 1u, p5 == 0u && p6 == 1u);
  t += select(0u, 1u, p6 == 0u && p7 == 1u);
  t += select(0u, 1u, p7 == 0u && p8 == 1u);
  t += select(0u, 1u, p8 == 0u && p9 == 1u);
  t += select(0u, 1u, p9 == 0u && p2 == 1u);
  return t;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= i32(params.width) || y >= i32(params.height)) { return; }
  let idx = u32(x) + u32(y) * params.width;
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;

  // Background → pass through
  if (luma <= params.threshold) { output[idx] = vec4<f32>(0.0, 0.0, 0.0, px.w); return; }

  let p2 = is_on(x, y-1); let p4 = is_on(x+1, y); let p6 = is_on(x, y+1); let p8 = is_on(x-1, y);
  let B = count_neighbors(x, y);
  let A = count_transitions(x, y);

  var should_delete = false;
  if (B >= 2u && B <= 6u && A == 1u) {
    if (params.sub_iteration == 0u) {
      if ((p2 * p4 * p6 == 0u) && (p4 * p6 * p8 == 0u)) { should_delete = true; }
    } else {
      if ((p2 * p4 * p8 == 0u) && (p2 * p6 * p8 == 0u)) { should_delete = true; }
    }
  }

  if (should_delete) {
    output[idx] = vec4<f32>(0.0, 0.0, 0.0, px.w);
    atomicAdd(&change_count[0], 1u);
  } else {
    output[idx] = vec4<f32>(1.0, 1.0, 1.0, px.w);
  }
}
"#;

/// Binarize shader — threshold luminance to white/black f32 RGBA.
const BINARIZE_WGSL: &str = r#"
struct Params { width: u32, height: u32, threshold: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  let v = select(0.0, 1.0, luma > params.threshold);
  output[idx] = vec4<f32>(v, v, v, px.w);
}
"#;

/// Skeletonize — iterative morphological thinning to 1-pixel skeleton.
/// Operates on luminance: pixel is "on" if luma > threshold.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "skeletonize", category = "morphology")]
pub struct Skeletonize {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
    #[param(min = 1, max = 100, step = 1, default = 50)]
    pub iterations: u32,
}

impl Filter for Skeletonize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Convert to binary image (luminance thresholding)
        let mut binary = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let luma = input[i] * 0.2126 + input[i + 1] * 0.7152 + input[i + 2] * 0.0722;
                binary[y * w + x] = luma > self.threshold;
            }
        }

        // Zhang-Suen thinning algorithm
        for _ in 0..self.iterations {
            let mut changed = false;

            // Step 1
            let mut to_remove = Vec::new();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !binary[y * w + x] { continue; }
                    let (transitions, neighbors) = zhang_suen_check(&binary, w, x, y);
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 {
                        let p2 = binary[(y - 1) * w + x] as u8;
                        let p4 = binary[y * w + x + 1] as u8;
                        let p6 = binary[(y + 1) * w + x] as u8;
                        let p8 = binary[y * w + x - 1] as u8;
                        if p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0 {
                            to_remove.push(y * w + x);
                        }
                    }
                }
            }
            for &idx in &to_remove { binary[idx] = false; changed = true; }

            // Step 2
            to_remove.clear();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !binary[y * w + x] { continue; }
                    let (transitions, neighbors) = zhang_suen_check(&binary, w, x, y);
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 {
                        let p2 = binary[(y - 1) * w + x] as u8;
                        let p4 = binary[y * w + x + 1] as u8;
                        let p6 = binary[(y + 1) * w + x] as u8;
                        let p8 = binary[y * w + x - 1] as u8;
                        if p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0 {
                            to_remove.push(y * w + x);
                        }
                    }
                }
            }
            for &idx in &to_remove { binary[idx] = false; changed = true; }

            if !changed { break; }
        }

        // Convert back to f32 RGBA
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = if binary[y * w + x] { 1.0f32 } else { 0.0 };
                out[i] = v;
                out[i + 1] = v;
                out[i + 2] = v;
                out[i + 3] = input[i + 3];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let change_buf_size = 4usize; // single atomic<u32>
        let change_buf_id = 99u32; // unique ID for convergence counter

        let mut passes = Vec::new();

        // Pass 0: binarize (threshold luminance to white/black)
        {
            let mut params = gpu_params_wh(width, height);
            params.extend_from_slice(&self.threshold.to_le_bytes());
            gpu_params_push_u32(&mut params, 0); // pad
            passes.push(GpuShader::new(BINARIZE_WGSL.to_string(), "main", [16, 16, 1], params));
        }

        // Passes 1..N: Zhang-Suen sub-iterations
        // Each iteration = step1 + step2. Both write to the same atomic counter.
        // Convergence check only on step2 — checks combined changes from both steps.
        for _ in 0..self.iterations {
            for sub in 0..2u32 {
                let mut params = gpu_params_wh(width, height);
                params.extend_from_slice(&self.threshold.to_le_bytes());
                gpu_params_push_u32(&mut params, sub);

                let shader = GpuShader {
                    body: ZHANG_SUEN_WGSL.to_string(),
                    entry_point: "main",
                    workgroup_size: [16, 16, 1],
                    params,
                    extra_buffers: vec![],
                    reduction_buffers: vec![ReductionBuffer {
                        id: change_buf_id,
                        initial_data: vec![0u8; change_buf_size],
                        read_write: true,
                    }],
                    // Only check after step2 — both steps accumulate into same counter
                    convergence_check: if sub == 1 { Some(change_buf_id) } else { None },
                    loop_dispatch: None,
                };
                passes.push(shader);
            }
        }

        Some(passes)
    }

    fn tile_overlap(&self) -> u32 { 1 }
}

/// Zhang-Suen helper: count transitions and neighbors in 8-connected ring.
fn zhang_suen_check(binary: &[bool], w: usize, x: usize, y: usize) -> (u8, u8) {
    let p = [
        binary[(y - 1) * w + x],     // P2
        binary[(y - 1) * w + x + 1], // P3
        binary[y * w + x + 1],       // P4
        binary[(y + 1) * w + x + 1], // P5
        binary[(y + 1) * w + x],     // P6
        binary[(y + 1) * w + x - 1], // P7
        binary[y * w + x - 1],       // P8
        binary[(y - 1) * w + x - 1], // P9
    ];
    let neighbors = p.iter().filter(|&&v| v).count() as u8;
    let mut transitions = 0u8;
    for i in 0..8 {
        if !p[i] && p[(i + 1) % 8] { transitions += 1; }
    }
    (transitions, neighbors)
}
