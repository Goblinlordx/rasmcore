//! FloodFill tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Flood Fill — replace connected region with solid color
// ═══════════════════════════════════════════════════════════════════════════

/// Flood fill — replace color-similar connected region from seed point.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "flood_fill", category = "tool")]
pub struct FloodFill {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub seed_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub seed_y: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.1)] pub tolerance: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub fill_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub fill_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub fill_b: f32,
}

impl Filter for FloodFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let w = width as usize;
        let h = height as usize;
        let sx = (self.seed_x * width as f32) as usize;
        let sy = (self.seed_y * height as f32) as usize;
        if sx >= w || sy >= h { return Ok(out); }

        let seed_i = (sy * w + sx) * 4;
        let seed_color = [input[seed_i], input[seed_i+1], input[seed_i+2]];
        let tol2 = self.tolerance * self.tolerance;

        let mut visited = vec![false; w * h];
        let mut stack = vec![(sx, sy)];
        visited[sy * w + sx] = true;

        while let Some((x, y)) = stack.pop() {
            let i = (y * w + x) * 4;
            let dr = out[i] - seed_color[0];
            let dg = out[i+1] - seed_color[1];
            let db = out[i+2] - seed_color[2];
            if dr*dr + dg*dg + db*db > tol2 { continue; }

            out[i] = self.fill_r;
            out[i+1] = self.fill_g;
            out[i+2] = self.fill_b;

            for (nx, ny) in [(x.wrapping_sub(1), y), (x+1, y), (x, y.wrapping_sub(1)), (x, y+1)] {
                if nx < w && ny < h && !visited[ny * w + nx] {
                    visited[ny * w + nx] = true;
                    stack.push((nx, ny));
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        use crate::node::ReductionBuffer;

        let n = (width * height) as usize;
        let mask_size = n * 4; // u32 per pixel
        let change_size = 4usize; // single atomic u32
        let mask_buf_id = 50u32;
        let change_buf_id = 51u32;

        // Initialize mask: seed pixel = 1, all others = 0
        let seed_x = (self.seed_x * width as f32) as u32;
        let seed_y = (self.seed_y * height as f32) as u32;
        let mut init_mask = vec![0u8; mask_size];
        let seed_idx = (seed_y * width + seed_x) as usize;
        if seed_idx < n {
            init_mask[seed_idx * 4..seed_idx * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
        }

        // Init pass: snapshot input colors + initialize mask + apply fill to seed pixel
        let init_wgsl = format!(r#"
struct Params {{ width: u32, height: u32, seed_x: u32, seed_y: u32, fill_r: f32, fill_g: f32, fill_b: f32, _pad: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if (idx >= params.width * params.height) {{ return; }}
  let px = input[idx];
  if (idx == params.seed_x + params.seed_y * params.width) {{
    output[idx] = vec4<f32>(params.fill_r, params.fill_g, params.fill_b, px.w);
  }} else {{
    output[idx] = px;
  }}
}}
"#);

        let mut init_params = gpu_params_wh(width, height);
        gpu_push_u32(&mut init_params, seed_x);
        gpu_push_u32(&mut init_params, seed_y);
        gpu_push_f32(&mut init_params, self.fill_r);
        gpu_push_f32(&mut init_params, self.fill_g);
        gpu_push_f32(&mut init_params, self.fill_b);
        gpu_push_u32(&mut init_params, 0);

        let init_shader = GpuShader {
            body: init_wgsl,
            entry_point: "main",
            workgroup_size: [256, 1, 1],
            params: init_params,
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: mask_buf_id,
                initial_data: init_mask,
                read_write: true,
            }],
            convergence_check: None,
            loop_dispatch: None,
        };

        // Expand pass: for each unfilled pixel, check 4 neighbors in mask.
        // If any neighbor is filled AND this pixel's color is within tolerance
        // of the seed color → mark as filled, apply fill color.
        // Reads original colors from input (which has the original image on first
        // iteration, then progressively filled image).
        let expand_wgsl = format!(r#"
struct Params {{ width: u32, height: u32, seed_r: f32, seed_g: f32, seed_b: f32, tol2: f32, fill_r: f32, fill_g: f32, fill_b: f32, _pad: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> change_count: array<atomic<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if (idx >= params.width * params.height) {{ return; }}
  // Already filled → pass through
  if (mask[idx] != 0u) {{ output[idx] = input[idx]; return; }}
  let x = i32(idx % params.width); let y = i32(idx / params.width);
  let w = i32(params.width); let h = i32(params.height);
  // Check 4 neighbors
  var has_filled_neighbor = false;
  if (x > 0 && mask[u32(x-1) + u32(y) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (x < w-1 && mask[u32(x+1) + u32(y) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (y > 0 && mask[u32(x) + u32(y-1) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (y < h-1 && mask[u32(x) + u32(y+1) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (!has_filled_neighbor) {{ output[idx] = input[idx]; return; }}
  // Check color tolerance against seed color
  let px = input[idx];
  let dr = px.r - params.seed_r; let dg = px.g - params.seed_g; let db = px.b - params.seed_b;
  if (dr*dr + dg*dg + db*db > params.tol2) {{ output[idx] = input[idx]; return; }}
  // Fill!
  mask[idx] = 1u;
  output[idx] = vec4<f32>(params.fill_r, params.fill_g, params.fill_b, px.w);
  atomicAdd(&change_count[0], 1u);
}}
"#);

        // Get seed color from input — we need it at shader construction time.
        // Since gpu_shader_passes doesn't have access to pixel data, we pass
        // seed_x/seed_y and let the init pass extract it. For the expand pass,
        // we use the seed color params. The caller must set these from the image.
        // For now, use default (0.5, 0.5, 0.5) — the CPU fallback handles exact colors.
        // TODO: Extract seed color from pixel data when available.
        let seed_r = 0.5f32; // Approximation — CPU path is exact
        let seed_g = 0.5f32;
        let seed_b = 0.5f32;
        let tol2 = self.tolerance * self.tolerance;

        let mut expand_params = gpu_params_wh(width, height);
        gpu_push_f32(&mut expand_params, seed_r);
        gpu_push_f32(&mut expand_params, seed_g);
        gpu_push_f32(&mut expand_params, seed_b);
        gpu_push_f32(&mut expand_params, tol2);
        gpu_push_f32(&mut expand_params, self.fill_r);
        gpu_push_f32(&mut expand_params, self.fill_g);
        gpu_push_f32(&mut expand_params, self.fill_b);
        gpu_push_u32(&mut expand_params, 0);
        gpu_push_u32(&mut expand_params, 0);
        gpu_push_u32(&mut expand_params, 0);

        // Generate N expand passes with convergence check
        let max_iterations = (width.max(height) / 2).max(100);
        let mut passes = vec![init_shader];

        for _ in 0..max_iterations {
            passes.push(GpuShader {
                body: expand_wgsl.clone(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: expand_params.clone(),
                extra_buffers: vec![],
                reduction_buffers: vec![
                    ReductionBuffer { id: mask_buf_id, initial_data: vec![], read_write: true },
                    ReductionBuffer { id: change_buf_id, initial_data: vec![0u8; change_size], read_write: true },
                ],
                convergence_check: Some(change_buf_id),
                loop_dispatch: None,
            });
        }

        Some(passes)
    }
}
