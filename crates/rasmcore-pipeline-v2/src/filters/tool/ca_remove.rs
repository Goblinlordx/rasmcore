//! CaRemove tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, sample_bilinear};

// CA Remove — chromatic aberration correction
// ═══════════════════════════════════════════════════════════════════════════

/// Chromatic aberration removal — shift R and B channels radially.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "ca_remove", category = "tool")]
pub struct CaRemove {
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0)] pub red_shift: f32,
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0)] pub blue_shift: f32,
}

const CA_REMOVE_WGSL: &str = r#"
struct Params { width: u32, height: u32, red_shift: f32, blue_shift: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let dx = x - cx; let dy = y - cy;
  let dist = sqrt(dx*dx + dy*dy);
  let max_dist = sqrt(cx*cx + cy*cy);
  let t = dist / max_dist;
  // Red channel: sample at shifted position
  let r_offset = t * params.red_shift;
  let r_sx = x + dx / max(dist, 0.001) * r_offset;
  let r_sy = y + dy / max(dist, 0.001) * r_offset;
  let r_idx = clamp(u32(round(r_sx)), 0u, params.width - 1u) + clamp(u32(round(r_sy)), 0u, params.height - 1u) * params.width;
  // Blue channel: sample at shifted position
  let b_offset = t * params.blue_shift;
  let b_sx = x + dx / max(dist, 0.001) * b_offset;
  let b_sy = y + dy / max(dist, 0.001) * b_offset;
  let b_idx = clamp(u32(round(b_sx)), 0u, params.width - 1u) + clamp(u32(round(b_sy)), 0u, params.height - 1u) * params.width;
  let px = input[idx];
  output[idx] = vec4<f32>(input[r_idx].r, px.g, input[b_idx].b, px.w);
}
"#;

impl Filter for CaRemove {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let max_dist = (cx*cx + cy*cy).sqrt();
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                let t = dist / max_dist;
                let i = ((y * width + x) * 4) as usize;
                // Red channel shift
                let r_off = t * self.red_shift;
                let r_src = sample_bilinear(input, width, height,
                    x as f32 + dx / dist.max(0.001) * r_off,
                    y as f32 + dy / dist.max(0.001) * r_off);
                out[i] = r_src[0];
                // Blue channel shift
                let b_off = t * self.blue_shift;
                let b_src = sample_bilinear(input, width, height,
                    x as f32 + dx / dist.max(0.001) * b_off,
                    y as f32 + dy / dist.max(0.001) * b_off);
                out[i+2] = b_src[2];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.red_shift);
        gpu_push_f32(&mut p, self.blue_shift);
        Some(vec![GpuShader::new(CA_REMOVE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
