//! Smudge tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, sample_bilinear};
use super::SAMPLE_BILINEAR_WGSL;

// Smudge — directional displacement within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Smudge — push pixels in a direction within a circular brush.
/// Similar to liquify but with softer gaussian falloff.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "smudge", category = "tool")]
pub struct Smudge {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")]
    pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub strength: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.5)]
    pub direction_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub direction_y: f32,
}

const SMUDGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, radius: f32, strength: f32, dir_x: f32, dir_y: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let dx = x - params.cx; let dy = y - params.cy;
  let dist = sqrt(dx*dx + dy*dy);
  if (dist >= params.radius) { output[idx] = input[idx]; return; }
  let t = dist / params.radius;
  let w = exp(-2.0 * t * t) * params.strength;
  let sx = x - params.dir_x * w * params.radius;
  let sy = y - params.dir_y * w * params.radius;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Smudge {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= self.radius {
                    continue;
                }
                let t = dist / self.radius;
                let w = (-2.0 * t * t).exp() * self.strength;
                let sx = x as f32 - self.direction_x * w * self.radius;
                let sy = y as f32 - self.direction_y * w * self.radius;
                let src = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&src);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SMUDGE_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.strength);
        gpu_push_f32(&mut p, self.direction_x);
        gpu_push_f32(&mut p, self.direction_y);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
