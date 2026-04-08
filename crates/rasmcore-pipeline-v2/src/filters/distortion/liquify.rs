//! Liquify distortion filter.

use crate::node::{PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32};

// Liquify
// ═══════════════════════════════════════════════════════════════════════════

/// Liquify push — directional displacement within circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "liquify", category = "distortion")]
pub struct Liquify {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 100.0, hint = "rc.pixels")]
    pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub strength: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 1.0)]
    pub direction_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub direction_y: f32,
}

const LIQUIFY_WGSL: &str = r#"
struct Params { width: u32, height: u32, center_x: f32, center_y: f32, radius: f32, strength: f32, direction_x: f32, direction_y: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn gaussian_weight(dist: f32, radius: f32) -> f32 { let t = dist / radius; return exp(-2.0 * t * t); }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let xf = f32(x); let yf = f32(y);
  let cx = params.center_x * f32(params.width); let cy = params.center_y * f32(params.height);
  let ddx = xf - cx; let ddy = yf - cy;
  let dist = sqrt(ddx * ddx + ddy * ddy);
  if (dist >= params.radius) { output[x + y * params.width] = input[x + y * params.width]; return; }
  let w = gaussian_weight(dist, params.radius) * params.strength;
  let sx = xf - params.direction_x * w * params.radius;
  let sy = yf - params.direction_y * w * params.radius;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Liquify {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32;
                let yf = y as f32;
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= self.radius { continue; }
                let t = dist / self.radius;
                let w = (-2.0 * t * t).exp() * self.strength;
                let sx = xf - self.direction_x * w * self.radius;
                let sy = yf - self.direction_y * w * self.radius;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{LIQUIFY_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.center_x * width as f32);
        gpu_params_push_f32(&mut params, self.center_y * height as f32);
        gpu_params_push_f32(&mut params, self.radius);
        gpu_params_push_f32(&mut params, self.strength);
        gpu_params_push_f32(&mut params, self.direction_x);
        gpu_params_push_f32(&mut params, self.direction_y);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}
