//! Ripple distortion filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32, gpu_params_push_u32};
use std::f32::consts::PI;

// Ripple
// ═══════════════════════════════════════════════════════════════════════════

/// Ripple — concentric wave distortion from center point.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "ripple", category = "distortion")]
pub struct Ripple {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 10.0)]
    pub amplitude: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub wavelength: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
}

const RIPPLE_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, amplitude: f32, wavelength: f32, center_x: f32, center_y: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = params.center_x * f32(params.width);
  let cy = params.center_y * f32(params.height);
  let dx = f32(x) - cx; let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);
  let disp = params.amplitude * sin(2.0 * PI * dist / params.wavelength);
  var sx: f32; var sy: f32;
  if (dist > 0.0) { sx = f32(x) + disp * dx / dist; sy = f32(y) + disp * dy / dist; }
  else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Ripple {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let disp = self.amplitude * (2.0 * PI * dist / self.wavelength).sin();
                let (sx, sy) = if dist > 0.0 {
                    (x as f32 + disp * dx / dist, y as f32 + disp * dy / dist)
                } else {
                    (x as f32, y as f32)
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{RIPPLE_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.amplitude);
        gpu_params_push_f32(&mut params, self.wavelength);
        gpu_params_push_f32(&mut params, self.center_x);
        gpu_params_push_f32(&mut params, self.center_y);
        gpu_params_push_u32(&mut params, 0); // pad
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(
            shader,
            "main",
            [16, 16, 1],
            params,
        )])
    }

    fn tile_overlap(&self) -> u32 {
        0
    }
}
