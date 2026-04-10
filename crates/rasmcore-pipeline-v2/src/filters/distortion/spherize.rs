//! Spherize distortion filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32, gpu_params_push_u32};

// Spherize
// ═══════════════════════════════════════════════════════════════════════════

/// Spherize distortion — spherical lens effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "spherize", category = "distortion")]
pub struct Spherize {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
}

const SPHERIZE_WGSL: &str = r#"
struct Params { width: u32, height: u32, amount: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let nx = (f32(x) - cx) / cx; let ny = (f32(y) - cy) / cy;
  let r = sqrt(nx * nx + ny * ny);
  var sx: f32; var sy: f32;
  if (r < 1.0 && r > 0.0) {
    let theta = asin(r) / r;
    let factor = mix(1.0, theta, params.amount);
    sx = nx * factor * cx + cx; sy = ny * factor * cy + cy;
  } else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Spherize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let nx = (x as f32 - cx) / cx;
                let ny = (y as f32 - cy) / cy;
                let r = (nx * nx + ny * ny).sqrt();
                let (sx, sy) = if r < 1.0 && r > 0.0 {
                    let theta = r.asin() / r;
                    let factor = 1.0 + self.amount * (theta - 1.0);
                    (nx * factor * cx + cx, ny * factor * cy + cy)
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
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SPHERIZE_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.amount);
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
