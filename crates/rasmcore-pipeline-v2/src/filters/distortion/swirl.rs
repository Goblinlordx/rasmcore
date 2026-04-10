//! Swirl distortion filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_f32};

// Swirl
// ═══════════════════════════════════════════════════════════════════════════

/// Swirl — rotational distortion decreasing with distance from center.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "swirl", category = "distortion")]
pub struct Swirl {
    #[param(min = -10.0, max = 10.0, step = 0.1, default = 2.0, hint = "rc.angle_deg")]
    pub angle: f32,
    #[param(
        min = 1.0,
        max = 2000.0,
        step = 1.0,
        default = 300.0,
        hint = "rc.pixels"
    )]
    pub radius: f32,
}

const SWIRL_WGSL: &str = r#"
struct Params { width: u32, height: u32, angle: f32, radius: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let dx = f32(x) - cx; let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);
  var sx: f32; var sy: f32;
  if (dist < params.radius && params.radius > 0.0) {
    let t = 1.0 - dist / params.radius;
    let sa = params.angle * t * t;
    let ct = cos(sa); let st = sin(sa);
    sx = dx * ct - dy * st + cx; sy = dx * st + dy * ct + cy;
  } else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Swirl {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let (sx, sy) = if dist < self.radius && self.radius > 0.0 {
                    let t = 1.0 - dist / self.radius;
                    let sa = self.angle * t * t;
                    let (st, ct) = sa.sin_cos();
                    (dx * ct - dy * st + cx, dx * st + dy * ct + cy)
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
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SWIRL_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.angle);
        gpu_params_push_f32(&mut params, self.radius);
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
