//! PerspectiveWarp filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32, sample_bilinear};
use super::SAMPLE_BILINEAR_WGSL;

// Perspective Warp
// ═══════════════════════════════════════════════════════════════════════════

/// Perspective warp — apply a 3x3 homography matrix.
/// Parameters are the 8 independent elements (h33 = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perspective_warp", category = "transform")]
pub struct PerspectiveWarp {
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 1.0)]
    pub h11: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 0.0)]
    pub h12: f32,
    #[param(min = -1000.0, max = 1000.0, step = 1.0, default = 0.0)]
    pub h13: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 0.0)]
    pub h21: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 1.0)]
    pub h22: f32,
    #[param(min = -1000.0, max = 1000.0, step = 1.0, default = 0.0)]
    pub h23: f32,
    #[param(min = -0.01, max = 0.01, step = 0.0001, default = 0.0)]
    pub h31: f32,
    #[param(min = -0.01, max = 0.01, step = 0.0001, default = 0.0)]
    pub h32: f32,
}

const PERSPECTIVE_WARP_WGSL: &str = r#"
struct Params { width: u32, height: u32, h11: f32, h12: f32, h13: f32, h21: f32, h22: f32, h23: f32, h31: f32, h32: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let w = params.h31 * x + params.h32 * y + 1.0;
  if (abs(w) < 0.0001) { output[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
  let sx = (params.h11 * x + params.h12 * y + params.h13) / w;
  let sy = (params.h21 * x + params.h22 * y + params.h23) / w;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for PerspectiveWarp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32;
                let yf = y as f32;
                let w = self.h31 * xf + self.h32 * yf + 1.0;
                let (sx, sy) = if w.abs() > 0.0001 {
                    (
                        (self.h11 * xf + self.h12 * yf + self.h13) / w,
                        (self.h21 * xf + self.h22 * yf + self.h23) / w,
                    )
                } else {
                    (0.0, 0.0)
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{PERSPECTIVE_WARP_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.h11);
        gpu_push_f32(&mut p, self.h12);
        gpu_push_f32(&mut p, self.h13);
        gpu_push_f32(&mut p, self.h21);
        gpu_push_f32(&mut p, self.h22);
        gpu_push_f32(&mut p, self.h23);
        gpu_push_f32(&mut p, self.h31);
        gpu_push_f32(&mut p, self.h32);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
