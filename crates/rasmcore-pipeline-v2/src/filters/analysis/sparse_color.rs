//! SparseColor filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Sparse Color — Shepard's weighted interpolation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse color interpolation — blend image toward sparse color points
/// using inverse-distance weighting (Shepard's method).
/// Uses 4 fixed color control points for simplicity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sparse_color", category = "effect")]
pub struct SparseColor {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub x1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub y1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub r1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub g1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub b1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub x2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub y2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub r2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub g2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub b2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub strength: f32,
}

const SPARSE_COLOR_WGSL: &str = r#"
struct Params { width: u32, height: u32, x1: f32, y1: f32, r1: f32, g1: f32, b1: f32, x2: f32, y2: f32, r2: f32, g2: f32, b2: f32, strength: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / f32(params.width);
  let y = f32(idx / params.width) / f32(params.height);
  let d1 = max(length(vec2<f32>(x - params.x1, y - params.y1)), 0.001);
  let d2 = max(length(vec2<f32>(x - params.x2, y - params.y2)), 0.001);
  let w1 = 1.0 / (d1 * d1); let w2 = 1.0 / (d2 * d2);
  let total = w1 + w2;
  let interp = (vec3<f32>(params.r1, params.g1, params.b1) * w1 + vec3<f32>(params.r2, params.g2, params.b2) * w2) / total;
  let px = input[idx];
  output[idx] = vec4<f32>(mix(px.rgb, interp, vec3<f32>(params.strength)), px.w);
}
"#;

impl Filter for SparseColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let d1 = ((nx - self.x1).powi(2) + (ny - self.y1).powi(2))
                    .sqrt()
                    .max(0.001);
                let d2 = ((nx - self.x2).powi(2) + (ny - self.y2).powi(2))
                    .sqrt()
                    .max(0.001);
                let w1 = 1.0 / (d1 * d1);
                let w2 = 1.0 / (d2 * d2);
                let total = w1 + w2;
                let i = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    let colors = [[self.r1, self.g1, self.b1], [self.r2, self.g2, self.b2]];
                    let interp = (colors[0][c] * w1 + colors[1][c] * w2) / total;
                    out[i + c] = out[i + c] + self.strength * (interp - out[i + c]);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.x1);
        gpu_push_f32(&mut p, self.y1);
        gpu_push_f32(&mut p, self.r1);
        gpu_push_f32(&mut p, self.g1);
        gpu_push_f32(&mut p, self.b1);
        gpu_push_f32(&mut p, self.x2);
        gpu_push_f32(&mut p, self.y2);
        gpu_push_f32(&mut p, self.r2);
        gpu_push_f32(&mut p, self.g2);
        gpu_push_f32(&mut p, self.b2);
        gpu_push_f32(&mut p, self.strength);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            SPARSE_COLOR_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
