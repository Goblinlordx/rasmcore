//! GradientLinear generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Gradient Linear
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a linear gradient between two colors.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gradient_linear", category = "generator")]
pub struct GradientLinear {
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub angle: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub start_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub start_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub start_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub end_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub end_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub end_b: f32,
}

const GRADIENT_LINEAR_WGSL: &str = r#"
struct Params { width: u32, height: u32, angle: f32, sr: f32, sg: f32, sb: f32, er: f32, eg: f32, eb: f32, _pad: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / f32(params.width);
  let y = f32(idx / params.width) / f32(params.height);
  let ca = cos(params.angle); let sa = sin(params.angle);
  let t = clamp(x * ca + y * sa, 0.0, 1.0);
  let start = vec3<f32>(params.sr, params.sg, params.sb);
  let end = vec3<f32>(params.er, params.eg, params.eb);
  output[idx] = vec4<f32>(mix(start, end, vec3<f32>(t)), 1.0);
}
"#;

impl Filter for GradientLinear {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let a = self.angle.to_radians();
        let (sa, ca) = a.sin_cos();
        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let t = (nx * ca + ny * sa).max(0.0).min(1.0);
                let i = ((y * width + x) * 4) as usize;
                out[i] = self.start_r + t * (self.end_r - self.start_r);
                out[i + 1] = self.start_g + t * (self.end_g - self.start_g);
                out[i + 2] = self.start_b + t * (self.end_b - self.start_b);
                out[i + 3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.angle.to_radians());
        gpu_push_f32(&mut p, self.start_r);
        gpu_push_f32(&mut p, self.start_g);
        gpu_push_f32(&mut p, self.start_b);
        gpu_push_f32(&mut p, self.end_r);
        gpu_push_f32(&mut p, self.end_g);
        gpu_push_f32(&mut p, self.end_b);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            GRADIENT_LINEAR_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
