//! SolidFill drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Solid Fill — fill entire image with a solid color
// ═══════════════════════════════════════════════════════════════════════════

/// Fill the image with a solid color (useful as a drawing primitive).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solid_fill", category = "draw")]
pub struct SolidFill {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
}

const SOLID_FILL_WGSL: &str = r#"
struct Params { width: u32, height: u32, cr: f32, cg: f32, cb: f32, ca: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let bg = input[idx];
  let ca = params.ca;
  output[idx] = vec4<f32>(bg.rgb * (1.0 - ca) + vec3<f32>(params.cr, params.cg, params.cb) * ca, bg.w * (1.0 - ca) + ca);
}
"#;

impl Filter for SolidFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let ca = self.color_a;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0] * (1.0 - ca) + self.color_r * ca;
            pixel[1] = pixel[1] * (1.0 - ca) + self.color_g * ca;
            pixel[2] = pixel[2] * (1.0 - ca) + self.color_b * ca;
            pixel[3] = pixel[3] * (1.0 - ca) + ca;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(SOLID_FILL_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
