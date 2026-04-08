//! Masked blend filter — blend toward a color using alpha.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

/// Blend toward a target color using the alpha channel as mix factor.
/// Where alpha=1, output=original. Where alpha=0, output=blend_color.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "masked_blend", category = "mask")]
pub struct MaskedBlend {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_b: f32,
}

const MASKED_BLEND_WGSL: &str = r#"
struct Params { width: u32, height: u32, blend_r: f32, blend_g: f32, blend_b: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let blend = vec3<f32>(params.blend_r, params.blend_g, params.blend_b);
  let rgb = mix(blend, px.rgb, vec3<f32>(px.w));
  output[idx] = vec4<f32>(rgb, 1.0);
}
"#;

impl Filter for MaskedBlend {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = input[o + 3];
            out[o] = self.blend_r + a * (input[o] - self.blend_r);
            out[o + 1] = self.blend_g + a * (input[o + 1] - self.blend_g);
            out[o + 2] = self.blend_b + a * (input[o + 2] - self.blend_b);
            out[o + 3] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.blend_r); gpu_push_f32(&mut p, self.blend_g);
        gpu_push_f32(&mut p, self.blend_b); gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(MASKED_BLEND_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
