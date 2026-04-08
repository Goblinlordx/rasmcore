//! Luminance range keying mask filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::smoothstep_f32;

/// Luminance range keying — pixels within luma range get alpha=1.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "luminance_range", category = "mask")]
pub struct LuminanceRange {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub low: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.8)]
    pub high: f32,
    #[param(min = 0.0, max = 0.5, step = 0.01, default = 0.05)]
    pub softness: f32,
}

const LUMINANCE_RANGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, low: f32, high: f32, softness: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  let lo = smoothstep(params.low - params.softness, params.low + params.softness, luma);
  let hi = 1.0 - smoothstep(params.high - params.softness, params.high + params.softness, luma);
  output[idx] = vec4<f32>(px.rgb, lo * hi);
}
"#;

impl Filter for LuminanceRange {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let luma = input[o] * 0.2126 + input[o + 1] * 0.7152 + input[o + 2] * 0.0722;
            let lo = smoothstep_f32(self.low - self.softness, self.low + self.softness, luma);
            let hi = 1.0 - smoothstep_f32(self.high - self.softness, self.high + self.softness, luma);
            out[o + 3] = lo * hi;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.low); gpu_push_f32(&mut p, self.high);
        gpu_push_f32(&mut p, self.softness); gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(LUMINANCE_RANGE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
