//! Mask apply filter — multiply RGB by alpha.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_u32};

/// Apply mask — multiply RGB channels by alpha (premultiply).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mask_apply", category = "mask")]
pub struct MaskApply;

const MASK_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb * px.w, px.w);
}
"#;

impl Filter for MaskApply {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = out[o + 3];
            out[o] *= a;
            out[o + 1] *= a;
            out[o + 2] *= a;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(MASK_APPLY_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
