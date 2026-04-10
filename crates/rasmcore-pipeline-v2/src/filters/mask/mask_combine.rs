//! Mask combine filter — combine two masks.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_u32};

/// Combine the current alpha with a generated alpha (e.g., from luminance).
/// Mode: 0=multiply, 1=add, 2=min, 3=max, 4=replace.
/// The "second mask" is derived from luminance of the RGB channels.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mask_combine", category = "mask")]
pub struct MaskCombine {
    #[param(min = 0, max = 4, step = 1, default = 0)]
    pub mode: u32,
}

const MASK_COMBINE_WGSL: &str = r#"
struct Params { width: u32, height: u32, mode: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let existing = px.w;
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  var result: f32;
  switch (params.mode) {
    case 0u: { result = existing * luma; }
    case 1u: { result = clamp(existing + luma, 0.0, 1.0); }
    case 2u: { result = min(existing, luma); }
    case 3u: { result = max(existing, luma); }
    default: { result = luma; }
  }
  output[idx] = vec4<f32>(px.rgb, result);
}
"#;

impl Filter for MaskCombine {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let existing = input[o + 3];
            let luma = input[o] * 0.2126 + input[o + 1] * 0.7152 + input[o + 2] * 0.0722;
            out[o + 3] = match self.mode {
                0 => existing * luma,
                1 => (existing + luma).min(1.0),
                2 => existing.min(luma),
                3 => existing.max(luma),
                _ => luma,
            };
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, self.mode);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            MASK_COMBINE_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
