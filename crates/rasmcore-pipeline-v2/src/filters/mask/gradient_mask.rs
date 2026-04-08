//! Gradient mask filter — linear gradient in alpha channel.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

/// Generate a linear gradient mask in the alpha channel.
/// Replaces from_path (which requires external path data).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gradient_mask", category = "mask")]
pub struct GradientMask {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub start: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub end: f32,
    #[param(min = 0, max = 1, default = 0)]
    pub vertical: bool,
}

const GRADIENT_MASK_WGSL: &str = r#"
struct Params { width: u32, height: u32, start_val: f32, end_val: f32, vertical: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = idx % params.width; let y = idx / params.width;
  var t: f32;
  if (params.vertical > 0.5) { t = f32(y) / f32(params.height - 1u); }
  else { t = f32(x) / f32(params.width - 1u); }
  let alpha = mix(params.start_val, params.end_val, t);
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, alpha);
}
"#;

impl Filter for GradientMask {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let x = (i % width as usize) as f32;
            let y = (i / width as usize) as f32;
            let t = if self.vertical {
                y / (height - 1).max(1) as f32
            } else {
                x / (width - 1).max(1) as f32
            };
            let alpha = self.start + t * (self.end - self.start);
            out[i * 4 + 3] = alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.start); gpu_push_f32(&mut p, self.end);
        gpu_push_f32(&mut p, if self.vertical { 1.0 } else { 0.0 });
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(GRADIENT_MASK_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
