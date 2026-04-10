use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_u32};

/// Remove alpha — set all pixels to fully opaque (alpha = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "remove_alpha", category = "alpha")]
pub struct RemoveAlpha;

const REMOVE_ALPHA_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, 1.0);
}
"#;

impl Filter for RemoveAlpha {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for i in (3..out.len()).step_by(4) {
            out[i] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            REMOVE_ALPHA_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_alpha_makes_opaque() {
        let input = vec![0.5, 0.5, 0.5, 0.3];
        let f = RemoveAlpha;
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[3] - 1.0).abs() < 0.001);
    }
}
