use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

/// Set the alpha channel to a constant value.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "add_alpha", category = "alpha")]
pub struct AddAlpha {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub alpha: f32,
}

const ADD_ALPHA_WGSL: &str = r#"
struct Params { width: u32, height: u32, alpha: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, params.alpha);
}
"#;

impl Filter for AddAlpha {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for i in (3..out.len()).step_by(4) {
            out[i] = self.alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.alpha);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(ADD_ALPHA_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_alpha_sets_constant() {
        let input = vec![0.5, 0.5, 0.5, 0.0, 0.8, 0.8, 0.8, 0.3];
        let f = AddAlpha { alpha: 0.7 };
        let out = f.compute(&input, 2, 1).unwrap();
        assert!((out[3] - 0.7).abs() < 0.001);
        assert!((out[7] - 0.7).abs() < 0.001);
    }
}
