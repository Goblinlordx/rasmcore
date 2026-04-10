use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

/// Flatten — composite the image over a solid background color using alpha.
/// Result is fully opaque (alpha = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "flatten", category = "alpha")]
pub struct Flatten {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_b: f32,
}

const FLATTEN_WGSL: &str = r#"
struct Params { width: u32, height: u32, bg_r: f32, bg_g: f32, bg_b: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let bg = vec3<f32>(params.bg_r, params.bg_g, params.bg_b);
  let rgb = mix(bg, px.rgb, vec3<f32>(px.w));
  output[idx] = vec4<f32>(rgb, 1.0);
}
"#;

impl Filter for Flatten {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = input[o + 3];
            out[o] = self.bg_r + a * (input[o] - self.bg_r);
            out[o + 1] = self.bg_g + a * (input[o + 1] - self.bg_g);
            out[o + 2] = self.bg_b + a * (input[o + 2] - self.bg_b);
            out[o + 3] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.bg_r);
        gpu_push_f32(&mut p, self.bg_g);
        gpu_push_f32(&mut p, self.bg_b);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            FLATTEN_WGSL.to_string(),
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
    fn flatten_composites_over_white() {
        let input = vec![1.0, 0.0, 0.0, 0.5]; // red at 50% alpha
        let f = Flatten {
            bg_r: 1.0,
            bg_g: 1.0,
            bg_b: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        // mix(white, red, 0.5) = (1.0, 0.5, 0.5)
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 0.5).abs() < 0.01);
        assert!((out[2] - 0.5).abs() < 0.01);
        assert!((out[3] - 1.0).abs() < 0.01); // fully opaque
    }
}
