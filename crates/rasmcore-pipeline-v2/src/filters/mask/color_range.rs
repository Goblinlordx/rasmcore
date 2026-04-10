//! Color range keying mask filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::smoothstep_f32;

/// Color range keying — pixels within range of target color get alpha=1, others alpha=0.
/// Smooth falloff between threshold and threshold+softness.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "color_range", category = "mask")]
pub struct ColorRange {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub threshold: f32,
    #[param(min = 0.0, max = 0.5, step = 0.01, default = 0.1)]
    pub softness: f32,
}

const COLOR_RANGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, target_r: f32, target_g: f32, target_b: f32, threshold: f32, softness: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let dr = px.r - params.target_r; let dg = px.g - params.target_g; let db = px.b - params.target_b;
  let dist = sqrt(dr*dr + dg*dg + db*db);
  var alpha: f32;
  if (params.softness > 0.0) {
    alpha = 1.0 - smoothstep(params.threshold, params.threshold + params.softness, dist);
  } else {
    alpha = select(0.0, 1.0, dist <= params.threshold);
  }
  output[idx] = vec4<f32>(px.rgb, alpha);
}
"#;

impl Filter for ColorRange {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let dr = input[o] - self.target_r;
            let dg = input[o + 1] - self.target_g;
            let db = input[o + 2] - self.target_b;
            let dist = (dr * dr + dg * dg + db * db).sqrt();
            let alpha = if self.softness > 0.0 {
                1.0 - smoothstep_f32(self.threshold, self.threshold + self.softness, dist)
            } else {
                if dist <= self.threshold { 1.0 } else { 0.0 }
            };
            out[o + 3] = alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.target_r);
        gpu_push_f32(&mut p, self.target_g);
        gpu_push_f32(&mut p, self.target_b);
        gpu_push_f32(&mut p, self.threshold);
        gpu_push_f32(&mut p, self.softness);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            COLOR_RANGE_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
