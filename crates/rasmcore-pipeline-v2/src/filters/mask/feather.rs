//! Feather mask filter — blur the alpha channel.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_u32};

/// Feather mask edges — gaussian blur on alpha channel only.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "feather", category = "mask")]
pub struct Feather {
    #[param(min = 1, max = 50, step = 1, default = 3, hint = "rc.pixels")]
    pub radius: u32,
}

const FEATHER_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var sum: f32 = 0.0; var weight: f32 = 0.0;
  let sigma = f32(r) / 2.0;
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1); let sy = clamp(y + dy, 0, h - 1);
      let d2 = f32(dx*dx + dy*dy);
      let w_g = exp(-d2 * inv2s2);
      sum += input[u32(sx) + u32(sy) * params.width].w * w_g;
      weight += w_g;
    }
  }
  let px = input[u32(x) + u32(y) * params.width];
  output[u32(x) + u32(y) * params.width] = vec4<f32>(px.rgb, sum / weight);
}
"#;

impl Filter for Feather {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let r = self.radius as i32;
        let w = width as i32;
        let h = height as i32;
        let sigma = r as f32 / 2.0;
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut weight = 0.0f32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = (x + dx).max(0).min(w - 1) as usize;
                        let sy = (y + dy).max(0).min(h - 1) as usize;
                        let d2 = (dx * dx + dy * dy) as f32;
                        let wg = (-d2 * inv2s2).exp();
                        sum += input[(sy * width as usize + sx) * 4 + 3] * wg;
                        weight += wg;
                    }
                }
                let i = (y as usize * width as usize + x as usize) * 4;
                out[i + 3] = sum / weight;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, self.radius);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            FEATHER_WGSL.to_string(),
            "main",
            [16, 16, 1],
            p,
        )])
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}
