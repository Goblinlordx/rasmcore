//! Plasma generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32};
use std::f32::consts::PI;

// Plasma
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a colorful plasma pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "plasma", category = "generator")]
pub struct Plasma {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 30.0)]
    pub scale: f32,
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 0.0)]
    pub time: f32,
}

const PLASMA_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, scale: f32, time: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / params.scale;
  let y = f32(idx / params.width) / params.scale;
  let t = params.time;
  let v1 = sin(x + t);
  let v2 = sin(y + t * 0.7);
  let v3 = sin(x + y + t * 0.5);
  let v4 = sin(sqrt(x * x + y * y) + t * 0.3);
  let v = (v1 + v2 + v3 + v4) * 0.25;
  let r = sin(v * PI) * 0.5 + 0.5;
  let g = sin(v * PI + 2.094) * 0.5 + 0.5;
  let b = sin(v * PI + 4.189) * 0.5 + 0.5;
  output[idx] = vec4<f32>(r, g, b, 1.0);
}
"#;

impl Filter for Plasma {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let t = self.time;
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32 / self.scale;
                let yf = y as f32 / self.scale;
                let v1 = (xf + t).sin();
                let v2 = (yf + t * 0.7).sin();
                let v3 = (xf + yf + t * 0.5).sin();
                let v4 = ((xf * xf + yf * yf).sqrt() + t * 0.3).sin();
                let v = (v1 + v2 + v3 + v4) * 0.25;
                let i = ((y * width + x) * 4) as usize;
                out[i] = (v * PI).sin() * 0.5 + 0.5;
                out[i + 1] = (v * PI + 2.094).sin() * 0.5 + 0.5;
                out[i + 2] = (v * PI + 4.189).sin() * 0.5 + 0.5;
                out[i + 3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_f32(&mut p, self.time);
        Some(vec![GpuShader::new(
            PLASMA_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
