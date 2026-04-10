//! SolidColor generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Solid Color
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a solid color fill — useful as mask, background, or pipeline building block.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solid_color", category = "generator")]
pub struct SolidColor {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub a: f32,
}

const SOLID_COLOR_WGSL: &str = r#"
struct Params { width: u32, height: u32, r: f32, g: f32, b: f32, a: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  output[idx] = vec4<f32>(params.r, params.g, params.b, params.a);
}
"#;

impl Filter for SolidColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let n = (width * height) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            out.extend_from_slice(&[self.r, self.g, self.b, self.a]);
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.r);
        gpu_push_f32(&mut p, self.g);
        gpu_push_f32(&mut p, self.b);
        gpu_push_f32(&mut p, self.a);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            SOLID_COLOR_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
