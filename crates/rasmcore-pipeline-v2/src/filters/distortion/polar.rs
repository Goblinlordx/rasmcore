//! Polar distortion filter.

use crate::node::{PipelineError};
use crate::ops::Filter;

use std::f32::consts::PI;
use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_u32};

// Polar (Cartesian → Polar)
// ═══════════════════════════════════════════════════════════════════════════

/// Polar coordinate transform — Cartesian to polar mapping.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "polar", category = "distortion")]
pub struct Polar;

const POLAR_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let wf = f32(params.width); let hf = f32(params.height);
  let cx = wf * 0.5; let cy = hf * 0.5;
  let max_radius = min(cx, cy);
  let dx = f32(x) + 0.5; let dy = f32(y) + 0.5;
  let angle = (dx - cx) / wf * 2.0 * PI;
  let radius = dy / hf * max_radius;
  let sx = cx + radius * sin(angle) - 0.5;
  let sy = cy + radius * cos(angle) - 0.5;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Polar {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let wf = width as f32;
        let hf = height as f32;
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let max_radius = cx.min(cy);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 + 0.5;
                let dy = y as f32 + 0.5;
                let angle = (dx - cx) / wf * 2.0 * PI;
                let radius = dy / hf * max_radius;
                let sx = cx + radius * angle.sin() - 0.5;
                let sy = cy + radius * angle.cos() - 0.5;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{POLAR_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut params, 0); // pad
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}
