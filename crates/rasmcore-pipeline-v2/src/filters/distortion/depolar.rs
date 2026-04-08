//! Depolar distortion filter.

use crate::node::{PipelineError};
use crate::ops::Filter;

use std::f32::consts::PI;
use super::super::helpers::{gpu_params_wh, sample_bilinear};
use super::{SAMPLE_BILINEAR_WGSL, gpu_params_push_u32};

// Depolar (Polar → Cartesian)
// ═══════════════════════════════════════════════════════════════════════════

/// Depolar — inverse polar coordinate transform (polar to Cartesian).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "depolar", category = "distortion")]
pub struct Depolar;

const DEPOLAR_WGSL: &str = r#"
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
  let dx = f32(x) + 0.5 - cx; let dy = f32(y) + 0.5 - cy;
  let radius = sqrt(dx * dx + dy * dy);
  var angle = atan2(dx, dy);
  var xx = angle / (2.0 * PI);
  xx = xx - round(xx);
  let sx = xx * wf + cx - 0.5;
  let sy = radius * (hf / max_radius) - 0.5;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Depolar {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let wf = width as f32;
        let hf = height as f32;
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let max_radius = cx.min(cy);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let radius = (dx * dx + dy * dy).sqrt();
                let angle = dx.atan2(dy);
                let mut xx = angle / (2.0 * PI);
                xx -= xx.round();
                let sx = xx * wf + cx - 0.5;
                let sy = radius * (hf / max_radius) - 0.5;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{DEPOLAR_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut params, 0);
        gpu_params_push_u32(&mut params, 0);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}
