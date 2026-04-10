//! Sponge tool filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::smoothstep_f32;

// Sponge — local saturation adjustment within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Sponge tool — boost or reduce saturation within a circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sponge", category = "tool")]
pub struct Sponge {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")]
    pub radius: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.3)]
    pub amount: f32,
}

const SPONGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, radius: f32, amount: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let dx = x - params.cx; let dy = y - params.cy;
  let dist = sqrt(dx*dx + dy*dy);
  if (dist >= params.radius) { output[idx] = input[idx]; return; }
  let falloff = 1.0 - smoothstep(params.radius * 0.5, params.radius, dist);
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  let gray = vec3<f32>(luma);
  let sat_factor = 1.0 + params.amount * falloff;
  let rgb = mix(gray, px.rgb, vec3<f32>(sat_factor));
  output[idx] = vec4<f32>(rgb, px.w);
}
"#;

impl Filter for Sponge {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= self.radius {
                    continue;
                }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.5, self.radius, dist);
                let i = ((y * width + x) * 4) as usize;
                let luma = out[i] * 0.2126 + out[i + 1] * 0.7152 + out[i + 2] * 0.0722;
                let sat = 1.0 + self.amount * falloff;
                out[i] = luma + sat * (out[i] - luma);
                out[i + 1] = luma + sat * (out[i + 1] - luma);
                out[i + 2] = luma + sat * (out[i + 2] - luma);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.amount);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(
            SPONGE_WGSL.to_string(),
            "main",
            [256, 1, 1],
            p,
        )])
    }
}
