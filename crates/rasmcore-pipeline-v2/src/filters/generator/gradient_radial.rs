//! GradientRadial generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};

// Gradient Radial
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a radial gradient from center outward.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gradient_radial", category = "generator")]
pub struct GradientRadial {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub center_y: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub inner_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub inner_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub inner_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub outer_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub outer_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub outer_b: f32,
}

const GRADIENT_RADIAL_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, ir: f32, ig: f32, ib: f32, or_: f32, og: f32, ob: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / f32(params.width);
  let y = f32(idx / params.width) / f32(params.height);
  let dx = x - params.cx; let dy = y - params.cy;
  let t = clamp(length(vec2<f32>(dx, dy)) * 2.0, 0.0, 1.0);
  let inner = vec3<f32>(params.ir, params.ig, params.ib);
  let outer = vec3<f32>(params.or_, params.og, params.ob);
  output[idx] = vec4<f32>(mix(inner, outer, vec3<f32>(t)), 1.0);
}
"#;

impl Filter for GradientRadial {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let dx = nx - self.center_x;
                let dy = ny - self.center_y;
                let t = ((dx * dx + dy * dy).sqrt() * 2.0).min(1.0);
                let i = ((y * width + x) * 4) as usize;
                out[i] = self.inner_r + t * (self.outer_r - self.inner_r);
                out[i+1] = self.inner_g + t * (self.outer_g - self.inner_g);
                out[i+2] = self.inner_b + t * (self.outer_b - self.inner_b);
                out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x); gpu_push_f32(&mut p, self.center_y);
        gpu_push_f32(&mut p, self.inner_r); gpu_push_f32(&mut p, self.inner_g); gpu_push_f32(&mut p, self.inner_b);
        gpu_push_f32(&mut p, self.outer_r); gpu_push_f32(&mut p, self.outer_g); gpu_push_f32(&mut p, self.outer_b);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(GRADIENT_RADIAL_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}
