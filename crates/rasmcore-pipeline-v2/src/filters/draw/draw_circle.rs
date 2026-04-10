//! DrawCircle drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::{SDF_BLEND_WGSL, blend, sdf_coverage};

// Draw Circle
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased circle (filled or stroked).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_circle", category = "draw")]
pub struct DrawCircle {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)]
    pub cx: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)]
    pub cy: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 80.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)]
    pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_a: f32,
}

const DRAW_CIRCLE_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, radius: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let d = length(vec2<f32>(x, y) - vec2<f32>(params.cx, params.cy)) - params.radius;
  var cov: f32;
  if (params.stroke_width > 0.0) { cov = sdf_coverage_stroke(d, params.stroke_width * 0.5); }
  else { cov = sdf_coverage_fill(d); }
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawCircle {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let fill = self.stroke_width <= 0.0;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - self.cx;
                let dy = y as f32 - self.cy;
                let dist = (dx * dx + dy * dy).sqrt() - self.radius;
                let cov = sdf_coverage(dist, self.stroke_width, fill);
                if cov > 0.0 {
                    let i = ((y * width + x) * 4) as usize;
                    blend(
                        &mut out[i..i + 4],
                        self.color_r,
                        self.color_g,
                        self.color_b,
                        self.color_a,
                        cov,
                    );
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_CIRCLE_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.cx);
        gpu_push_f32(&mut p, self.cy);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r);
        gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b);
        gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
