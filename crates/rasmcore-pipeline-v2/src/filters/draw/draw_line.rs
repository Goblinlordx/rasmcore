//! DrawLine drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::{SDF_BLEND_WGSL, blend, sdf_coverage};

// Draw Line
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased line segment.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_line", category = "draw")]
pub struct DrawLine {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 10.0)]
    pub x1: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 10.0)]
    pub y1: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)]
    pub x2: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)]
    pub y2: f32,
    #[param(min = 0.5, max = 50.0, step = 0.5, default = 2.0)]
    pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_a: f32,
}

const DRAW_LINE_WGSL: &str = r#"
struct Params { width: u32, height: u32, x1: f32, y1: f32, x2: f32, y2: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn sdf_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let d = sdf_line(vec2<f32>(x, y), vec2<f32>(params.x1, params.y1), vec2<f32>(params.x2, params.y2));
  let half_w = params.stroke_width * 0.5;
  let cov = smoothstep(half_w + 0.5, half_w - 0.5, d);
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawLine {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let ax = self.x1;
        let ay = self.y1;
        let bx = self.x2;
        let by = self.y2;
        let bax = bx - ax;
        let bay = by - ay;
        let bab = bax * bax + bay * bay;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 - ax;
                let py = y as f32 - ay;
                let h = ((px * bax + py * bay) / bab).max(0.0).min(1.0);
                let dx = px - bax * h;
                let dy = py - bay * h;
                let dist = (dx * dx + dy * dy).sqrt();
                let cov = sdf_coverage(dist, self.stroke_width, false);
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
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_LINE_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.x1);
        gpu_push_f32(&mut p, self.y1);
        gpu_push_f32(&mut p, self.x2);
        gpu_push_f32(&mut p, self.y2);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r);
        gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b);
        gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
