//! DrawRect drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::{SDF_BLEND_WGSL, blend, sdf_coverage};

// Draw Rect
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased rectangle (filled or stroked).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_rect", category = "draw")]
pub struct DrawRect {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 50.0)]
    pub x: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 50.0)]
    pub y: f32,
    #[param(min = 1.0, max = 8192.0, step = 1.0, default = 200.0)]
    pub rect_width: f32,
    #[param(min = 1.0, max = 8192.0, step = 1.0, default = 100.0)]
    pub rect_height: f32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)]
    pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub color_a: f32,
}

const DRAW_RECT_WGSL: &str = r#"
struct Params { width: u32, height: u32, rx: f32, ry: f32, rw: f32, rh: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn sdf_box(p: vec2<f32>, center: vec2<f32>, half: vec2<f32>) -> f32 {
  let d = abs(p - center) - half;
  return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let center = vec2<f32>(params.rx + params.rw * 0.5, params.ry + params.rh * 0.5);
  let half = vec2<f32>(params.rw * 0.5, params.rh * 0.5);
  let d = sdf_box(vec2<f32>(x, y), center, half);
  var cov: f32;
  if (params.stroke_width > 0.0) { cov = sdf_coverage_stroke(d, params.stroke_width * 0.5); }
  else { cov = sdf_coverage_fill(d); }
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawRect {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.x + self.rect_width * 0.5;
        let cy = self.y + self.rect_height * 0.5;
        let hx = self.rect_width * 0.5;
        let hy = self.rect_height * 0.5;
        let fill = self.stroke_width <= 0.0;
        for y in 0..height {
            for x in 0..width {
                let dx = (x as f32 - cx).abs() - hx;
                let dy = (y as f32 - cy).abs() - hy;
                let dist = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0);
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
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_RECT_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.x);
        gpu_push_f32(&mut p, self.y);
        gpu_push_f32(&mut p, self.rect_width);
        gpu_push_f32(&mut p, self.rect_height);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r);
        gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b);
        gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
