//! DrawEllipse drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::{SDF_BLEND_WGSL, sdf_coverage, blend};

// Draw Ellipse
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased ellipse (filled or stroked).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_ellipse", category = "draw")]
pub struct DrawEllipse {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cx: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cy: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 120.0)] pub rx: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 60.0)] pub ry: f32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.8)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
}

const DRAW_ELLIPSE_WGSL: &str = r#"
struct Params { width: u32, height: u32, ecx: f32, ecy: f32, erx: f32, ery: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let nx = (x - params.ecx) / params.erx; let ny = (y - params.ecy) / params.ery;
  // Approximate ellipse SDF via normalized-space circle distance scaled by avg radius
  let r = length(vec2<f32>(nx, ny));
  let avg_r = (params.erx + params.ery) * 0.5;
  let d = (r - 1.0) * avg_r;
  var cov: f32;
  if (params.stroke_width > 0.0) { cov = sdf_coverage_stroke(d, params.stroke_width * 0.5); }
  else { cov = sdf_coverage_fill(d); }
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawEllipse {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let fill = self.stroke_width <= 0.0;
        let avg_r = (self.rx + self.ry) * 0.5;
        for y in 0..height {
            for x in 0..width {
                let nx = (x as f32 - self.cx) / self.rx;
                let ny = (y as f32 - self.cy) / self.ry;
                let r = (nx * nx + ny * ny).sqrt();
                let dist = (r - 1.0) * avg_r;
                let cov = sdf_coverage(dist, self.stroke_width, fill);
                if cov > 0.0 {
                    let i = ((y * width + x) * 4) as usize;
                    blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_ELLIPSE_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.cx); gpu_push_f32(&mut p, self.cy);
        gpu_push_f32(&mut p, self.rx); gpu_push_f32(&mut p, self.ry);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
