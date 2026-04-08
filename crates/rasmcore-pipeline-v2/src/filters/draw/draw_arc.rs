//! DrawArc drawing filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use std::f32::consts::PI;
use super::super::helpers::{gpu_params_wh, gpu_push_f32};
use super::{SDF_BLEND_WGSL, sdf_coverage, blend};

// Draw Arc
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased arc (section of a circle).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_arc", category = "draw")]
pub struct DrawArc {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cx: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cy: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 80.0)] pub radius: f32,
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0, hint = "rc.angle_deg")] pub start_angle: f32,
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 270.0, hint = "rc.angle_deg")] pub end_angle: f32,
    #[param(min = 0.5, max = 50.0, step = 0.5, default = 3.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
}

const DRAW_ARC_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, acx: f32, acy: f32, radius: f32, start_angle: f32, end_angle: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let dx = x - params.acx; let dy = y - params.acy;
  let dist = abs(length(vec2<f32>(dx, dy)) - params.radius);
  var angle = atan2(dy, dx);
  if (angle < 0.0) { angle += 2.0 * PI; }
  let sa = params.start_angle; let ea = params.end_angle;
  var in_arc: bool;
  if (sa <= ea) { in_arc = angle >= sa && angle <= ea; }
  else { in_arc = angle >= sa || angle <= ea; }
  if (!in_arc) { output[idx] = input[idx]; return; }
  let half_w = params.stroke_width * 0.5;
  let cov = sdf_coverage_stroke(dist, half_w);
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawArc {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let sa = self.start_angle.to_radians();
        let ea = self.end_angle.to_radians();
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - self.cx;
                let dy = y as f32 - self.cy;
                let dist_from_circle = ((dx * dx + dy * dy).sqrt() - self.radius).abs();
                let angle = dy.atan2(dx).rem_euclid(2.0 * PI);
                let in_arc = if sa <= ea { angle >= sa && angle <= ea }
                             else { angle >= sa || angle <= ea };
                if in_arc {
                    let cov = sdf_coverage(dist_from_circle, self.stroke_width, false);
                    if cov > 0.0 {
                        let i = ((y * width + x) * 4) as usize;
                        blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                    }
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_ARC_WGSL}");
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.cx); gpu_push_f32(&mut p, self.cy);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.start_angle.to_radians());
        gpu_push_f32(&mut p, self.end_angle.to_radians());
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}
