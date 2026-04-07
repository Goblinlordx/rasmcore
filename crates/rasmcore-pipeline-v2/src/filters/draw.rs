//! Drawing operation filters — SDF-based shape rendering.
//!
//! Each filter composites a shape onto the input image using signed distance
//! fields for anti-aliased edges. GPU shaders compute the SDF per-pixel.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;
use std::f32::consts::PI;

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}
fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

/// Blend a color onto a pixel with given coverage (0..1).
#[inline(always)]
fn blend(dst: &mut [f32], r: f32, g: f32, b: f32, a: f32, coverage: f32) {
    let ca = a * coverage;
    dst[0] = dst[0] * (1.0 - ca) + r * ca;
    dst[1] = dst[1] * (1.0 - ca) + g * ca;
    dst[2] = dst[2] * (1.0 - ca) + b * ca;
    dst[3] = dst[3] * (1.0 - ca) + ca;
}

/// SDF anti-aliasing: smoothstep from stroke_width to stroke_width+1 pixel.
#[inline(always)]
fn sdf_coverage(dist: f32, stroke_width: f32, fill: bool) -> f32 {
    if fill {
        smoothstep_f32(0.5, -0.5, dist)
    } else {
        let half = stroke_width * 0.5;
        smoothstep_f32(half + 0.5, half - 0.5, dist.abs())
    }
}

#[inline(always)]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Common SDF blending WGSL snippet.
const SDF_BLEND_WGSL: &str = r#"
fn sdf_coverage_fill(d: f32) -> f32 { return smoothstep(0.5, -0.5, d); }
fn sdf_coverage_stroke(d: f32, half_w: f32) -> f32 { return smoothstep(half_w + 0.5, half_w - 0.5, abs(d)); }
fn sdf_blend(bg: vec4<f32>, color: vec4<f32>, coverage: f32) -> vec4<f32> {
  let ca = color.w * coverage;
  return vec4<f32>(bg.rgb * (1.0 - ca) + color.rgb * ca, bg.w * (1.0 - ca) + ca);
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Draw Line
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased line segment.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_line", category = "draw")]
pub struct DrawLine {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 10.0)] pub x1: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 10.0)] pub y1: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub x2: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub y2: f32,
    #[param(min = 0.5, max = 50.0, step = 0.5, default = 2.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
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
        let ax = self.x1; let ay = self.y1;
        let bx = self.x2; let by = self.y2;
        let bax = bx - ax; let bay = by - ay;
        let bab = bax * bax + bay * bay;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 - ax; let py = y as f32 - ay;
                let h = ((px * bax + py * bay) / bab).max(0.0).min(1.0);
                let dx = px - bax * h; let dy = py - bay * h;
                let dist = (dx * dx + dy * dy).sqrt();
                let cov = sdf_coverage(dist, self.stroke_width, false);
                if cov > 0.0 {
                    let i = ((y * width + x) * 4) as usize;
                    blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_LINE_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.x1); gpu_push_f32(&mut p, self.y1);
        gpu_push_f32(&mut p, self.x2); gpu_push_f32(&mut p, self.y2);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Draw Rect
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased rectangle (filled or stroked).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_rect", category = "draw")]
pub struct DrawRect {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 50.0)] pub x: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 50.0)] pub y: f32,
    #[param(min = 1.0, max = 8192.0, step = 1.0, default = 200.0)] pub rect_width: f32,
    #[param(min = 1.0, max = 8192.0, step = 1.0, default = 100.0)] pub rect_height: f32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
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
                    blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_RECT_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.x); gpu_push_f32(&mut p, self.y);
        gpu_push_f32(&mut p, self.rect_width); gpu_push_f32(&mut p, self.rect_height);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Draw Circle
// ═══════════════════════════════════════════════════════════════════════════

/// Draw an anti-aliased circle (filled or stroked).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_circle", category = "draw")]
pub struct DrawCircle {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cx: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cy: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 80.0)] pub radius: f32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
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
                    blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_CIRCLE_WGSL}");
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.cx); gpu_push_f32(&mut p, self.cy);
        gpu_push_f32(&mut p, self.radius); gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
// Draw Polygon (regular N-gon)
// ═══════════════════════════════════════════════════════════════════════════

/// Draw a regular polygon with N sides.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "draw_polygon", category = "draw")]
pub struct DrawPolygon {
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cx: f32,
    #[param(min = 0.0, max = 8192.0, step = 1.0, default = 200.0)] pub cy: f32,
    #[param(min = 1.0, max = 4096.0, step = 1.0, default = 80.0)] pub radius: f32,
    #[param(min = 3, max = 24, step = 1, default = 6)] pub sides: u32,
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 0.0)] pub stroke_width: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.8)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
}

const DRAW_POLYGON_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, pcx: f32, pcy: f32, radius: f32, sides: f32, stroke_width: f32, cr: f32, cg: f32, cb: f32, ca: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let dx = x - params.pcx; let dy = y - params.pcy;
  let angle = atan2(dy, dx);
  let n = params.sides;
  let sector = 2.0 * PI / n;
  // Fold angle into one sector
  let a = ((angle % sector) + sector) % sector - sector * 0.5;
  let r = length(vec2<f32>(dx, dy));
  let d = r * cos(a) - params.radius * cos(PI / n);
  var cov: f32;
  if (params.stroke_width > 0.0) { cov = sdf_coverage_stroke(d, params.stroke_width * 0.5); }
  else { cov = sdf_coverage_fill(d); }
  let color = vec4<f32>(params.cr, params.cg, params.cb, params.ca);
  output[idx] = sdf_blend(input[idx], color, cov);
}
"#;

impl Filter for DrawPolygon {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = self.sides.max(3) as f32;
        let fill = self.stroke_width <= 0.0;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - self.cx;
                let dy = y as f32 - self.cy;
                let angle = dy.atan2(dx);
                let sector = (2.0 * PI) / n;
                let a = (angle % sector) - sector * 0.5;
                let r = (dx * dx + dy * dy).sqrt();
                let dist = r * a.cos() - self.radius * (PI / n).cos();
                let cov = sdf_coverage(dist, self.stroke_width, fill);
                if cov > 0.0 {
                    let i = ((y * width + x) * 4) as usize;
                    blend(&mut out[i..i+4], self.color_r, self.color_g, self.color_b, self.color_a, cov);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SDF_BLEND_WGSL}\n{DRAW_POLYGON_WGSL}");
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.cx); gpu_push_f32(&mut p, self.cy);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.sides.max(3) as f32);
        gpu_push_f32(&mut p, self.stroke_width);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Solid Fill — fill entire image with a solid color
// ═══════════════════════════════════════════════════════════════════════════

/// Fill the image with a solid color (useful as a drawing primitive).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solid_fill", category = "draw")]
pub struct SolidFill {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub color_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color_a: f32,
}

const SOLID_FILL_WGSL: &str = r#"
struct Params { width: u32, height: u32, cr: f32, cg: f32, cb: f32, ca: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let bg = input[idx];
  let ca = params.ca;
  output[idx] = vec4<f32>(bg.rgb * (1.0 - ca) + vec3<f32>(params.cr, params.cg, params.cb) * ca, bg.w * (1.0 - ca) + ca);
}
"#;

impl Filter for SolidFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let ca = self.color_a;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0] * (1.0 - ca) + self.color_r * ca;
            pixel[1] = pixel[1] * (1.0 - ca) + self.color_g * ca;
            pixel[2] = pixel[2] * (1.0 - ca) + self.color_b * ca;
            pixel[3] = pixel[3] * (1.0 - ca) + ca;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.color_r); gpu_push_f32(&mut p, self.color_g);
        gpu_push_f32(&mut p, self.color_b); gpu_push_f32(&mut p, self.color_a);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(SOLID_FILL_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Text rendering
// ═══════════════════════════════════════════════════════════════════════════

/// Draw text onto the image using an externally-provided Font resource.
///
/// The font must be set on the pipeline via set_font() before using draw_text.
/// Text is rendered using CPU glyph rasterization from the cached Font atlas.
/// GPU path: not yet implemented (future: atlas texture + instanced quads).
pub struct DrawTextNode {
    upstream: u32,
    info: crate::node::NodeInfo,
    font: std::rc::Rc<crate::font::Font>,
    text: String,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 4],
}

impl DrawTextNode {
    pub fn new(
        upstream: u32,
        info: crate::node::NodeInfo,
        font: std::rc::Rc<crate::font::Font>,
        text: String,
        x: f32,
        y: f32,
        size: f32,
        color: [f32; 4],
    ) -> Self {
        Self { upstream, info, font, text, x, y, size, color }
    }
}

impl crate::node::Node for DrawTextNode {
    fn info(&self) -> crate::node::NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: crate::rect::Rect,
        upstream: &mut dyn crate::node::Upstream,
    ) -> Result<Vec<f32>, crate::node::PipelineError> {
        let mut pixels = upstream.request(self.upstream, request)?;
        self.font.render_text(
            &mut pixels,
            request.width,
            request.height,
            &self.text,
            self.x,
            self.y,
            self.size,
            self.color,
        );
        Ok(pixels)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }
}


// Register draw_text as an operation (handled specially by pipeline, not via FilterFactory)
inventory::submit! {
    &crate::registry::OperationRegistration {
        name: "draw_text",
        display_name: "Draw Text",
        category: "draw",
        kind: crate::registry::OperationKind::Filter,
        capabilities: crate::registry::OperationCapabilities {
            gpu: false, analytic: false, affine: false, clut: false,
        },
        doc_path: "",
        cost: "O(n)",
        params: &[
            crate::registry::ParamDescriptor {
                name: "text", value_type: crate::registry::ParamType::String,
                min: None, max: None, step: None, default: None,
                hint: Some("text to render"), description: "Text string to render",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "x", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(10000.0), step: Some(1.0), default: Some(10.0),
                hint: Some("rc.pixels"), description: "X position",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "y", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(10000.0), step: Some(1.0), default: Some(10.0),
                hint: Some("rc.pixels"), description: "Y position",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "size", value_type: crate::registry::ParamType::F32,
                min: Some(4.0), max: Some(500.0), step: Some(1.0), default: Some(24.0),
                hint: Some("rc.pixels"), description: "Font size in pixels",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_r", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color red",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_g", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color green",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_b", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color blue",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_a", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color alpha",
                constraints: &[],
            },
        ],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_draw_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["draw_line", "draw_rect", "draw_circle", "draw_ellipse",
                       "draw_arc", "draw_polygon", "solid_fill"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn draw_circle_modifies_pixels() {
        let input = vec![0.0f32; 100 * 100 * 4];
        let f = DrawCircle {
            cx: 50.0, cy: 50.0, radius: 20.0, stroke_width: 0.0,
            color_r: 1.0, color_g: 0.0, color_b: 0.0, color_a: 1.0,
        };
        let out = f.compute(&input, 100, 100).unwrap();
        // Center pixel should be red
        let i = (50 * 100 + 50) * 4;
        assert!(out[i] > 0.9, "center R should be ~1.0, got {}", out[i]);
        // Corner pixel should be unchanged
        assert!(out[0] < 0.01, "corner should be black");
    }

    #[test]
    fn solid_fill_blends() {
        let input = vec![1.0, 1.0, 1.0, 1.0]; // white pixel
        let f = SolidFill { color_r: 0.0, color_g: 0.0, color_b: 0.0, color_a: 0.5 };
        let out = f.compute(&input, 1, 1).unwrap();
        // 50% black over white = 0.5 gray
        assert!((out[0] - 0.5).abs() < 0.01);
    }
}
