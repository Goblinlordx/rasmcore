//! Distortion filters — coordinate remapping with bilinear sampling.
//!
//! All filters use inverse mapping: for each output pixel, compute the
//! source coordinate, then bilinear-sample the input. GPU shaders follow
//! the same pattern with `sample_bilinear_f32()`.

use crate::node::PipelineError;
use crate::ops::Filter;
use std::f32::consts::PI;

use super::helpers::{gpu_params_wh, sample_bilinear};

/// WGSL bilinear sampling helper — prepended to each distortion shader.
const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.width];
  let p10 = input[u32(x1) + u32(y0) * params.width];
  let p01 = input[u32(x0) + u32(y1) * params.width];
  let p11 = input[u32(x1) + u32(y1) * params.width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}
"#;

// ─── GPU param helpers ─────────────────────────────────────────────────────

fn gpu_params_push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════
// Barrel Distortion
// ═══════════════════════════════════════════════════════════════════════════

/// Barrel/pincushion distortion.
/// k1 > 0: barrel, k1 < 0: pincushion.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "barrel", category = "distortion")]
pub struct Barrel {
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.3)]
    pub k1: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub k2: f32,
}

const BARREL_WGSL: &str = r#"
struct Params { width: u32, height: u32, k1: f32, k2: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let nx = (f32(x) - cx) / cx; let ny = (f32(y) - cy) / cy;
  let r2 = nx * nx + ny * ny; let r4 = r2 * r2;
  let d = 1.0 + params.k1 * r2 + params.k2 * r4;
  let sx = nx * d * cx + cx; let sy = ny * d * cy + cy;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Barrel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let nx = (x as f32 - cx) / cx;
                let ny = (y as f32 - cy) / cy;
                let r2 = nx * nx + ny * ny;
                let r4 = r2 * r2;
                let d = 1.0 + self.k1 * r2 + self.k2 * r4;
                let sx = nx * d * cx + cx;
                let sy = ny * d * cy + cy;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> { None }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{BARREL_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.k1);
        gpu_params_push_f32(&mut params, self.k2);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spherize
// ═══════════════════════════════════════════════════════════════════════════

/// Spherize distortion — spherical lens effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "spherize", category = "distortion")]
pub struct Spherize {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
}

const SPHERIZE_WGSL: &str = r#"
struct Params { width: u32, height: u32, amount: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let nx = (f32(x) - cx) / cx; let ny = (f32(y) - cy) / cy;
  let r = sqrt(nx * nx + ny * ny);
  var sx: f32; var sy: f32;
  if (r < 1.0 && r > 0.0) {
    let theta = asin(r) / r;
    let factor = mix(1.0, theta, params.amount);
    sx = nx * factor * cx + cx; sy = ny * factor * cy + cy;
  } else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Spherize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let nx = (x as f32 - cx) / cx;
                let ny = (y as f32 - cy) / cy;
                let r = (nx * nx + ny * ny).sqrt();
                let (sx, sy) = if r < 1.0 && r > 0.0 {
                    let theta = r.asin() / r;
                    let factor = 1.0 + self.amount * (theta - 1.0);
                    (nx * factor * cx + cx, ny * factor * cy + cy)
                } else {
                    (x as f32, y as f32)
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SPHERIZE_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.amount);
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Swirl
// ═══════════════════════════════════════════════════════════════════════════

/// Swirl — rotational distortion decreasing with distance from center.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "swirl", category = "distortion")]
pub struct Swirl {
    #[param(min = -10.0, max = 10.0, step = 0.1, default = 2.0, hint = "rc.angle_deg")]
    pub angle: f32,
    #[param(min = 1.0, max = 2000.0, step = 1.0, default = 300.0, hint = "rc.pixels")]
    pub radius: f32,
}

const SWIRL_WGSL: &str = r#"
struct Params { width: u32, height: u32, angle: f32, radius: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let dx = f32(x) - cx; let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);
  var sx: f32; var sy: f32;
  if (dist < params.radius && params.radius > 0.0) {
    let t = 1.0 - dist / params.radius;
    let sa = params.angle * t * t;
    let ct = cos(sa); let st = sin(sa);
    sx = dx * ct - dy * st + cx; sy = dx * st + dy * ct + cy;
  } else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Swirl {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let (sx, sy) = if dist < self.radius && self.radius > 0.0 {
                    let t = 1.0 - dist / self.radius;
                    let sa = self.angle * t * t;
                    let (st, ct) = sa.sin_cos();
                    (dx * ct - dy * st + cx, dx * st + dy * ct + cy)
                } else {
                    (x as f32, y as f32)
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SWIRL_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.angle);
        gpu_params_push_f32(&mut params, self.radius);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ripple
// ═══════════════════════════════════════════════════════════════════════════

/// Ripple — concentric wave distortion from center point.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "ripple", category = "distortion")]
pub struct Ripple {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 10.0)]
    pub amplitude: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub wavelength: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
}

const RIPPLE_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, amplitude: f32, wavelength: f32, center_x: f32, center_y: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let cx = params.center_x * f32(params.width);
  let cy = params.center_y * f32(params.height);
  let dx = f32(x) - cx; let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);
  let disp = params.amplitude * sin(2.0 * PI * dist / params.wavelength);
  var sx: f32; var sy: f32;
  if (dist > 0.0) { sx = f32(x) + disp * dx / dist; sy = f32(y) + disp * dy / dist; }
  else { sx = f32(x); sy = f32(y); }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Ripple {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let disp = self.amplitude * (2.0 * PI * dist / self.wavelength).sin();
                let (sx, sy) = if dist > 0.0 {
                    (x as f32 + disp * dx / dist, y as f32 + disp * dy / dist)
                } else {
                    (x as f32, y as f32)
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{RIPPLE_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.amplitude);
        gpu_params_push_f32(&mut params, self.wavelength);
        gpu_params_push_f32(&mut params, self.center_x);
        gpu_params_push_f32(&mut params, self.center_y);
        gpu_params_push_u32(&mut params, 0); // pad
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave
// ═══════════════════════════════════════════════════════════════════════════

/// Wave — sinusoidal displacement along one axis.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "wave", category = "distortion")]
pub struct Wave {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 10.0)]
    pub amplitude: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub wavelength: f32,
    #[param(min = 0, max = 1, default = 0)]
    pub vertical: bool,
}

const WAVE_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, amplitude: f32, wavelength: f32, vertical: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  var sx: f32; var sy: f32;
  if (params.vertical > 0.5) {
    sx = f32(x) + params.amplitude * sin(2.0 * PI * f32(y) / params.wavelength); sy = f32(y);
  } else {
    sx = f32(x); sy = f32(y) + params.amplitude * sin(2.0 * PI * f32(x) / params.wavelength);
  }
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Wave {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = if self.vertical {
                    (x as f32 + self.amplitude * (2.0 * PI * y as f32 / self.wavelength).sin(), y as f32)
                } else {
                    (x as f32, y as f32 + self.amplitude * (2.0 * PI * x as f32 / self.wavelength).sin())
                };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{WAVE_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.amplitude);
        gpu_params_push_f32(&mut params, self.wavelength);
        gpu_params_push_f32(&mut params, if self.vertical { 1.0 } else { 0.0 });
        gpu_params_push_u32(&mut params, 0); // pad
        gpu_params_push_u32(&mut params, 0); // pad
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
// Liquify
// ═══════════════════════════════════════════════════════════════════════════

/// Liquify push — directional displacement within circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "liquify", category = "distortion")]
pub struct Liquify {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)]
    pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 100.0, hint = "rc.pixels")]
    pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub strength: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 1.0)]
    pub direction_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub direction_y: f32,
}

const LIQUIFY_WGSL: &str = r#"
struct Params { width: u32, height: u32, center_x: f32, center_y: f32, radius: f32, strength: f32, direction_x: f32, direction_y: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn gaussian_weight(dist: f32, radius: f32) -> f32 { let t = dist / radius; return exp(-2.0 * t * t); }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let xf = f32(x); let yf = f32(y);
  let cx = params.center_x * f32(params.width); let cy = params.center_y * f32(params.height);
  let ddx = xf - cx; let ddy = yf - cy;
  let dist = sqrt(ddx * ddx + ddy * ddy);
  if (dist >= params.radius) { output[x + y * params.width] = input[x + y * params.width]; return; }
  let w = gaussian_weight(dist, params.radius) * params.strength;
  let sx = xf - params.direction_x * w * params.radius;
  let sy = yf - params.direction_y * w * params.radius;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Liquify {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32;
                let yf = y as f32;
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= self.radius { continue; }
                let t = dist / self.radius;
                let w = (-2.0 * t * t).exp() * self.strength;
                let sx = xf - self.direction_x * w * self.radius;
                let sy = yf - self.direction_y * w * self.radius;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{LIQUIFY_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.center_x * width as f32);
        gpu_params_push_f32(&mut params, self.center_y * height as f32);
        gpu_params_push_f32(&mut params, self.radius);
        gpu_params_push_f32(&mut params, self.strength);
        gpu_params_push_f32(&mut params, self.direction_x);
        gpu_params_push_f32(&mut params, self.direction_y);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mesh Warp (simplified — uniform grid displacement)
// ═══════════════════════════════════════════════════════════════════════════

/// Mesh warp — grid-based displacement mapping.
/// Uses a uniform displacement field for simplicity (no control point grid).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mesh_warp", category = "distortion")]
pub struct MeshWarp {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 10.0)]
    pub strength: f32,
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 40.0)]
    pub frequency: f32,
}

const MESH_WARP_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, strength: f32, frequency: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let dx = params.strength * sin(2.0 * PI * f32(y) / params.frequency);
  let dy = params.strength * sin(2.0 * PI * f32(x) / params.frequency);
  let sx = f32(x) + dx; let sy = f32(y) + dy;
  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for MeshWarp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let dx = self.strength * (2.0 * PI * y as f32 / self.frequency).sin();
                let dy = self.strength * (2.0 * PI * x as f32 / self.frequency).sin();
                let sx = x as f32 + dx;
                let sy = y as f32 + dy;
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{MESH_WARP_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.strength);
        gpu_params_push_f32(&mut params, self.frequency);
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Displacement Map
// ═══════════════════════════════════════════════════════════════════════════

/// Displacement map — uses the image's own color channels as displacement.
/// Red channel displaces X, green channel displaces Y.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "displacement_map", category = "distortion")]
pub struct DisplacementMap {
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 10.0)]
    pub scale: f32,
}

const DISPLACEMENT_MAP_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  let dx = (px.r - 0.5) * params.scale;
  let dy = (px.g - 0.5) * params.scale;
  let sx = f32(x) + dx; let sy = f32(y) + dy;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for DisplacementMap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                let dx = (input[i] - 0.5) * self.scale;
                let dy = (input[i + 1] - 0.5) * self.scale;
                let sx = x as f32 + dx;
                let sy = y as f32 + dy;
                let px = sample_bilinear(input, width, height, sx, sy);
                out[i..i + 4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<crate::node::GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{DISPLACEMENT_MAP_WGSL}");
        let mut params = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut params, self.scale);
        gpu_params_push_u32(&mut params, 0); // pad
        Some(vec![crate::node::GpuShader::new(shader, "main", [16, 16, 1], params)])
    }

    fn tile_overlap(&self) -> u32 { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&color);
        }
        pixels
    }

    #[test]
    fn barrel_identity_at_zero() {
        let input = solid_image(4, 4, [0.5, 0.3, 0.1, 1.0]);
        let f = Barrel { k1: 0.0, k2: 0.0 };
        let out = f.compute(&input, 4, 4).unwrap();
        // At k1=k2=0, distortion factor = 1.0, so output ≈ input
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }

    #[test]
    fn swirl_identity_at_zero_angle() {
        let input = solid_image(4, 4, [0.5, 0.3, 0.1, 1.0]);
        let f = Swirl { angle: 0.0, radius: 100.0 };
        let out = f.compute(&input, 4, 4).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }

    #[test]
    fn wave_changes_pixels() {
        let mut input = vec![0.0f32; 16 * 16 * 4];
        for i in 0..input.len() { input[i] = (i as f32 / input.len() as f32); }
        let f = Wave { amplitude: 5.0, wavelength: 8.0, vertical: false };
        let out = f.compute(&input, 16, 16).unwrap();
        assert_ne!(input, out);
    }

    #[test]
    fn all_distortion_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["barrel", "spherize", "swirl", "ripple", "wave",
                       "polar", "depolar", "liquify", "mesh_warp", "displacement_map"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }
}
