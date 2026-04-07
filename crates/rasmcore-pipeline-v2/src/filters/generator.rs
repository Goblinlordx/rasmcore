//! Generator filters — procedural pattern generation.
//!
//! These replace the input image with a generated pattern. All are pure
//! per-pixel math — ideal for GPU compute shaders.

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

// ═══════════════════════════════════════════════════════════════════════════
// Checkerboard
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a checkerboard pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "checkerboard", category = "generator")]
pub struct Checkerboard {
    #[param(min = 2.0, max = 500.0, step = 1.0, default = 32.0)] pub size: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color1_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color1_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub color1_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)] pub color2_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)] pub color2_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)] pub color2_b: f32,
}

const CHECKERBOARD_WGSL: &str = r#"
struct Params { width: u32, height: u32, size: f32, c1r: f32, c1g: f32, c1b: f32, c2r: f32, c2g: f32, c2b: f32, _pad: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let cx = u32(floor(x / params.size)); let cy = u32(floor(y / params.size));
  let check = (cx + cy) % 2u;
  var color: vec3<f32>;
  if (check == 0u) { color = vec3<f32>(params.c1r, params.c1g, params.c1b); }
  else { color = vec3<f32>(params.c2r, params.c2g, params.c2b); }
  output[idx] = vec4<f32>(color, 1.0);
}
"#;

impl Filter for Checkerboard {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let cx = (x as f32 / self.size).floor() as u32;
                let cy = (y as f32 / self.size).floor() as u32;
                let i = ((y * width + x) * 4) as usize;
                if (cx + cy) % 2 == 0 {
                    out[i] = self.color1_r; out[i+1] = self.color1_g; out[i+2] = self.color1_b;
                } else {
                    out[i] = self.color2_r; out[i+1] = self.color2_g; out[i+2] = self.color2_b;
                }
                out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.size);
        gpu_push_f32(&mut p, self.color1_r); gpu_push_f32(&mut p, self.color1_g); gpu_push_f32(&mut p, self.color1_b);
        gpu_push_f32(&mut p, self.color2_r); gpu_push_f32(&mut p, self.color2_g); gpu_push_f32(&mut p, self.color2_b);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(CHECKERBOARD_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gradient Linear
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a linear gradient between two colors.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gradient_linear", category = "generator")]
pub struct GradientLinear {
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0, hint = "rc.angle_deg")] pub angle: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub start_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub start_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub start_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub end_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub end_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub end_b: f32,
}

const GRADIENT_LINEAR_WGSL: &str = r#"
struct Params { width: u32, height: u32, angle: f32, sr: f32, sg: f32, sb: f32, er: f32, eg: f32, eb: f32, _pad: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / f32(params.width);
  let y = f32(idx / params.width) / f32(params.height);
  let ca = cos(params.angle); let sa = sin(params.angle);
  let t = clamp(x * ca + y * sa, 0.0, 1.0);
  let start = vec3<f32>(params.sr, params.sg, params.sb);
  let end = vec3<f32>(params.er, params.eg, params.eb);
  output[idx] = vec4<f32>(mix(start, end, vec3<f32>(t)), 1.0);
}
"#;

impl Filter for GradientLinear {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let a = self.angle.to_radians();
        let (sa, ca) = a.sin_cos();
        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let t = (nx * ca + ny * sa).max(0.0).min(1.0);
                let i = ((y * width + x) * 4) as usize;
                out[i] = self.start_r + t * (self.end_r - self.start_r);
                out[i+1] = self.start_g + t * (self.end_g - self.start_g);
                out[i+2] = self.start_b + t * (self.end_b - self.start_b);
                out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.angle.to_radians());
        gpu_push_f32(&mut p, self.start_r); gpu_push_f32(&mut p, self.start_g); gpu_push_f32(&mut p, self.start_b);
        gpu_push_f32(&mut p, self.end_r); gpu_push_f32(&mut p, self.end_g); gpu_push_f32(&mut p, self.end_b);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(GRADIENT_LINEAR_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
// Perlin Noise
// ═══════════════════════════════════════════════════════════════════════════

/// Generate Perlin noise pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perlin_noise", category = "generator")]
pub struct PerlinNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 50.0)] pub scale: f32,
    #[param(min = 1, max = 8, step = 1, default = 4)] pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub persistence: f32,
    #[param(min = 0, max = 99999, step = 1, default = 42, hint = "rc.seed")] pub seed: u32,
}

const PERLIN_NOISE_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, octaves: u32, persistence: f32, seed: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let k = vec2<f32>(0.3183099, 0.3678794);
  let pp = p * k + k.yx;
  return fract(16.0 * k.x * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

fn fbm(p: vec2<f32>, octaves: u32, persistence: f32) -> f32 {
  var value = 0.0; var amplitude = 1.0; var frequency = 1.0; var total_amp = 0.0;
  for (var i = 0u; i < octaves; i = i + 1u) {
    value += noise2(p * frequency) * amplitude;
    total_amp += amplitude;
    amplitude *= persistence;
    frequency *= 2.0;
  }
  return value / total_amp;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let p = vec2<f32>(x, y) / params.scale + vec2<f32>(f32(params.seed) * 0.1, f32(params.seed) * 0.07);
  let v = fbm(p, params.octaves, params.persistence);
  output[idx] = vec4<f32>(v, v, v, 1.0);
}
"#;

impl Filter for PerlinNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let seed_offset_x = self.seed as f32 * 0.1;
        let seed_offset_y = self.seed as f32 * 0.07;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + seed_offset_x;
                let py = y as f32 / self.scale + seed_offset_y;
                let v = fbm_cpu(px, py, self.octaves, self.persistence);
                let i = ((y * width + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v; out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_u32(&mut p, self.seed);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(PERLIN_NOISE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplex Noise
// ═══════════════════════════════════════════════════════════════════════════

/// Generate simplex noise pattern (faster variant of Perlin noise).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "simplex_noise", category = "generator")]
pub struct SimplexNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 50.0)] pub scale: f32,
    #[param(min = 1, max = 8, step = 1, default = 4)] pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub persistence: f32,
    #[param(min = 0, max = 99999, step = 1, default = 7, hint = "rc.seed")] pub seed: u32,
}

impl Filter for SimplexNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Reuse Perlin FBM with different seed offset for visual distinction
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let seed_offset_x = self.seed as f32 * 0.13 + 100.0;
        let seed_offset_y = self.seed as f32 * 0.09 + 200.0;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + seed_offset_x;
                let py = y as f32 / self.scale + seed_offset_y;
                let v = fbm_cpu(px, py, self.octaves, self.persistence);
                let i = ((y * width + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v; out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        // Reuse Perlin shader with offset seed
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_u32(&mut p, self.seed.wrapping_add(10000));
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(PERLIN_NOISE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Plasma
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a colorful plasma pattern.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "plasma", category = "generator")]
pub struct Plasma {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 30.0)] pub scale: f32,
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 0.0)] pub time: f32,
}

const PLASMA_WGSL: &str = r#"
const PI: f32 = 3.14159265358979;
struct Params { width: u32, height: u32, scale: f32, time: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / params.scale;
  let y = f32(idx / params.width) / params.scale;
  let t = params.time;
  let v1 = sin(x + t);
  let v2 = sin(y + t * 0.7);
  let v3 = sin(x + y + t * 0.5);
  let v4 = sin(sqrt(x * x + y * y) + t * 0.3);
  let v = (v1 + v2 + v3 + v4) * 0.25;
  let r = sin(v * PI) * 0.5 + 0.5;
  let g = sin(v * PI + 2.094) * 0.5 + 0.5;
  let b = sin(v * PI + 4.189) * 0.5 + 0.5;
  output[idx] = vec4<f32>(r, g, b, 1.0);
}
"#;

impl Filter for Plasma {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let t = self.time;
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32 / self.scale;
                let yf = y as f32 / self.scale;
                let v1 = (xf + t).sin();
                let v2 = (yf + t * 0.7).sin();
                let v3 = (xf + yf + t * 0.5).sin();
                let v4 = ((xf * xf + yf * yf).sqrt() + t * 0.3).sin();
                let v = (v1 + v2 + v3 + v4) * 0.25;
                let i = ((y * width + x) * 4) as usize;
                out[i] = (v * PI).sin() * 0.5 + 0.5;
                out[i+1] = (v * PI + 2.094).sin() * 0.5 + 0.5;
                out[i+2] = (v * PI + 4.189).sin() * 0.5 + 0.5;
                out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.scale); gpu_push_f32(&mut p, self.time);
        Some(vec![GpuShader::new(PLASMA_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ─── CPU noise helpers ─────────────────────────────────────────────────────

fn hash2_cpu(x: f32, y: f32) -> f32 {
    let kx = 0.3183099f32;
    let ky = 0.3678794f32;
    let px = x * kx + ky;
    let py = y * ky + kx;
    (16.0 * kx * (px * py * (px + py)).fract()).fract()
}

fn noise2_cpu(px: f32, py: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let fx = px - ix;
    let fy = py - iy;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2_cpu(ix, iy);
    let b = hash2_cpu(ix + 1.0, iy);
    let c = hash2_cpu(ix, iy + 1.0);
    let d = hash2_cpu(ix + 1.0, iy + 1.0);
    let ab = a + ux * (b - a);
    let cd = c + ux * (d - c);
    ab + uy * (cd - ab)
}

fn fbm_cpu(x: f32, y: f32, octaves: u32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += noise2_cpu(x * frequency, y * frequency) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    value / total
}

// ═══════════════════════════════════════════════════════════════════════════
// Solid Color
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a solid color fill — useful as mask, background, or pipeline building block.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solid_color", category = "generator")]
pub struct SolidColor {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub a: f32,
}

const SOLID_COLOR_WGSL: &str = r#"
struct Params { width: u32, height: u32, r: f32, g: f32, b: f32, a: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  output[idx] = vec4<f32>(params.r, params.g, params.b, params.a);
}
"#;

impl Filter for SolidColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let n = (width * height) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            out.extend_from_slice(&[self.r, self.g, self.b, self.a]);
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.r); gpu_push_f32(&mut p, self.g);
        gpu_push_f32(&mut p, self.b); gpu_push_f32(&mut p, self.a);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(SOLID_COLOR_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fractal Noise (fBm with lacunarity)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate fractal Brownian motion noise with lacunarity and persistence controls.
/// More control than perlin_noise — lacunarity sets frequency multiplier per octave.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "fractal_noise", category = "generator")]
pub struct FractalNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 50.0)] pub scale: f32,
    #[param(min = 1, max = 10, step = 1, default = 6)] pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub persistence: f32,
    #[param(min = 1.0, max = 4.0, step = 0.1, default = 2.0)] pub lacunarity: f32,
    #[param(min = 0, max = 99999, step = 1, default = 42, hint = "rc.seed")] pub seed: u32,
}

const FRACTAL_NOISE_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, octaves: u32, persistence: f32, lacunarity: f32, seed: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let k = vec2<f32>(0.3183099, 0.3678794);
  let pp = p * k + k.yx;
  return fract(16.0 * k.x * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i + vec2<f32>(0.0, 0.0)), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let base = vec2<f32>(x, y) / params.scale + vec2<f32>(f32(params.seed) * 0.1, f32(params.seed) * 0.07);
  var value = 0.0; var amplitude = 1.0; var frequency = 1.0; var total_amp = 0.0;
  for (var i = 0u; i < params.octaves; i++) {
    value += noise2(base * frequency) * amplitude;
    total_amp += amplitude;
    amplitude *= params.persistence;
    frequency *= params.lacunarity;
  }
  let v = value / total_amp;
  output[idx] = vec4<f32>(v, v, v, 1.0);
}
"#;

impl Filter for FractalNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let sx = self.seed as f32 * 0.1;
        let sy = self.seed as f32 * 0.07;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + sx;
                let py = y as f32 / self.scale + sy;
                let v = fbm_lacunarity_cpu(px, py, self.octaves, self.persistence, self.lacunarity);
                let i = ((y * width + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v; out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_f32(&mut p, self.lacunarity);
        gpu_push_u32(&mut p, self.seed);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(FRACTAL_NOISE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cloud Noise (Worley + value noise blend)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate cloud-like noise by blending Worley (cellular) noise with value noise.
/// The worley_blend parameter controls the mix: 0 = pure value, 1 = pure Worley.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "cloud_noise", category = "generator")]
pub struct CloudNoise {
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 60.0)] pub scale: f32,
    #[param(min = 1, max = 8, step = 1, default = 5)] pub octaves: u32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub persistence: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.4)] pub worley_blend: f32,
    #[param(min = 0, max = 99999, step = 1, default = 42, hint = "rc.seed")] pub seed: u32,
}

const CLOUD_NOISE_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, octaves: u32, persistence: f32, worley_blend: f32, seed: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let k = vec2<f32>(0.3183099, 0.3678794);
  let pp = p * k + k.yx;
  return fract(16.0 * k.x * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn hash2v(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(hash2(p), hash2(p + vec2<f32>(127.1, 311.7)));
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

fn worley(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  var min_dist = 1.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let neighbor = vec2<f32>(f32(dx), f32(dy));
      let point = hash2v(i + neighbor);
      let diff = neighbor + point - f;
      min_dist = min(min_dist, length(diff));
    }
  }
  return min_dist;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let base = vec2<f32>(x, y) / params.scale + vec2<f32>(f32(params.seed) * 0.1, f32(params.seed) * 0.07);

  var val = 0.0; var wor = 0.0;
  var amp = 1.0; var freq = 1.0; var total = 0.0;
  for (var i = 0u; i < params.octaves; i++) {
    val += noise2(base * freq) * amp;
    wor += (1.0 - worley(base * freq)) * amp;
    total += amp;
    amp *= params.persistence;
    freq *= 2.0;
  }
  val /= total; wor /= total;
  let v = mix(val, wor, params.worley_blend);
  output[idx] = vec4<f32>(v, v, v, 1.0);
}
"#;

impl Filter for CloudNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = input;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        let sx = self.seed as f32 * 0.1;
        let sy = self.seed as f32 * 0.07;
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / self.scale + sx;
                let py = y as f32 / self.scale + sy;
                let val = fbm_cpu(px, py, self.octaves, self.persistence);
                let wor = worley_fbm_cpu(px, py, self.octaves, self.persistence);
                let v = val * (1.0 - self.worley_blend) + wor * self.worley_blend;
                let i = ((y * width + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v; out[i+3] = 1.0;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.scale);
        gpu_push_u32(&mut p, self.octaves);
        gpu_push_f32(&mut p, self.persistence);
        gpu_push_f32(&mut p, self.worley_blend);
        gpu_push_u32(&mut p, self.seed);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(CLOUD_NOISE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern Fill (tile a repeating pattern)
// ═══════════════════════════════════════════════════════════════════════════

/// Tile the input image as a repeating pattern. Useful for creating seamless backgrounds.
/// tile_w/tile_h control the pattern repeat period in pixels.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "pattern_fill", category = "generator")]
pub struct PatternFill {
    #[param(min = 4.0, max = 512.0, step = 1.0, default = 64.0)] pub tile_w: f32,
    #[param(min = 4.0, max = 512.0, step = 1.0, default = 64.0)] pub tile_h: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub offset_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub offset_y: f32,
}

const PATTERN_FILL_WGSL: &str = r#"
struct Params { width: u32, height: u32, tile_w: f32, tile_h: f32, offset_x: f32, offset_y: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height;
  if (idx >= total) { return; }
  let x = f32(idx % params.width);
  let y = f32(idx / params.width);
  let ox = params.offset_x * params.tile_w;
  let oy = params.offset_y * params.tile_h;
  let src_x = u32(((x + ox) % params.tile_w + params.tile_w) % params.tile_w) % params.width;
  let src_y = u32(((y + oy) % params.tile_h + params.tile_h) % params.tile_h) % params.height;
  let src_idx = src_y * params.width + src_x;
  output[idx] = input[src_idx];
}
"#;

impl Filter for PatternFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let tw = self.tile_w;
        let th = self.tile_h;
        let ox = self.offset_x * tw;
        let oy = self.offset_y * th;
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let sx = ((x as f32 + ox) % tw + tw) % tw;
                let sy = ((y as f32 + oy) % th + th) % th;
                let src_x = (sx as u32).min(width - 1);
                let src_y = (sy as u32).min(height - 1);
                let src_i = ((src_y * width + src_x) * 4) as usize;
                let dst_i = ((y * width + x) * 4) as usize;
                out[dst_i..dst_i + 4].copy_from_slice(&input[src_i..src_i + 4]);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, w: u32, h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(w, h);
        gpu_push_f32(&mut p, self.tile_w); gpu_push_f32(&mut p, self.tile_h);
        gpu_push_f32(&mut p, self.offset_x); gpu_push_f32(&mut p, self.offset_y);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(PATTERN_FILL_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ─── CPU noise helpers (Worley + lacunarity) ──────────────────────────────

fn hash2v_cpu(x: f32, y: f32) -> (f32, f32) {
    (hash2_cpu(x, y), hash2_cpu(x + 127.1, y + 311.7))
}

fn worley_cpu(px: f32, py: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let fx = px - ix;
    let fy = py - iy;
    let mut min_dist = 1.0f32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (hx, hy) = hash2v_cpu(ix + dx as f32, iy + dy as f32);
            let diff_x = dx as f32 + hx - fx;
            let diff_y = dy as f32 + hy - fy;
            let d = (diff_x * diff_x + diff_y * diff_y).sqrt();
            min_dist = min_dist.min(d);
        }
    }
    min_dist
}

fn worley_fbm_cpu(x: f32, y: f32, octaves: u32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += (1.0 - worley_cpu(x * frequency, y * frequency)) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    value / total
}

fn fbm_lacunarity_cpu(x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..octaves {
        value += noise2_cpu(x * frequency, y * frequency) * amplitude;
        total += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    value / total
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_generator_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["checkerboard", "gradient_linear", "gradient_radial",
                       "perlin_noise", "simplex_noise", "plasma",
                       "solid_color", "fractal_noise", "cloud_noise", "pattern_fill"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn checkerboard_produces_two_colors() {
        let input = vec![0.0f32; 64 * 64 * 4];
        let f = Checkerboard {
            size: 32.0, color1_r: 1.0, color1_g: 1.0, color1_b: 1.0,
            color2_r: 0.0, color2_g: 0.0, color2_b: 0.0,
        };
        let out = f.compute(&input, 64, 64).unwrap();
        // (0,0) should be color1 (white)
        assert!(out[0] > 0.9);
        // (32,0) should be color2 (black)
        let i = (0 * 64 + 32) * 4;
        assert!(out[i] < 0.1);
    }

    #[test]
    fn plasma_produces_color() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = Plasma { scale: 10.0, time: 0.0 };
        let out = f.compute(&input, 32, 32).unwrap();
        let has_color = out.chunks(4).any(|px| px[0] > 0.01 || px[1] > 0.01 || px[2] > 0.01);
        assert!(has_color, "plasma should produce visible color");
    }

    #[test]
    fn solid_color_fills_constant() {
        let input = vec![0.0f32; 8 * 8 * 4];
        let f = SolidColor { r: 0.3, g: 0.6, b: 0.9, a: 1.0 };
        let out = f.compute(&input, 8, 8).unwrap();
        // Every pixel should be the same
        for px in out.chunks(4) {
            assert!((px[0] - 0.3).abs() < 1e-6);
            assert!((px[1] - 0.6).abs() < 1e-6);
            assert!((px[2] - 0.9).abs() < 1e-6);
            assert!((px[3] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn fractal_noise_deterministic() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = FractalNoise { scale: 20.0, octaves: 4, persistence: 0.5, lacunarity: 2.0, seed: 42 };
        let out1 = f.compute(&input, 32, 32).unwrap();
        let out2 = f.compute(&input, 32, 32).unwrap();
        assert_eq!(out1, out2, "same params should produce identical output");
    }

    #[test]
    fn fractal_noise_lacunarity_differs_from_perlin() {
        let input = vec![0.0f32; 16 * 16 * 4];
        let frac = FractalNoise { scale: 20.0, octaves: 4, persistence: 0.5, lacunarity: 3.0, seed: 42 };
        let perl = PerlinNoise { scale: 20.0, octaves: 4, persistence: 0.5, seed: 42 };
        let out_frac = frac.compute(&input, 16, 16).unwrap();
        let out_perl = perl.compute(&input, 16, 16).unwrap();
        // With lacunarity=3 vs implicit 2, results should differ
        assert_ne!(out_frac, out_perl);
    }

    #[test]
    fn cloud_noise_produces_visible_output() {
        let input = vec![0.0f32; 32 * 32 * 4];
        let f = CloudNoise { scale: 20.0, octaves: 3, persistence: 0.5, worley_blend: 0.4, seed: 7 };
        let out = f.compute(&input, 32, 32).unwrap();
        let has_visible = out.chunks(4).any(|px| px[0] > 0.01);
        assert!(has_visible, "cloud noise should produce visible output");
    }

    #[test]
    fn cloud_noise_deterministic() {
        let input = vec![0.0f32; 16 * 16 * 4];
        let f = CloudNoise { scale: 20.0, octaves: 3, persistence: 0.5, worley_blend: 0.4, seed: 7 };
        let out1 = f.compute(&input, 16, 16).unwrap();
        let out2 = f.compute(&input, 16, 16).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn pattern_fill_tiles_input() {
        // Create a 4x4 image with known values, tile at 2x2
        let mut input = vec![0.0f32; 4 * 4 * 4];
        // Set pixel (0,0) to red
        input[0] = 1.0; input[3] = 1.0;
        // Set pixel (1,0) to green
        input[4+1] = 1.0; input[4+3] = 1.0;
        let f = PatternFill { tile_w: 2.0, tile_h: 2.0, offset_x: 0.0, offset_y: 0.0 };
        let out = f.compute(&input, 4, 4).unwrap();
        // (2,0) should match (0,0) = red
        let i = (0 * 4 + 2) * 4;
        assert!((out[i] - 1.0).abs() < 1e-6, "tiled pixel should be red");
        // (3,0) should match (1,0) = green
        let j = (0 * 4 + 3) * 4;
        assert!((out[j+1] - 1.0).abs() < 1e-6, "tiled pixel should be green");
    }
}
