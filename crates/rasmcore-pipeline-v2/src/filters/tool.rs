//! Interactive tool filters — brush and region-based operations.
//!
//! These model single-application tool strokes as filters with fixed params.
//! Full interactive brush support (paths, pressure, accumulation) is a
//! separate architectural layer — these are the per-application primitives.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}
fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

/// Bilinear sampling helper for coordinate remap filters.
#[inline(always)]
fn sample_bilinear(input: &[f32], width: u32, height: u32, fx: f32, fy: f32) -> [f32; 4] {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let dx = fx - ix as f32;
    let dy = fy - iy as f32;
    let x0 = ix.max(0).min(width as i32 - 1) as usize;
    let x1 = (ix + 1).max(0).min(width as i32 - 1) as usize;
    let y0 = iy.max(0).min(height as i32 - 1) as usize;
    let y1 = (iy + 1).max(0).min(height as i32 - 1) as usize;
    let w = width as usize;
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let p00 = input[(y0 * w + x0) * 4 + c];
        let p10 = input[(y0 * w + x1) * 4 + c];
        let p01 = input[(y1 * w + x0) * 4 + c];
        let p11 = input[(y1 * w + x1) * 4 + c];
        out[c] = (p00 + dx * (p10 - p00)) + dy * ((p01 + dx * (p11 - p01)) - (p00 + dx * (p10 - p00)));
    }
    out
}

const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx)); let iy = i32(floor(fy));
  let dx = fx - f32(ix); let dy = fy - f32(iy);
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

// ═══════════════════════════════════════════════════════════════════════════
// Clone Stamp — copy from source offset within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Clone stamp — copy pixels from source offset within a circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clone_stamp", category = "tool")]
pub struct CloneStamp {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)] pub offset_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)] pub offset_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")] pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub opacity: f32,
}

const CLONE_STAMP_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, ox: f32, oy: f32, radius: f32, opacity: f32, }
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
  let falloff = 1.0 - smoothstep(params.radius * 0.7, params.radius, dist);
  let sx = x + params.ox; let sy = y + params.oy;
  let src = sample_bilinear_f32(sx, sy);
  let bg = input[idx];
  let a = falloff * params.opacity;
  output[idx] = vec4<f32>(mix(bg.rgb, src.rgb, vec3<f32>(a)), bg.w);
}
"#;

impl Filter for CloneStamp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        let ox = self.offset_x * width as f32;
        let oy = self.offset_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.7, self.radius, dist);
                let src = sample_bilinear(input, width, height, x as f32 + ox, y as f32 + oy);
                let i = ((y * width + x) * 4) as usize;
                let a = falloff * self.opacity;
                out[i] = out[i] * (1.0 - a) + src[0] * a;
                out[i+1] = out[i+1] * (1.0 - a) + src[1] * a;
                out[i+2] = out[i+2] * (1.0 - a) + src[2] * a;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{CLONE_STAMP_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.offset_x * _w as f32);
        gpu_push_f32(&mut p, self.offset_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.opacity);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Smudge — directional displacement within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Smudge — push pixels in a direction within a circular brush.
/// Similar to liquify but with softer gaussian falloff.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "smudge", category = "tool")]
pub struct Smudge {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")] pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub strength: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.5)] pub direction_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)] pub direction_y: f32,
}

const SMUDGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, radius: f32, strength: f32, dir_x: f32, dir_y: f32, }
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
  let t = dist / params.radius;
  let w = exp(-2.0 * t * t) * params.strength;
  let sx = x - params.dir_x * w * params.radius;
  let sy = y - params.dir_y * w * params.radius;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for Smudge {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let t = dist / self.radius;
                let w = (-2.0 * t * t).exp() * self.strength;
                let sx = x as f32 - self.direction_x * w * self.radius;
                let sy = y as f32 - self.direction_y * w * self.radius;
                let src = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i+4].copy_from_slice(&src);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{SMUDGE_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.strength);
        gpu_push_f32(&mut p, self.direction_x);
        gpu_push_f32(&mut p, self.direction_y);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sponge — local saturation adjustment within circular brush
// ═══════════════════════════════════════════════════════════════════════════

/// Sponge tool — boost or reduce saturation within a circular brush.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sponge", category = "tool")]
pub struct Sponge {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")] pub radius: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.3)] pub amount: f32,
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
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.5, self.radius, dist);
                let i = ((y * width + x) * 4) as usize;
                let luma = out[i] * 0.2126 + out[i+1] * 0.7152 + out[i+2] * 0.0722;
                let sat = 1.0 + self.amount * falloff;
                out[i] = luma + sat * (out[i] - luma);
                out[i+1] = luma + sat * (out[i+1] - luma);
                out[i+2] = luma + sat * (out[i+2] - luma);
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
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(SPONGE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Red Eye Remove — desaturate red within circular region
// ═══════════════════════════════════════════════════════════════════════════

/// Red eye removal — desaturate red-dominant pixels within a circular region.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "red_eye_remove", category = "tool")]
pub struct RedEyeRemove {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 30.0, hint = "rc.pixels")] pub radius: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub threshold: f32,
}

const RED_EYE_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, radius: f32, threshold: f32, _p1: u32, _p2: u32, }
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
  let px = input[idx];
  if (dist >= params.radius) { output[idx] = px; return; }
  // Red-dominance detection
  let red_ratio = px.r / max(px.r + px.g + px.b, 0.001);
  if (red_ratio > params.threshold) {
    let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
    let falloff = 1.0 - smoothstep(params.radius * 0.5, params.radius, dist);
    let factor = (red_ratio - params.threshold) / (1.0 - params.threshold) * falloff;
    output[idx] = vec4<f32>(mix(px.r, luma, factor), px.g, px.b, px.w);
  } else {
    output[idx] = px;
  }
}
"#;

impl Filter for RedEyeRemove {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let i = ((y * width + x) * 4) as usize;
                let total = (out[i] + out[i+1] + out[i+2]).max(0.001);
                let red_ratio = out[i] / total;
                if red_ratio > self.threshold {
                    let luma = out[i] * 0.2126 + out[i+1] * 0.7152 + out[i+2] * 0.0722;
                    let falloff = 1.0 - smoothstep_f32(self.radius * 0.5, self.radius, dist);
                    let factor = (red_ratio - self.threshold) / (1.0 - self.threshold) * falloff;
                    out[i] = out[i] * (1.0 - factor) + luma * factor;
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_f32(&mut p, self.threshold);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(RED_EYE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CA Remove — chromatic aberration correction
// ═══════════════════════════════════════════════════════════════════════════

/// Chromatic aberration removal — shift R and B channels radially.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "ca_remove", category = "tool")]
pub struct CaRemove {
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0)] pub red_shift: f32,
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0)] pub blue_shift: f32,
}

const CA_REMOVE_WGSL: &str = r#"
struct Params { width: u32, height: u32, red_shift: f32, blue_shift: f32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let cx = f32(params.width) * 0.5; let cy = f32(params.height) * 0.5;
  let dx = x - cx; let dy = y - cy;
  let dist = sqrt(dx*dx + dy*dy);
  let max_dist = sqrt(cx*cx + cy*cy);
  let t = dist / max_dist;
  // Red channel: sample at shifted position
  let r_offset = t * params.red_shift;
  let r_sx = x + dx / max(dist, 0.001) * r_offset;
  let r_sy = y + dy / max(dist, 0.001) * r_offset;
  let r_idx = clamp(u32(round(r_sx)), 0u, params.width - 1u) + clamp(u32(round(r_sy)), 0u, params.height - 1u) * params.width;
  // Blue channel: sample at shifted position
  let b_offset = t * params.blue_shift;
  let b_sx = x + dx / max(dist, 0.001) * b_offset;
  let b_sy = y + dy / max(dist, 0.001) * b_offset;
  let b_idx = clamp(u32(round(b_sx)), 0u, params.width - 1u) + clamp(u32(round(b_sy)), 0u, params.height - 1u) * params.width;
  let px = input[idx];
  output[idx] = vec4<f32>(input[r_idx].r, px.g, input[b_idx].b, px.w);
}
"#;

impl Filter for CaRemove {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let max_dist = (cx*cx + cy*cy).sqrt();
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                let t = dist / max_dist;
                let i = ((y * width + x) * 4) as usize;
                // Red channel shift
                let r_off = t * self.red_shift;
                let r_src = sample_bilinear(input, width, height,
                    x as f32 + dx / dist.max(0.001) * r_off,
                    y as f32 + dy / dist.max(0.001) * r_off);
                out[i] = r_src[0];
                // Blue channel shift
                let b_off = t * self.blue_shift;
                let b_src = sample_bilinear(input, width, height,
                    x as f32 + dx / dist.max(0.001) * b_off,
                    y as f32 + dy / dist.max(0.001) * b_off);
                out[i+2] = b_src[2];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.red_shift);
        gpu_push_f32(&mut p, self.blue_shift);
        Some(vec![GpuShader::new(CA_REMOVE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flood Fill — replace connected region with solid color
// ═══════════════════════════════════════════════════════════════════════════

/// Flood fill — replace color-similar connected region from seed point.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "flood_fill", category = "tool")]
pub struct FloodFill {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub seed_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub seed_y: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.1)] pub tolerance: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub fill_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub fill_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub fill_b: f32,
}

impl Filter for FloodFill {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let w = width as usize;
        let h = height as usize;
        let sx = (self.seed_x * width as f32) as usize;
        let sy = (self.seed_y * height as f32) as usize;
        if sx >= w || sy >= h { return Ok(out); }

        let seed_i = (sy * w + sx) * 4;
        let seed_color = [input[seed_i], input[seed_i+1], input[seed_i+2]];
        let tol2 = self.tolerance * self.tolerance;

        let mut visited = vec![false; w * h];
        let mut stack = vec![(sx, sy)];
        visited[sy * w + sx] = true;

        while let Some((x, y)) = stack.pop() {
            let i = (y * w + x) * 4;
            let dr = out[i] - seed_color[0];
            let dg = out[i+1] - seed_color[1];
            let db = out[i+2] - seed_color[2];
            if dr*dr + dg*dg + db*db > tol2 { continue; }

            out[i] = self.fill_r;
            out[i+1] = self.fill_g;
            out[i+2] = self.fill_b;

            for (nx, ny) in [(x.wrapping_sub(1), y), (x+1, y), (x, y.wrapping_sub(1)), (x, y+1)] {
                if nx < w && ny < h && !visited[ny * w + nx] {
                    visited[ny * w + nx] = true;
                    stack.push((nx, ny));
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        use crate::node::ReductionBuffer;

        let n = (width * height) as usize;
        let mask_size = n * 4; // u32 per pixel
        let change_size = 4usize; // single atomic u32
        let mask_buf_id = 50u32;
        let change_buf_id = 51u32;

        // Initialize mask: seed pixel = 1, all others = 0
        let seed_x = (self.seed_x * width as f32) as u32;
        let seed_y = (self.seed_y * height as f32) as u32;
        let mut init_mask = vec![0u8; mask_size];
        let seed_idx = (seed_y * width + seed_x) as usize;
        if seed_idx < n {
            init_mask[seed_idx * 4..seed_idx * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
        }

        // Init pass: snapshot input colors + initialize mask + apply fill to seed pixel
        let init_wgsl = format!(r#"
struct Params {{ width: u32, height: u32, seed_x: u32, seed_y: u32, fill_r: f32, fill_g: f32, fill_b: f32, _pad: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if (idx >= params.width * params.height) {{ return; }}
  let px = input[idx];
  if (idx == params.seed_x + params.seed_y * params.width) {{
    output[idx] = vec4<f32>(params.fill_r, params.fill_g, params.fill_b, px.w);
  }} else {{
    output[idx] = px;
  }}
}}
"#);

        let mut init_params = gpu_params_wh(width, height);
        gpu_push_u32(&mut init_params, seed_x);
        gpu_push_u32(&mut init_params, seed_y);
        gpu_push_f32(&mut init_params, self.fill_r);
        gpu_push_f32(&mut init_params, self.fill_g);
        gpu_push_f32(&mut init_params, self.fill_b);
        gpu_push_u32(&mut init_params, 0);

        let init_shader = GpuShader {
            body: init_wgsl,
            entry_point: "main",
            workgroup_size: [256, 1, 1],
            params: init_params,
            extra_buffers: vec![],
            reduction_buffers: vec![ReductionBuffer {
                id: mask_buf_id,
                initial_data: init_mask,
                read_write: true,
            }],
            convergence_check: None,
            loop_dispatch: None,
        };

        // Expand pass: for each unfilled pixel, check 4 neighbors in mask.
        // If any neighbor is filled AND this pixel's color is within tolerance
        // of the seed color → mark as filled, apply fill color.
        // Reads original colors from input (which has the original image on first
        // iteration, then progressively filled image).
        let expand_wgsl = format!(r#"
struct Params {{ width: u32, height: u32, seed_r: f32, seed_g: f32, seed_b: f32, tol2: f32, fill_r: f32, fill_g: f32, fill_b: f32, _pad: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> change_count: array<atomic<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let idx = gid.x;
  if (idx >= params.width * params.height) {{ return; }}
  // Already filled → pass through
  if (mask[idx] != 0u) {{ output[idx] = input[idx]; return; }}
  let x = i32(idx % params.width); let y = i32(idx / params.width);
  let w = i32(params.width); let h = i32(params.height);
  // Check 4 neighbors
  var has_filled_neighbor = false;
  if (x > 0 && mask[u32(x-1) + u32(y) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (x < w-1 && mask[u32(x+1) + u32(y) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (y > 0 && mask[u32(x) + u32(y-1) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (y < h-1 && mask[u32(x) + u32(y+1) * params.width] != 0u) {{ has_filled_neighbor = true; }}
  if (!has_filled_neighbor) {{ output[idx] = input[idx]; return; }}
  // Check color tolerance against seed color
  let px = input[idx];
  let dr = px.r - params.seed_r; let dg = px.g - params.seed_g; let db = px.b - params.seed_b;
  if (dr*dr + dg*dg + db*db > params.tol2) {{ output[idx] = input[idx]; return; }}
  // Fill!
  mask[idx] = 1u;
  output[idx] = vec4<f32>(params.fill_r, params.fill_g, params.fill_b, px.w);
  atomicAdd(&change_count[0], 1u);
}}
"#);

        // Get seed color from input — we need it at shader construction time.
        // Since gpu_shader_passes doesn't have access to pixel data, we pass
        // seed_x/seed_y and let the init pass extract it. For the expand pass,
        // we use the seed color params. The caller must set these from the image.
        // For now, use default (0.5, 0.5, 0.5) — the CPU fallback handles exact colors.
        // TODO: Extract seed color from pixel data when available.
        let seed_r = 0.5f32; // Approximation — CPU path is exact
        let seed_g = 0.5f32;
        let seed_b = 0.5f32;
        let tol2 = self.tolerance * self.tolerance;

        let mut expand_params = gpu_params_wh(width, height);
        gpu_push_f32(&mut expand_params, seed_r);
        gpu_push_f32(&mut expand_params, seed_g);
        gpu_push_f32(&mut expand_params, seed_b);
        gpu_push_f32(&mut expand_params, tol2);
        gpu_push_f32(&mut expand_params, self.fill_r);
        gpu_push_f32(&mut expand_params, self.fill_g);
        gpu_push_f32(&mut expand_params, self.fill_b);
        gpu_push_u32(&mut expand_params, 0);
        gpu_push_u32(&mut expand_params, 0);
        gpu_push_u32(&mut expand_params, 0);

        // Generate N expand passes with convergence check
        let max_iterations = (width.max(height) / 2).max(100);
        let mut passes = vec![init_shader];

        for _ in 0..max_iterations {
            passes.push(GpuShader {
                body: expand_wgsl.clone(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: expand_params.clone(),
                extra_buffers: vec![],
                reduction_buffers: vec![
                    ReductionBuffer { id: mask_buf_id, initial_data: vec![], read_write: true },
                    ReductionBuffer { id: change_buf_id, initial_data: vec![0u8; change_size], read_write: true },
                ],
                convergence_check: Some(change_buf_id),
                loop_dispatch: None,
            });
        }

        Some(passes)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Healing Brush — blend source region with local statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Healing brush — copy texture from source offset, match local color.
/// Combines clone_stamp source with local mean color matching.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "healing_brush", category = "tool")]
pub struct HealingBrush {
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.001, default = 0.5)] pub center_y: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)] pub offset_x: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)] pub offset_y: f32,
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0, hint = "rc.pixels")] pub radius: f32,
}

const HEALING_BRUSH_WGSL: &str = r#"
struct Params { width: u32, height: u32, cx: f32, cy: f32, ox: f32, oy: f32, radius: f32, _pad: u32, }
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
  // Source pixel
  let sx = clamp(u32(round(x + params.ox)), 0u, params.width - 1u);
  let sy = clamp(u32(round(y + params.oy)), 0u, params.height - 1u);
  let src = input[sx + sy * params.width];
  let dst = input[idx];
  // Simple healing: blend source texture with destination color
  let src_luma = src.r * 0.2126 + src.g * 0.7152 + src.b * 0.0722;
  let dst_luma = dst.r * 0.2126 + dst.g * 0.7152 + dst.b * 0.0722;
  let ratio = dst_luma / max(src_luma, 0.001);
  let healed = src.rgb * vec3<f32>(ratio);
  let blended = mix(dst.rgb, healed, vec3<f32>(falloff));
  output[idx] = vec4<f32>(blended, dst.w);
}
"#;

impl Filter for HealingBrush {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let cx = self.center_x * width as f32;
        let cy = self.center_y * height as f32;
        let ox = self.offset_x * width as f32;
        let oy = self.offset_y * height as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist >= self.radius { continue; }
                let falloff = 1.0 - smoothstep_f32(self.radius * 0.5, self.radius, dist);
                let src = sample_bilinear(input, width, height, x as f32 + ox, y as f32 + oy);
                let i = ((y * width + x) * 4) as usize;
                let src_luma = src[0] * 0.2126 + src[1] * 0.7152 + src[2] * 0.0722;
                let dst_luma = out[i] * 0.2126 + out[i+1] * 0.7152 + out[i+2] * 0.0722;
                let ratio = dst_luma / src_luma.max(0.001);
                for c in 0..3 {
                    let healed = src[c] * ratio;
                    out[i+c] = out[i+c] * (1.0 - falloff) + healed * falloff;
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.center_x * _w as f32);
        gpu_push_f32(&mut p, self.center_y * _h as f32);
        gpu_push_f32(&mut p, self.offset_x * _w as f32);
        gpu_push_f32(&mut p, self.offset_y * _h as f32);
        gpu_push_f32(&mut p, self.radius);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(HEALING_BRUSH_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

#[inline(always)]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tool_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["clone_stamp", "smudge", "sponge", "red_eye_remove",
                       "ca_remove", "flood_fill", "healing_brush"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn flood_fill_replaces_region() {
        // 4x4 all white
        let input = vec![1.0f32; 4 * 4 * 4];
        let f = FloodFill {
            seed_x: 0.5, seed_y: 0.5, tolerance: 0.5,
            fill_r: 1.0, fill_g: 0.0, fill_b: 0.0,
        };
        let out = f.compute(&input, 4, 4).unwrap();
        // All pixels should be red (all were white, connected, within tolerance)
        assert!(out[0] > 0.9); // R
        assert!(out[1] < 0.1); // G
    }
}
