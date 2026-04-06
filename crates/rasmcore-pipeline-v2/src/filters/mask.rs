//! Mask operation filters — alpha channel generation, manipulation, and blending.
//!
//! All filters operate on f32 RGBA. Mask operations primarily affect the alpha
//! channel or use it for blending. GPU shaders are per-pixel (no neighborhood).

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

// ─── GPU helpers ───────────────────────────────────────────────────────────

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}

fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

// ═══════════════════════════════════════════════════════════════════════════
// Color Range — generate mask from color proximity
// ═══════════════════════════════════════════════════════════════════════════

/// Color range keying — pixels within range of target color get alpha=1, others alpha=0.
/// Smooth falloff between threshold and threshold+softness.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "color_range", category = "mask")]
pub struct ColorRange {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_b: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub threshold: f32,
    #[param(min = 0.0, max = 0.5, step = 0.01, default = 0.1)]
    pub softness: f32,
}

const COLOR_RANGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, target_r: f32, target_g: f32, target_b: f32, threshold: f32, softness: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let dr = px.r - params.target_r; let dg = px.g - params.target_g; let db = px.b - params.target_b;
  let dist = sqrt(dr*dr + dg*dg + db*db);
  var alpha: f32;
  if (params.softness > 0.0) {
    alpha = 1.0 - smoothstep(params.threshold, params.threshold + params.softness, dist);
  } else {
    alpha = select(0.0, 1.0, dist <= params.threshold);
  }
  output[idx] = vec4<f32>(px.rgb, alpha);
}
"#;

impl Filter for ColorRange {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let dr = input[o] - self.target_r;
            let dg = input[o + 1] - self.target_g;
            let db = input[o + 2] - self.target_b;
            let dist = (dr * dr + dg * dg + db * db).sqrt();
            let alpha = if self.softness > 0.0 {
                1.0 - smoothstep_f32(self.threshold, self.threshold + self.softness, dist)
            } else {
                if dist <= self.threshold { 1.0 } else { 0.0 }
            };
            out[o + 3] = alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.target_r); gpu_push_f32(&mut p, self.target_g);
        gpu_push_f32(&mut p, self.target_b); gpu_push_f32(&mut p, self.threshold);
        gpu_push_f32(&mut p, self.softness); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(COLOR_RANGE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Luminance Range — generate mask from luminance
// ═══════════════════════════════════════════════════════════════════════════

/// Luminance range keying — pixels within luma range get alpha=1.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "luminance_range", category = "mask")]
pub struct LuminanceRange {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub low: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.8)]
    pub high: f32,
    #[param(min = 0.0, max = 0.5, step = 0.01, default = 0.05)]
    pub softness: f32,
}

const LUMINANCE_RANGE_WGSL: &str = r#"
struct Params { width: u32, height: u32, low: f32, high: f32, softness: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  let lo = smoothstep(params.low - params.softness, params.low + params.softness, luma);
  let hi = 1.0 - smoothstep(params.high - params.softness, params.high + params.softness, luma);
  output[idx] = vec4<f32>(px.rgb, lo * hi);
}
"#;

impl Filter for LuminanceRange {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let luma = input[o] * 0.2126 + input[o + 1] * 0.7152 + input[o + 2] * 0.0722;
            let lo = smoothstep_f32(self.low - self.softness, self.low + self.softness, luma);
            let hi = 1.0 - smoothstep_f32(self.high - self.softness, self.high + self.softness, luma);
            out[o + 3] = lo * hi;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.low); gpu_push_f32(&mut p, self.high);
        gpu_push_f32(&mut p, self.softness); gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(LUMINANCE_RANGE_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Feather — blur the alpha channel
// ═══════════════════════════════════════════════════════════════════════════

/// Feather mask edges — gaussian blur on alpha channel only.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "feather", category = "mask")]
pub struct Feather {
    #[param(min = 1, max = 50, step = 1, default = 3, hint = "rc.pixels")]
    pub radius: u32,
}

const FEATHER_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var sum: f32 = 0.0; var weight: f32 = 0.0;
  let sigma = f32(r) / 2.0;
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1); let sy = clamp(y + dy, 0, h - 1);
      let d2 = f32(dx*dx + dy*dy);
      let w_g = exp(-d2 * inv2s2);
      sum += input[u32(sx) + u32(sy) * params.width].w * w_g;
      weight += w_g;
    }
  }
  let px = input[u32(x) + u32(y) * params.width];
  output[u32(x) + u32(y) * params.width] = vec4<f32>(px.rgb, sum / weight);
}
"#;

impl Filter for Feather {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let r = self.radius as i32;
        let w = width as i32;
        let h = height as i32;
        let sigma = r as f32 / 2.0;
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut weight = 0.0f32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = (x + dx).max(0).min(w - 1) as usize;
                        let sy = (y + dy).max(0).min(h - 1) as usize;
                        let d2 = (dx * dx + dy * dy) as f32;
                        let wg = (-d2 * inv2s2).exp();
                        sum += input[(sy * width as usize + sx) * 4 + 3] * wg;
                        weight += wg;
                    }
                }
                let i = (y as usize * width as usize + x as usize) * 4;
                out[i + 3] = sum / weight;
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, self.radius); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(FEATHER_WGSL.to_string(), "main", [16, 16, 1], p)])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gradient Mask — generate linear gradient in alpha channel
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a linear gradient mask in the alpha channel.
/// Replaces from_path (which requires external path data).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gradient_mask", category = "mask")]
pub struct GradientMask {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub start: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub end: f32,
    #[param(min = 0, max = 1, default = 0)]
    pub vertical: bool,
}

const GRADIENT_MASK_WGSL: &str = r#"
struct Params { width: u32, height: u32, start_val: f32, end_val: f32, vertical: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = idx % params.width; let y = idx / params.width;
  var t: f32;
  if (params.vertical > 0.5) { t = f32(y) / f32(params.height - 1u); }
  else { t = f32(x) / f32(params.width - 1u); }
  let alpha = mix(params.start_val, params.end_val, t);
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, alpha);
}
"#;

impl Filter for GradientMask {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let x = (i % width as usize) as f32;
            let y = (i / width as usize) as f32;
            let t = if self.vertical {
                y / (height - 1).max(1) as f32
            } else {
                x / (width - 1).max(1) as f32
            };
            let alpha = self.start + t * (self.end - self.start);
            out[i * 4 + 3] = alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.start); gpu_push_f32(&mut p, self.end);
        gpu_push_f32(&mut p, if self.vertical { 1.0 } else { 0.0 });
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(GRADIENT_MASK_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mask Apply — multiply RGB by alpha
// ═══════════════════════════════════════════════════════════════════════════

/// Apply mask — multiply RGB channels by alpha (premultiply).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mask_apply", category = "mask")]
pub struct MaskApply;

const MASK_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb * px.w, px.w);
}
"#;

impl Filter for MaskApply {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = out[o + 3];
            out[o] *= a;
            out[o + 1] *= a;
            out[o + 2] *= a;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(MASK_APPLY_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Masked Blend — blend toward a color using alpha as mix factor
// ═══════════════════════════════════════════════════════════════════════════

/// Blend toward a target color using the alpha channel as mix factor.
/// Where alpha=1, output=original. Where alpha=0, output=blend_color.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "masked_blend", category = "mask")]
pub struct MaskedBlend {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub blend_b: f32,
}

const MASKED_BLEND_WGSL: &str = r#"
struct Params { width: u32, height: u32, blend_r: f32, blend_g: f32, blend_b: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let blend = vec3<f32>(params.blend_r, params.blend_g, params.blend_b);
  let rgb = mix(blend, px.rgb, vec3<f32>(px.w));
  output[idx] = vec4<f32>(rgb, 1.0);
}
"#;

impl Filter for MaskedBlend {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = input[o + 3];
            out[o] = self.blend_r + a * (input[o] - self.blend_r);
            out[o + 1] = self.blend_g + a * (input[o + 1] - self.blend_g);
            out[o + 2] = self.blend_b + a * (input[o + 2] - self.blend_b);
            out[o + 3] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_f32(&mut p, self.blend_r); gpu_push_f32(&mut p, self.blend_g);
        gpu_push_f32(&mut p, self.blend_b); gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(MASKED_BLEND_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mask Combine — combine two masks (multiply, add, subtract, min, max)
// ═══════════════════════════════════════════════════════════════════════════

/// Combine the current alpha with a generated alpha (e.g., from luminance).
/// Mode: 0=multiply, 1=add, 2=min, 3=max, 4=replace.
/// The "second mask" is derived from luminance of the RGB channels.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mask_combine", category = "mask")]
pub struct MaskCombine {
    #[param(min = 0, max = 4, step = 1, default = 0)]
    pub mode: u32,
}

const MASK_COMBINE_WGSL: &str = r#"
struct Params { width: u32, height: u32, mode: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let existing = px.w;
  let luma = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
  var result: f32;
  switch (params.mode) {
    case 0u: { result = existing * luma; }
    case 1u: { result = clamp(existing + luma, 0.0, 1.0); }
    case 2u: { result = min(existing, luma); }
    case 3u: { result = max(existing, luma); }
    default: { result = luma; }
  }
  output[idx] = vec4<f32>(px.rgb, result);
}
"#;

impl Filter for MaskCombine {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let existing = input[o + 3];
            let luma = input[o] * 0.2126 + input[o + 1] * 0.7152 + input[o + 2] * 0.0722;
            out[o + 3] = match self.mode {
                0 => existing * luma,
                1 => (existing + luma).min(1.0),
                2 => existing.min(luma),
                3 => existing.max(luma),
                _ => luma,
            };
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(width, height);
        gpu_push_u32(&mut p, self.mode); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(MASK_COMBINE_WGSL.to_string(), "main", [256, 1, 1], p)])
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
    fn all_mask_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["color_range", "luminance_range", "feather",
                       "gradient_mask", "mask_apply", "masked_blend", "mask_combine"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn mask_apply_premultiplies() {
        let input = vec![0.8, 0.6, 0.4, 0.5,  1.0, 1.0, 1.0, 0.0];
        let f = MaskApply;
        let out = f.compute(&input, 2, 1).unwrap();
        assert!((out[0] - 0.4).abs() < 0.001); // 0.8 * 0.5
        assert!((out[4] - 0.0).abs() < 0.001); // 1.0 * 0.0
    }

    #[test]
    fn luminance_range_creates_mask() {
        let input = vec![0.1, 0.1, 0.1, 1.0,  0.9, 0.9, 0.9, 1.0];
        let f = LuminanceRange { low: 0.3, high: 0.7, softness: 0.0 };
        let out = f.compute(&input, 2, 1).unwrap();
        // Dark pixel (luma ~0.1) → outside range → alpha ≈ 0
        assert!(out[3] < 0.1);
        // Bright pixel (luma ~0.9) → outside range → alpha ≈ 0
        assert!(out[7] < 0.1);
    }
}
