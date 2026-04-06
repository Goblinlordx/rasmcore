//! Alpha channel and color matching filters — per-pixel operations.
//!
//! add_alpha, remove_alpha, flatten (composite over background),
//! match_color (histogram-based color transfer).

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}
fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

// ═══════════════════════════════════════════════════════════════════════════
// Add Alpha — set alpha channel to a constant value
// ═══════════════════════════════════════════════════════════════════════════

/// Set the alpha channel to a constant value.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "add_alpha", category = "alpha")]
pub struct AddAlpha {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub alpha: f32,
}

const ADD_ALPHA_WGSL: &str = r#"
struct Params { width: u32, height: u32, alpha: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, params.alpha);
}
"#;

impl Filter for AddAlpha {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for i in (3..out.len()).step_by(4) {
            out[i] = self.alpha;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.alpha);
        gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(ADD_ALPHA_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Remove Alpha — set alpha to 1.0 (fully opaque)
// ═══════════════════════════════════════════════════════════════════════════

/// Remove alpha — set all pixels to fully opaque (alpha = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "remove_alpha", category = "alpha")]
pub struct RemoveAlpha;

const REMOVE_ALPHA_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  output[idx] = vec4<f32>(px.rgb, 1.0);
}
"#;

impl Filter for RemoveAlpha {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for i in (3..out.len()).step_by(4) {
            out[i] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(REMOVE_ALPHA_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — composite over a solid background color
// ═══════════════════════════════════════════════════════════════════════════

/// Flatten — composite the image over a solid background color using alpha.
/// Result is fully opaque (alpha = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "flatten", category = "alpha")]
pub struct Flatten {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub bg_b: f32,
}

const FLATTEN_WGSL: &str = r#"
struct Params { width: u32, height: u32, bg_r: f32, bg_g: f32, bg_b: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let px = input[idx];
  let bg = vec3<f32>(params.bg_r, params.bg_g, params.bg_b);
  let rgb = mix(bg, px.rgb, vec3<f32>(px.w));
  output[idx] = vec4<f32>(rgb, 1.0);
}
"#;

impl Filter for Flatten {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let n = (width * height) as usize;
        for i in 0..n {
            let o = i * 4;
            let a = input[o + 3];
            out[o] = self.bg_r + a * (input[o] - self.bg_r);
            out[o + 1] = self.bg_g + a * (input[o + 1] - self.bg_g);
            out[o + 2] = self.bg_b + a * (input[o + 2] - self.bg_b);
            out[o + 3] = 1.0;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _width: u32, _height: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_width, _height);
        gpu_push_f32(&mut p, self.bg_r); gpu_push_f32(&mut p, self.bg_g);
        gpu_push_f32(&mut p, self.bg_b); gpu_push_u32(&mut p, 0);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(FLATTEN_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Match Color — Reinhard color transfer
// ═══════════════════════════════════════════════════════════════════════════

/// Reinhard color transfer — shift mean/std of each channel toward target values.
/// Simple per-channel normalization: out = (in - mean_in) * (std_target / std_in) + mean_target.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "match_color", category = "color")]
pub struct MatchColor {
    /// Target mean for R channel.
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_mean_r: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_mean_g: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub target_mean_b: f32,
    /// Target standard deviation.
    #[param(min = 0.01, max = 0.5, step = 0.01, default = 0.15)]
    pub target_std: f32,
    /// Blend strength (0 = no change, 1 = full transfer).
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub strength: f32,
}

impl Filter for MatchColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let n = (width * height) as usize;
        if n == 0 { return Ok(input.to_vec()); }

        // Compute mean and std per channel
        let mut sum = [0.0f64; 3];
        for i in 0..n {
            let o = i * 4;
            sum[0] += input[o] as f64;
            sum[1] += input[o + 1] as f64;
            sum[2] += input[o + 2] as f64;
        }
        let mean = [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64];

        let mut var = [0.0f64; 3];
        for i in 0..n {
            let o = i * 4;
            let dr = input[o] as f64 - mean[0];
            let dg = input[o + 1] as f64 - mean[1];
            let db = input[o + 2] as f64 - mean[2];
            var[0] += dr * dr;
            var[1] += dg * dg;
            var[2] += db * db;
        }
        let std_in = [
            (var[0] / n as f64).sqrt().max(0.001),
            (var[1] / n as f64).sqrt().max(0.001),
            (var[2] / n as f64).sqrt().max(0.001),
        ];

        let target_mean = [self.target_mean_r as f64, self.target_mean_g as f64, self.target_mean_b as f64];
        let target_std = self.target_std as f64;
        let strength = self.strength;

        let mut out = input.to_vec();
        for i in 0..n {
            let o = i * 4;
            for c in 0..3 {
                let v = input[o + c] as f64;
                let transferred = (v - mean[c]) * (target_std / std_in[c]) + target_mean[c];
                out[o + c] = (input[o + c] + strength * (transferred as f32 - input[o + c])).max(0.0).min(1.0);
            }
        }
        Ok(out)
    }

    // GPU: requires 2-pass (histogram stats → apply). CPU-only for now.
    // The stats pass would use reduction buffers for sum/sum_sq.
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_alpha_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["add_alpha", "remove_alpha", "flatten", "match_color"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn add_alpha_sets_constant() {
        let input = vec![0.5, 0.5, 0.5, 0.0, 0.8, 0.8, 0.8, 0.3];
        let f = AddAlpha { alpha: 0.7 };
        let out = f.compute(&input, 2, 1).unwrap();
        assert!((out[3] - 0.7).abs() < 0.001);
        assert!((out[7] - 0.7).abs() < 0.001);
    }

    #[test]
    fn remove_alpha_makes_opaque() {
        let input = vec![0.5, 0.5, 0.5, 0.3];
        let f = RemoveAlpha;
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn flatten_composites_over_white() {
        let input = vec![1.0, 0.0, 0.0, 0.5]; // red at 50% alpha
        let f = Flatten { bg_r: 1.0, bg_g: 1.0, bg_b: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // mix(white, red, 0.5) = (1.0, 0.5, 0.5)
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 0.5).abs() < 0.01);
        assert!((out[2] - 0.5).abs() < 0.01);
        assert!((out[3] - 1.0).abs() < 0.01); // fully opaque
    }
}
