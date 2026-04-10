//! Film grain simulation — position-dependent, NOT CLUT-compatible.

use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use super::super::helpers::luminance;

/// Film grain simulation — position-dependent, NOT CLUT-compatible.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "film_grain_grading", category = "grading", cost = "O(n)")]
pub struct FilmGrain {
    /// Grain amount (0.0 = none, 1.0 = heavy).
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub amount: f32,
    /// Grain size in pixels (1.0 = fine, 4.0+ = coarse).
    #[param(min = 0.1, max = 10.0, default = 1.0)]
    pub size: f32,
    /// Color grain (true) or monochrome (false).
    #[param(default = false)]
    pub color: bool,
    /// Random seed for deterministic output.
    #[param(min = 0, max = 100, default = 42)]
    pub seed: u32,
}

impl Filter for FilmGrain {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let inv_size = 1.0 / self.size.max(0.1);
        let seed = self.seed as u64 ^ noise::SEED_FILM_GRAIN;
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 4;
                let sx = (x as f32 * inv_size) as u32;
                let sy = (y as f32 * inv_size) as u32;
                let (r, g, b) = (out[idx], out[idx + 1], out[idx + 2]);
                let luma = luminance(r, g, b);
                let intensity = 4.0 * luma * (1.0 - luma) * self.amount;
                if self.color {
                    out[idx] = r + noise::noise_2d(sx, sy, seed) * intensity;
                    out[idx + 1] = g + noise::noise_2d(sx, sy, seed.wrapping_add(1)) * intensity;
                    out[idx + 2] = b + noise::noise_2d(sx, sy, seed.wrapping_add(2)) * intensity;
                } else {
                    let n = noise::noise_2d(sx, sy, seed) * intensity;
                    out[idx] = r + n;
                    out[idx + 1] = g + n;
                    out[idx + 2] = b + n;
                }
            }
        }
        Ok(out)
    }
}

/// WGSL compute shader body for film grain (without noise functions).
///
/// Uses SplitMix64 noise from `noise::NOISE_WGSL` (composed at runtime).
const FILM_GRAIN_WGSL_BODY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  amount: f32,
  inv_size: f32,
  seed_lo: u32,
  seed_hi: u32,
  color_grain: u32,
  _pad: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }
  let idx = y * params.width + x;
  let pixel = load_pixel(idx);
  let luma = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  let intensity = 4.0 * luma * (1.0 - luma) * params.amount;
  let sx = u32(f32(x) * params.inv_size);
  let sy = u32(f32(y) * params.inv_size);
  var result = pixel;
  if (params.color_grain != 0u) {
    result.x = pixel.x + noise_2d(sx, sy, params.seed_lo, params.seed_hi) * intensity;
    result.y = pixel.y + noise_2d(sx, sy, params.seed_lo + 1u, params.seed_hi) * intensity;
    result.z = pixel.z + noise_2d(sx, sy, params.seed_lo + 2u, params.seed_hi) * intensity;
  } else {
    let n = noise_2d(sx, sy, params.seed_lo, params.seed_hi) * intensity;
    result.x = pixel.x + n;
    result.y = pixel.y + n;
    result.z = pixel.z + n;
  }
  store_pixel(idx, result);
}
"#;

/// Compose the full film grain shader: NOISE_WGSL + FILM_GRAIN_WGSL_BODY.
fn film_grain_shader() -> String {
    let mut s = String::with_capacity(noise::NOISE_WGSL.len() + FILM_GRAIN_WGSL_BODY.len() + 1);
    s.push_str(noise::NOISE_WGSL);
    s.push('\n');
    s.push_str(FILM_GRAIN_WGSL_BODY);
    s
}

impl GpuFilter for FilmGrain {
    fn shader_body(&self) -> &str {
        // Return the body portion only — the noise functions are composed
        // via the full shader in gpu_shader() override below.
        FILM_GRAIN_WGSL_BODY
    }

    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }

    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let inv_size = 1.0 / self.size.max(0.1);
        let seed = self.seed as u64 ^ noise::SEED_FILM_GRAIN;
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        let color_grain: u32 = if self.color { 1 } else { 0 };
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.amount.to_le_bytes());
        buf.extend_from_slice(&inv_size.to_le_bytes());
        buf.extend_from_slice(&seed_lo.to_le_bytes());
        buf.extend_from_slice(&seed_hi.to_le_bytes());
        buf.extend_from_slice(&color_grain.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // _pad
        buf
    }

    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: film_grain_shader(),
            entry_point: self.entry_point(),
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: self.extra_buffers(),
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
            setup: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    fn test_pixel(r: f32, g: f32, b: f32) -> Vec<f32> {
        vec![r, g, b, 1.0]
    }

    fn assert_rgb_close(actual: &[f32], expected: (f32, f32, f32), tol: f32, label: &str) {
        assert!(
            (actual[0] - expected.0).abs() < tol
                && (actual[1] - expected.1).abs() < tol
                && (actual[2] - expected.2).abs() < tol,
            "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4})",
            expected.0,
            expected.1,
            expected.2,
            actual[0],
            actual[1],
            actual[2]
        );
    }

    #[test]
    fn film_grain_deterministic() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = FilmGrain {
            amount: 0.3,
            size: 1.0,
            color: false,
            seed: 42,
        };
        let out1 = f.compute(&input, 1, 1).unwrap();
        let out2 = f.compute(&input, 1, 1).unwrap();
        assert_eq!(
            out1, out2,
            "Film grain should be deterministic with same seed"
        );
    }

    #[test]
    fn film_grain_zero_amount_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = FilmGrain {
            amount: 0.0,
            size: 1.0,
            color: false,
            seed: 42,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-5, "grain amount=0 identity");
    }

    #[test]
    fn film_grain_preserves_alpha() {
        let input = vec![0.5, 0.5, 0.5, 0.42];
        let f = FilmGrain {
            amount: 0.5,
            size: 1.0,
            color: true,
            seed: 7,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_eq!(out[3], 0.42, "Grain should preserve alpha");
    }

    #[test]
    fn film_grain_gpu_shader_valid() {
        let shader = film_grain_shader();
        assert!(
            shader.contains("fn splitmix64("),
            "Shader should contain splitmix64"
        );
        assert!(
            shader.contains("fn noise_2d("),
            "Shader should contain noise_2d"
        );
        assert!(
            shader.contains("@compute @workgroup_size(16, 16, 1)"),
            "Shader should have workgroup_size"
        );
        assert!(
            shader.contains("load_pixel"),
            "Shader should use load_pixel"
        );
        assert!(
            shader.contains("store_pixel"),
            "Shader should use store_pixel"
        );
        assert!(
            shader.contains("params.amount"),
            "Shader should reference params.amount"
        );
        assert!(
            shader.contains("params.color_grain"),
            "Shader should reference color_grain flag"
        );
        assert!(
            shader.contains("params.seed_lo"),
            "Shader should use seed_lo/seed_hi"
        );
    }

    #[test]
    fn film_grain_gpu_params_layout() {
        let f = FilmGrain {
            amount: 0.3,
            size: 2.0,
            color: true,
            seed: 99,
        };
        let params = f.params(100, 50);
        assert_eq!(params.len(), 32, "Params should be 8 u32s = 32 bytes");
        let width = u32::from_le_bytes(params[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(params[4..8].try_into().unwrap());
        let amount = f32::from_le_bytes(params[8..12].try_into().unwrap());
        let color_grain = u32::from_le_bytes(params[24..28].try_into().unwrap());
        assert_eq!(width, 100);
        assert_eq!(height, 50);
        assert!((amount - 0.3).abs() < 1e-6);
        assert_eq!(color_grain, 1);
        // seed_lo and seed_hi should be non-zero (XOR'd with SEED_FILM_GRAIN)
        let seed_lo = u32::from_le_bytes(params[16..20].try_into().unwrap());
        let seed_hi = u32::from_le_bytes(params[20..24].try_into().unwrap());
        assert!(
            seed_lo != 0 || seed_hi != 0,
            "Seed should be mixed with offset"
        );
    }
}
