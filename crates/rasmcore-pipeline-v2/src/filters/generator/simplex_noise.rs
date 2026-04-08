//! SimplexNoise generator filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::{gpu_params_wh, gpu_push_f32, gpu_push_u32};
use super::fbm_cpu;
use super::perlin_noise::PERLIN_NOISE_WGSL;

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
