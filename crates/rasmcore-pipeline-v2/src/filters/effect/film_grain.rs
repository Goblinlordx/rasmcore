use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::gpu_params_wh;
use noise::Rng;

use super::{gpu_params_push_f32, gpu_params_push_u32, luminance};

/// Film grain — photographic grain overlay with noise texture.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "film_grain", category = "effect", cost = "O(n)")]
pub struct FilmGrain {
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub amount: f32,
    #[param(min = 1.0, max = 10.0, default = 1.0)]
    pub size: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for FilmGrain {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.amount <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let mut rng = Rng::with_offset(self.seed, noise::SEED_FILM_GRAIN);

        // Generate grain at reduced resolution then upsample
        let grain_w = (w as f32 / self.size.max(1.0)).ceil() as usize;
        let grain_h = (h as f32 / self.size.max(1.0)).ceil() as usize;
        let mut grain = vec![0.0f32; grain_w.max(1) * grain_h.max(1)];
        for v in &mut grain {
            *v = rng.next_gaussian() * self.amount;
        }

        let mut out = input.to_vec();
        let inv_size = 1.0 / self.size.max(1.0);

        for y in 0..h {
            for x in 0..w {
                let gx = ((x as f32 * inv_size) as usize).min(grain_w.saturating_sub(1));
                let gy = ((y as f32 * inv_size) as usize).min(grain_h.saturating_sub(1));
                let g = grain[gy * grain_w.max(1) + gx];

                let idx = (y * w + x) * 4;
                // Grain weighted by luminance (more visible in midtones)
                let luma = luminance(out[idx], out[idx + 1], out[idx + 2]).clamp(0.0, 1.0);
                let weight = 4.0 * luma * (1.0 - luma); // midtone peak
                for c in 0..3 {
                    out[idx + c] += g * weight;
                }
            }
        }
        Ok(out)
    }
}

pub(crate) const EFFECT_FILM_GRAIN_WGSL: &str = include_str!("../../shaders/film_grain.wgsl");

impl GpuFilter for FilmGrain {
    fn shader_body(&self) -> &str {
        EFFECT_FILM_GRAIN_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let inv_size = 1.0 / self.size.max(1.0);
        let seed = self.seed ^ noise::SEED_FILM_GRAIN;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.amount);
        gpu_params_push_f32(&mut buf, inv_size);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, EFFECT_FILM_GRAIN_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}
