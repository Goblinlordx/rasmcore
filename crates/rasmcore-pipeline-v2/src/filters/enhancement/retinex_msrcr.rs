use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// Multi-Scale Retinex with Color Restoration (Jobson et al. 1997).
///
/// MSR + chromaticity-based gain for color preservation.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "retinex_msrcr", category = "enhancement", cost = "O(3 * n * sigma) via gaussian_blur")]
pub struct RetinexMsrcr {
    #[param(min = 0.0, max = 200.0, default = 15.0)]
    pub sigma_small: f32,
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma_medium: f32,
    #[param(min = 0.0, max = 500.0, default = 250.0)]
    pub sigma_large: f32,
    #[param(min = 0.0, max = 200.0, default = 125.0)]
    pub alpha: f32,
    #[param(min = 0.0, max = 100.0, default = 46.0)]
    pub beta: f32,
}

impl Filter for RetinexMsrcr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let scales = [self.sigma_small, self.sigma_medium, self.sigma_large];
        let n = input.len();
        let mut msr = vec![0.0f32; n];

        // MSR computation
        for sigma in &scales {
            let blur = GaussianBlur { radius: *sigma };
            let blurred = blur.compute(input, width, height)?;
            for (i, (inp, blr)) in input.iter().zip(blurred.iter()).enumerate() {
                if i % 4 == 3 {
                    continue;
                }
                msr[i] += inp.max(1e-10).ln() - blr.max(1e-10).ln();
            }
        }

        let inv_scales = 1.0 / scales.len() as f32;
        for v in msr.iter_mut() {
            *v *= inv_scales;
        }

        // Color restoration
        let mut out = vec![0.0f32; n];
        for (pixel_idx, (in_pixel, msr_pixel)) in input
            .chunks_exact(4)
            .zip(msr.chunks_exact(4))
            .enumerate()
        {
            let sum = in_pixel[0] + in_pixel[1] + in_pixel[2];
            let idx = pixel_idx * 4;
            for c in 0..3 {
                let chromaticity = if sum > 1e-10 {
                    in_pixel[c] / sum
                } else {
                    1.0 / 3.0
                };
                let color_gain = self.beta * (self.alpha * chromaticity).ln().max(-10.0);
                out[idx + c] = color_gain * msr_pixel[c];
            }
            out[idx + 3] = in_pixel[3];
        }

        // Normalize to [0, 1]
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];
        for pixel in out.chunks_exact(4) {
            for c in 0..3 {
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }
        for pixel in out.chunks_exact_mut(4) {
            for c in 0..3 {
                let range = max_val[c] - min_val[c];
                if range > 1e-10 {
                    pixel[c] = (pixel[c] - min_val[c]) / range;
                }
            }
        }

        Ok(out)
    }
}

// ── RetinexMsrcr GPU (3 blur scales + accumulate + color restoration) ───

use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::{GpuShader, ReductionBuffer};
use crate::filters::spatial::{gaussian_kernel_bytes, blur_params};

gpu_filter_passes_only!(RetinexMsrcr,
    passes(self_, w, h) => {
        let scales = [self_.sigma_small, self_.sigma_medium, self_.sigma_large];
        let total_pixels = w * h;
        let acc_size = total_pixels as usize * 16;

        let mut passes = Vec::new();

        for sigma in &scales {
            let (kr, kb) = gaussian_kernel_bytes(*sigma);
            let bp = blur_params(w, h, kr);

            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                    .with_extra_buffers(vec![kb.clone()])
            );
            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                    .with_extra_buffers(vec![kb])
            );

            let mut acc_params = Vec::with_capacity(16);
            acc_params.extend_from_slice(&w.to_le_bytes());
            acc_params.extend_from_slice(&h.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());

            passes.push(
                GpuShader::new(enh_shaders::RETINEX_MSR_ACCUMULATE.to_string(), "main", [256, 1, 1], acc_params)
                    .with_reduction_buffers(vec![ReductionBuffer {
                        id: 0,
                        initial_data: vec![0u8; acc_size],
                        read_write: true,
                    }])
            );
        }

        // MSRCR color restoration pass
        let mut apply_params = Vec::with_capacity(32);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&3.0f32.to_le_bytes()); // num_scales
        apply_params.extend_from_slice(&self_.alpha.to_le_bytes());
        apply_params.extend_from_slice(&self_.beta.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        passes.push(
            GpuShader::new(enh_shaders::RETINEX_MSRCR_APPLY.to_string(), "main", [256, 1, 1], apply_params)
                .with_reduction_buffers(vec![
                    ReductionBuffer { id: 0, initial_data: vec![], read_write: false },
                ])
        );

        passes
    }
);
