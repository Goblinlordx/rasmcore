use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// Multi-Scale Retinex (Jobson et al. 1997).
///
/// Averages SSR at three scales for better overall contrast.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "retinex_msr",
    category = "enhancement",
    cost = "O(3 * n * sigma) via gaussian_blur"
)]
pub struct RetinexMsr {
    #[param(min = 0.0, max = 200.0, default = 15.0)]
    pub sigma_small: f32,
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma_medium: f32,
    #[param(min = 0.0, max = 500.0, default = 250.0)]
    pub sigma_large: f32,
}

impl Filter for RetinexMsr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let scales = [self.sigma_small, self.sigma_medium, self.sigma_large];
        let n = input.len();
        let mut accum = vec![0.0f32; n];

        for sigma in &scales {
            let blur = GaussianBlur { radius: *sigma };
            let blurred = blur.compute(input, width, height)?;
            for (i, (inp, blr)) in input.iter().zip(blurred.iter()).enumerate() {
                if i % 4 == 3 {
                    continue; // skip alpha
                }
                let log_input = inp.max(1e-10).ln();
                let log_blur = blr.max(1e-10).ln();
                accum[i] += log_input - log_blur;
            }
        }

        // Average and normalize
        let inv_scales = 1.0 / scales.len() as f32;
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];
        for pixel in accum.chunks_exact_mut(4) {
            for c in 0..3 {
                pixel[c] *= inv_scales;
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }
        for (out_pixel, in_pixel) in accum.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
            for c in 0..3 {
                let range = max_val[c] - min_val[c];
                if range > 1e-10 {
                    out_pixel[c] = (out_pixel[c] - min_val[c]) / range;
                }
            }
            out_pixel[3] = in_pixel[3]; // alpha from input
        }

        Ok(accum)
    }
}

// ── RetinexMsr GPU (3 blur scales + accumulate + normalize) ─────────────

use crate::filters::spatial::{blur_params, gaussian_kernel_bytes};
use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::{GpuShader, ReductionBuffer};

gpu_filter_passes_only!(RetinexMsr,
    passes(self_, w, h) => {
        let scales = [self_.sigma_small, self_.sigma_medium, self_.sigma_large];
        let total_pixels = w * h;
        let acc_size = total_pixels as usize * 16; // vec4<f32> per pixel

        let mut passes = Vec::new();

        // For each scale: blur H, blur V, accumulate
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

        // Final normalize pass (reads accumulator, needs min/max reduction)
        // For simplicity, use the ChannelMinMax reduction on the accumulator
        let reduction = crate::gpu_shaders::reduction::GpuReduction::channel_min_max(256);
        let red_passes = reduction.build_passes(w, h);
        let red_read_buf = reduction.read_buffer(&red_passes);
        passes.push(red_passes.pass1);
        passes.push(red_passes.pass2);

        let mut norm_params = Vec::with_capacity(16);
        norm_params.extend_from_slice(&w.to_le_bytes());
        norm_params.extend_from_slice(&h.to_le_bytes());
        norm_params.extend_from_slice(&3.0f32.to_le_bytes()); // num_scales
        norm_params.extend_from_slice(&0u32.to_le_bytes());

        passes.push(
            GpuShader::new(enh_shaders::RETINEX_MSR_NORMALIZE.to_string(), "main", [256, 1, 1], norm_params)
                .with_reduction_buffers(vec![
                    ReductionBuffer { id: 0, initial_data: vec![], read_write: false },
                    red_read_buf,
                ])
        );

        passes
    }
);
