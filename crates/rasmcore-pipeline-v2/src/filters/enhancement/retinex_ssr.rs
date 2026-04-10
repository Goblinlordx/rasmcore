use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// Single-Scale Retinex (Land 1977, Jobson et al. 1997).
///
/// `R(x,y) = log(I(x,y)) - log(G * I(x,y))`
/// Enhances local contrast by removing illumination estimate.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "retinex_ssr",
    category = "enhancement",
    cost = "O(n * sigma) via gaussian_blur"
)]
pub struct RetinexSsr {
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma: f32,
}

impl Filter for RetinexSsr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(input, width, height)?;

        let mut out = input.to_vec();
        // Compute log-ratio and normalize
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];

        for (pixel, blur_pixel) in out.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
            for c in 0..3 {
                let log_input = (pixel[c].max(1e-10)).ln();
                let log_blur = (blur_pixel[c].max(1e-10)).ln();
                pixel[c] = log_input - log_blur;
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }

        // Normalize to [0, 1]
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

// ── RetinexSsr GPU (blur + log-domain apply) ────────────────────────────

use crate::filters::spatial::{blur_params, gaussian_kernel_bytes};
use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::GpuShader;

gpu_filter_passes_only!(RetinexSsr,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&1.0f32.to_le_bytes()); // gain
        apply_params.extend_from_slice(&0.0f32.to_le_bytes()); // offset

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::RETINEX_SSR_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);
