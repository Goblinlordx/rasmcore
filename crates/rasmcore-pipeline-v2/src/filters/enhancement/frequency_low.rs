use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// Low-pass frequency layer — Gaussian blur.
///
/// Extracts large-scale color/tone structure.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "frequency_low",
    category = "enhancement",
    cost = "O(n * sigma) via gaussian_blur"
)]
pub struct FrequencyLow {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub sigma: f32,
}

impl Filter for FrequencyLow {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        blur.compute(input, width, height)
    }
}

// ── FrequencyLow GPU (same as GaussianBlur) ─────────────────────────────

use crate::filters::spatial::{blur_params, gaussian_kernel_bytes};
use crate::gpu_shaders::spatial;
use crate::node::GpuShader;

gpu_filter_passes_only!(FrequencyLow,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
        ]
    }
);
