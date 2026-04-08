use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

/// High-pass frequency layer — detail extraction.
///
/// `output = (input - blur(input)) + 0.5`
/// The 0.5 offset provides a neutral midpoint for compositing.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "frequency_high", category = "enhancement", cost = "O(n * sigma) via gaussian_blur")]
pub struct FrequencyHigh {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub sigma: f32,
}

impl Filter for FrequencyHigh {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(input, width, height)?;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha preserved
            } else {
                out.push((v - blurred[i]) + 0.5);
            }
        }
        Ok(out)
    }
}

// ── FrequencyHigh GPU (blur + subtract apply) ───────────────────────────

use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::GpuShader;
use crate::filters::spatial::{gaussian_kernel_bytes, blur_params};

gpu_filter_passes_only!(FrequencyHigh,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::FREQUENCY_HIGH_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);
