use crate::node::PipelineError;
use crate::ops::Filter;

use super::gaussian_blur::GaussianBlur;
use super::{gaussian_kernel_bytes, blur_params};
use crate::gpu_shaders::spatial;
use crate::node::GpuShader;

/// High pass filter — subtracts blur from original, adding mid-gray offset.
///
/// `output = (input - blur(input)) + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "high_pass", category = "spatial", cost = "O(n * r) via gaussian_blur")]
pub struct HighPass {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub radius: f32,
}

impl Filter for HighPass {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha
            } else {
                out.push((v - blurred[i]) + 0.5);
            }
        }
        Ok(out)
    }
}

// ── HighPass GPU (blur H + blur V + subtract apply) ─────────────────────────

gpu_filter_passes_only!(HighPass,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let blur_p = blur_params(w, h, kr);
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], blur_p.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], blur_p)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(spatial::HIGH_PASS_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);
