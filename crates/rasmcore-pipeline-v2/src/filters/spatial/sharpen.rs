use crate::node::PipelineError;
use crate::ops::Filter;

use super::gaussian_blur::GaussianBlur;
use super::{blur_params, gaussian_kernel_bytes};
use crate::gpu_shaders::spatial;
use crate::node::GpuShader;

/// Unsharp mask sharpening — enhances edges by subtracting blurred from original.
///
/// `output = input + amount * (input - blur(input, radius))`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "sharpen",
    category = "spatial",
    cost = "O(n * r) via gaussian_blur"
)]
pub struct Sharpen {
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 10.0, default = 1.0)]
    pub amount: f32,
}

impl Filter for Sharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur {
            radius: self.radius,
        };
        let blurred = blur.compute(input, width, height)?;
        let amount = self.amount;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            // Alpha (every 4th channel) preserved unchanged
            if i % 4 == 3 {
                out.push(v);
            } else {
                out.push(v + amount * (v - blurred[i]));
            }
        }
        Ok(out)
    }
}

// ── Sharpen GPU (blur H + blur V + unsharp apply) ────────────────────────────

gpu_filter_passes_only!(Sharpen,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let blur_p = blur_params(w, h, kr);
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&self_.amount.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], blur_p.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], blur_p)
                .with_extra_buffers(vec![kb]),
            // Apply shader reads blurred input; extra_buffer = original (snapshotted before blur)
            GpuShader::new(spatial::SHARPEN_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);
