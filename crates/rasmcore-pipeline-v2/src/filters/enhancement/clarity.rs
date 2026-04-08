use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::luminance;

/// Clarity — midtone-weighted local contrast enhancement (Lightroom/Photoshop style).
///
/// Large-radius unsharp mask weighted by midtone curve:
/// `w(l) = 4 * l * (1 - l)` where l is normalized luminance.
/// `output = input + amount * (input - blur) * w(luminance)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clarity", category = "enhancement", cost = "O(n * radius) via gaussian_blur")]
pub struct Clarity {
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub amount: f32,
    #[param(min = 0.0, max = 100.0, default = 20.0)]
    pub radius: f32,
}

impl Filter for Clarity {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let amount = self.amount;
        let mut out = input.to_vec();

        for (pixel, blurred_pixel) in out.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
            let luma = luminance(pixel[0], pixel[1], pixel[2]);
            let aw = amount * 4.0 * luma * (1.0 - luma); // midtone-weighted amount
            pixel[0] += aw * (pixel[0] - blurred_pixel[0]);
            pixel[1] += aw * (pixel[1] - blurred_pixel[1]);
            pixel[2] += aw * (pixel[2] - blurred_pixel[2]);
        }

        Ok(out)
    }
}

// ── Clarity GPU (blur + midtone-weighted blend) ─────────────────────────

use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::GpuShader;
use crate::filters::spatial::{gaussian_kernel_bytes, blur_params};

gpu_filter_passes_only!(Clarity,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&self_.amount.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::CLARITY_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);
