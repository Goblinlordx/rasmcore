use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::luminance;

/// Shadow/Highlight adjustment — local tone mapping.
///
/// Independently lighten shadows and darken highlights via soft-light blending
/// on the luminance channel with compress-gated weight masks.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "shadow_highlight", category = "enhancement", cost = "O(n * radius)")]
pub struct ShadowHighlight {
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub shadows: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub highlights: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub whitepoint: f32,
    #[param(min = 0.0, max = 200.0, default = 30.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub compress: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub shadows_ccorrect: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub highlights_ccorrect: f32,
}

impl Filter for ShadowHighlight {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let shadows = self.shadows / 100.0;
        let highlights = self.highlights / 100.0;
        let whitepoint = self.whitepoint;
        let compress = self.compress / 100.0;
        let sc = self.shadows_ccorrect / 100.0;
        let hc = self.highlights_ccorrect / 100.0;

        // Extract luminance and blur it
        let luma: Vec<f32> = input
            .chunks_exact(4)
            .map(|p| luminance(p[0], p[1], p[2]))
            .collect();

        // Blur luminance
        let mut luma_rgba: Vec<f32> = luma.iter().flat_map(|&v| [v, v, v, 1.0]).collect();
        let blur = GaussianBlur { radius: self.radius };
        luma_rgba = blur.compute(&luma_rgba, width, height)?;
        let blurred_luma: Vec<f32> = luma_rgba.chunks_exact(4).map(|p| p[0]).collect();

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let bl = blurred_luma[y * w + x];

                // Shadow weight: strongest at dark pixels
                let sw = 1.0 - bl;
                let sw = sw * sw; // quadratic falloff

                // Highlight weight: strongest at bright pixels
                let hw = bl;
                let hw = hw * hw;

                // Compress: reduce effect near midtones
                let sw = if compress > 0.0 {
                    sw * (1.0 - compress * 4.0 * bl * (1.0 - bl))
                } else {
                    sw
                };
                let hw = if compress > 0.0 {
                    hw * (1.0 - compress * 4.0 * bl * (1.0 - bl))
                } else {
                    hw
                };

                // Luminance adjustment
                let luma_adj = shadows * sw - highlights * hw + whitepoint * 0.01;

                let cur_luma = luminance(out[idx], out[idx + 1], out[idx + 2]).max(1e-10);
                let new_luma = (cur_luma + luma_adj).max(0.0);
                let ratio = new_luma / cur_luma;

                // Apply luminance ratio with saturation correction
                for c in 0..3 {
                    let v = out[idx + c];
                    let gray = cur_luma;
                    let chroma = v - gray;

                    // Shadow saturation correction
                    let sat_adj = 1.0 + chroma.signum() * sw * (sc - 1.0)
                        + chroma.signum() * hw * (hc - 1.0);

                    out[idx + c] = new_luma + chroma * sat_adj.max(0.0) * ratio;
                }
            }
        }

        Ok(out)
    }
}

// ── ShadowHighlight GPU (blur luma + apply) ─────────────────────────────

use crate::gpu_shaders::{enhancement as enh_shaders, spatial};
use crate::node::GpuShader;
use crate::filters::spatial::{gaussian_kernel_bytes, blur_params};

gpu_filter_passes_only!(ShadowHighlight,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let bp = blur_params(w, h, kr);

        // Pass 1-2: blur the input (used to estimate local luminance)
        let blur_h = GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
            .with_extra_buffers(vec![kb.clone()]);
        let blur_v = GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
            .with_extra_buffers(vec![kb]);

        // Pass 3: shadow/highlight apply with blurred luma from previous passes
        let shadows_norm = self_.shadows / 100.0;
        let highlights_norm = self_.highlights / 100.0;
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&shadows_norm.to_le_bytes());
        apply_params.extend_from_slice(&highlights_norm.to_le_bytes());
        let apply = GpuShader::new(
            enh_shaders::SHADOW_HIGHLIGHT_APPLY.to_string(), "main", [256, 1, 1], apply_params,
        );

        vec![blur_h, blur_v, apply]
    }
);
