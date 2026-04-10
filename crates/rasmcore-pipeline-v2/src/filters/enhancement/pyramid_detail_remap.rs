use crate::node::PipelineError;
use crate::ops::Filter;

/// Laplacian pyramid detail remap — enhance or suppress fine detail.
///
/// `sigma < 1.0`: enhance fine detail (compress large gradients).
/// `sigma > 1.0`: suppress fine detail (smoothing).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "pyramid_detail_remap",
    category = "enhancement",
    cost = "O(n * levels * sigma)"
)]
pub struct PyramidDetailRemap {
    #[param(min = 0.0, max = 5.0, default = 0.5)]
    pub sigma: f32,
    #[param(min = 0, max = 10, default = 0)]
    pub levels: u32,
}

impl Filter for PyramidDetailRemap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let sigma = self.sigma;
        let levels = if self.levels == 0 {
            // Auto: log2(min(w,h)) - 2, clamped to [3, 7]
            ((w.min(h) as f32).log2() as u32)
                .saturating_sub(2)
                .clamp(3, 7)
        } else {
            self.levels
        };

        // Process each RGB channel independently
        let mut out = input.to_vec();
        for c in 0..3 {
            // Extract single channel
            let mut channel: Vec<f32> = input.chunks_exact(4).map(|p| p[c]).collect();

            // Build Gaussian pyramid
            let mut gaussians = vec![channel.clone()];
            let mut cw = w;
            let mut ch = h;
            for _ in 0..levels {
                let nw = cw.div_ceil(2);
                let nh = ch.div_ceil(2);
                let prev = gaussians.last().unwrap();
                let mut next = vec![0.0f32; nw * nh];
                for y in 0..nh {
                    for x in 0..nw {
                        let sx = (x * 2).min(cw - 1);
                        let sy = (y * 2).min(ch - 1);
                        // Simple 2x2 average downsample
                        let sx1 = (sx + 1).min(cw - 1);
                        let sy1 = (sy + 1).min(ch - 1);
                        next[y * nw + x] = (prev[sy * cw + sx]
                            + prev[sy * cw + sx1]
                            + prev[sy1 * cw + sx]
                            + prev[sy1 * cw + sx1])
                            * 0.25;
                    }
                }
                gaussians.push(next);
                cw = nw;
                ch = nh;
            }

            // Build Laplacian pyramid and remap detail coefficients
            for level in 0..levels as usize {
                let lw = if level == 0 { w } else { (w >> level).max(1) };
                let lh = if level == 0 { h } else { (h >> level).max(1) };
                // Upsample coarser level
                let uw = lw;
                let uh = lh;
                let mut upsampled = vec![0.0f32; uw * uh];
                let coarse = &gaussians[level + 1];
                let cw_coarse = lw.div_ceil(2);
                for y in 0..uh {
                    for x in 0..uw {
                        let cx = x / 2;
                        let cy = y / 2;
                        let cx = cx.min(cw_coarse.saturating_sub(1));
                        let cy = cy.min((lh.div_ceil(2)).saturating_sub(1));
                        upsampled[y * uw + x] = coarse[cy * cw_coarse + cx];
                    }
                }

                // Remap Laplacian detail: d * sigma / (sigma + |d|)
                let fine = &mut gaussians[level];
                for i in 0..fine.len().min(upsampled.len()) {
                    let detail = fine[i] - upsampled[i];
                    let remapped = if sigma.abs() > 1e-10 {
                        detail * sigma / (sigma + detail.abs())
                    } else {
                        0.0
                    };
                    fine[i] = upsampled[i] + remapped;
                }
            }

            // Write remapped channel back
            channel = gaussians[0].clone();
            for (i, pixel) in out.chunks_exact_mut(4).enumerate() {
                if i < channel.len() {
                    pixel[c] = channel[i];
                }
            }
        }

        Ok(out)
    }
}

// ── PyramidDetailRemap GPU (multi-pass Laplacian pyramid) ───────────────

use crate::gpu_shaders::enhancement as enh_shaders;
use crate::node::GpuShader;

gpu_filter_passes_only!(PyramidDetailRemap,
    passes(self_, w, h) => {
        let levels = if self_.levels == 0 {
            ((w.min(h) as f32).log2() as u32).saturating_sub(2).clamp(3, 7)
        } else {
            self_.levels
        };

        let mut passes = Vec::new();
        let mut dims: Vec<(u32, u32)> = vec![(w, h)];

        // Build downsample chain
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            let nw = cw.div_ceil(2);
            let nh = ch.div_ceil(2);
            let mut ds_params = Vec::with_capacity(16);
            ds_params.extend_from_slice(&cw.to_le_bytes());
            ds_params.extend_from_slice(&ch.to_le_bytes());
            ds_params.extend_from_slice(&nw.to_le_bytes());
            ds_params.extend_from_slice(&nh.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::DOWNSAMPLE_2X.to_string(), "main", [256, 1, 1], ds_params)
            );
            dims.push((nw, nh));
        }

        // Remap + upsample chain (coarsest to finest)
        for level in (0..levels as usize).rev() {
            let (lw, lh) = dims[level];
            let (cw, ch) = dims[level + 1];

            // Upsample coarser level
            let mut us_params = Vec::with_capacity(16);
            us_params.extend_from_slice(&cw.to_le_bytes());
            us_params.extend_from_slice(&ch.to_le_bytes());
            us_params.extend_from_slice(&lw.to_le_bytes());
            us_params.extend_from_slice(&lh.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::UPSAMPLE_2X.to_string(), "main", [256, 1, 1], us_params)
            );

            // Remap Laplacian detail at this level
            let mut remap_params = Vec::with_capacity(16);
            remap_params.extend_from_slice(&lw.to_le_bytes());
            remap_params.extend_from_slice(&lh.to_le_bytes());
            remap_params.extend_from_slice(&self_.sigma.to_le_bytes());
            remap_params.extend_from_slice(&0u32.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::PYRAMID_REMAP_LEVEL.to_string(), "main", [256, 1, 1], remap_params)
            );
        }

        passes
    }
);
