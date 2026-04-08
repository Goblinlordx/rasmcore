use crate::node::PipelineError;
use crate::ops::Filter;

use super::clamp_coord;

/// Non-Local Means denoising (Buades et al. 2005).
///
/// Compares patches in search window, weights by similarity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "nlm_denoise", category = "enhancement", cost = "O(n * sr^2 * pr^2)")]
pub struct NlmDenoise {
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub h: f32,
    #[param(min = 1, max = 10, default = 3)]
    pub patch_radius: u32,
    #[param(min = 1, max = 30, default = 10)]
    pub search_radius: u32,
}

impl Filter for NlmDenoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let pr = self.patch_radius as i32;
        let sr = self.search_radius as i32;
        let h2 = self.h * self.h;
        if h2 < 1e-10 {
            return Ok(input.to_vec());
        }
        let inv_h2 = -1.0 / h2;
        let patch_size = ((2 * pr + 1) * (2 * pr + 1)) as f32;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for sy in -sr..=sr {
                    for sx in -sr..=sr {
                        let nx = clamp_coord(x as i32 + sx, w);
                        let ny = clamp_coord(y as i32 + sy, h);

                        // Patch distance
                        let mut dist2 = 0.0f32;
                        for py in -pr..=pr {
                            for px in -pr..=pr {
                                let cx1 = clamp_coord(x as i32 + px, w);
                                let cy1 = clamp_coord(y as i32 + py, h);
                                let cx2 = clamp_coord(nx as i32 + px, w);
                                let cy2 = clamp_coord(ny as i32 + py, h);
                                let i1 = (cy1 * w + cx1) * 4;
                                let i2 = (cy2 * w + cx2) * 4;
                                for c in 0..3 {
                                    let d = input[i1 + c] - input[i2 + c];
                                    dist2 += d * d;
                                }
                            }
                        }
                        dist2 /= patch_size * 3.0;

                        let weight = (dist2 * inv_h2).exp();
                        let nidx = (ny * w + nx) * 4;
                        for c in 0..3 {
                            sum[c] += weight * input[nidx + c];
                        }
                        weight_sum += weight;
                    }
                }

                let idx = (y * w + x) * 4;
                let inv_w = if weight_sum > 1e-10 { 1.0 / weight_sum } else { 1.0 };
                for c in 0..3 {
                    out[idx + c] = sum[c] * inv_w;
                }
                out[idx + 3] = input[idx + 3]; // alpha
            }
        }

        Ok(out)
    }
}

// ── NlmDenoise GPU (single-pass, compute-heavy) ─────────────────────────

use crate::gpu_shaders::analysis;

gpu_filter!(NlmDenoise,
    shader: analysis::NLM_DENOISE,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h, self_.search_radius, self_.patch_radius,
        self_.h, 0u32, 0u32, 0u32
    ]
);
