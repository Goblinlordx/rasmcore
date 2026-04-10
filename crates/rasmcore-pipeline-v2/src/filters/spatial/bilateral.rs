use crate::node::PipelineError;
use crate::ops::Filter;

use super::{accum3, clamp_coord};
use crate::gpu_shaders::spatial;

/// Bilateral filter — edge-preserving smoothing.
///
/// Weights pixels by both spatial distance and color similarity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "bilateral", category = "spatial", cost = "O(n * d^2)")]
pub struct Bilateral {
    #[param(min = 1, max = 50, default = 5)]
    pub diameter: u32,
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub sigma_color: f32,
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub sigma_space: f32,
}

impl Filter for Bilateral {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = (self.diameter / 2) as i32;
        let sc2 = -0.5 / (self.sigma_color * self.sigma_color);
        let ss2 = -0.5 / (self.sigma_space * self.sigma_space);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) * 4;
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = clamp_coord(x as i32 + dx, w);
                        let sy = clamp_coord(y as i32 + dy, h);
                        let idx = (sy * w + sx) * 4;

                        // Spatial weight
                        let dist2 = (dx * dx + dy * dy) as f32;
                        let ws = (dist2 * ss2).exp();

                        // Color weight (Euclidean distance in RGB)
                        let dr = input[idx] - input[center_idx];
                        let dg = input[idx + 1] - input[center_idx + 1];
                        let db = input[idx + 2] - input[center_idx + 2];
                        let color_dist2 = dr * dr + dg * dg + db * db;
                        let wc = (color_dist2 * sc2).exp();

                        let weight = ws * wc;
                        accum3(&mut sum, &input[idx..], weight);
                        weight_sum += weight;
                    }
                }

                let inv_w = if weight_sum > 1e-10 {
                    1.0 / weight_sum
                } else {
                    0.0
                };
                out[center_idx] = sum[0] * inv_w;
                out[center_idx + 1] = sum[1] * inv_w;
                out[center_idx + 2] = sum[2] * inv_w;
                out[center_idx + 3] = input[center_idx + 3]; // alpha
            }
        }

        Ok(out)
    }
}

// ── Bilateral GPU (single-pass neighborhood) ────────────────────────────────

gpu_filter!(Bilateral,
    shader: spatial::BILATERAL,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h, self_.diameter / 2, 0u32,
        -0.5f32 / (self_.sigma_color * self_.sigma_color),
        -0.5f32 / (self_.sigma_space * self_.sigma_space),
        0u32, 0u32
    ]
);
