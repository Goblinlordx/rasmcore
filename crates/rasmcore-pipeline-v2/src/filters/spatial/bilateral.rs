use crate::node::PipelineError;
use crate::ops::Filter;

use super::{accum3, clamp_coord};
use crate::gpu_shaders::spatial;

/// Bilateral filter — edge-preserving smoothing.
///
/// Uses CIE-Lab L2 Euclidean distance for color similarity, matching the
/// original Tomasi & Manduchi 1998 recommendation and MATLAB's imbilatfilt.
/// Perceptually uniform: colors that look similar to humans are treated as
/// similar by the filter.
///
/// The smoothing operates in the input color space (linear RGB), but the
/// edge-detection (color weight) uses Lab distance for perceptual accuracy.
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

/// Convert linear RGB to CIE Lab (D65) for perceptual color distance.
/// Inline to avoid pulling in the sRGB-input version from color/mod.rs.
#[inline]
fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Linear RGB → XYZ (D65), IEC 61966-2-1 matrix
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
    // XYZ → Lab (D65 whitepoint)
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA;
    let lab_f = |t: f32| -> f32 {
        if t > DELTA3 {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    };
    let fx = lab_f(x / XN);
    let fy = lab_f(y / YN);
    let fz = lab_f(z / ZN);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

impl Filter for Bilateral {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = (self.diameter / 2) as i32;
        let sc2 = -0.5 / (self.sigma_color * self.sigma_color);
        let ss2 = -0.5 / (self.sigma_space * self.sigma_space);

        // Pre-compute Lab values for all pixels (avoids redundant conversion)
        let mut lab = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            let idx = i * 4;
            let (l, a, b) = linear_rgb_to_lab(input[idx], input[idx + 1], input[idx + 2]);
            lab.push(l);
            lab.push(a);
            lab.push(b);
        }

        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) * 4;
                let center_lab = (y * w + x) * 3;
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = clamp_coord(x as i32 + dx, w);
                        let sy = clamp_coord(y as i32 + dy, h);
                        let idx = (sy * w + sx) * 4;
                        let lab_idx = (sy * w + sx) * 3;

                        // Spatial weight
                        let dist2 = (dx * dx + dy * dy) as f32;
                        let ws = (dist2 * ss2).exp();

                        // Color weight: L2 Euclidean in CIE-Lab (perceptual)
                        let dl = lab[lab_idx] - lab[center_lab];
                        let da = lab[lab_idx + 1] - lab[center_lab + 1];
                        let db = lab[lab_idx + 2] - lab[center_lab + 2];
                        let color_dist2 = dl * dl + da * da + db * db;
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
