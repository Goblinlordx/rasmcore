use crate::node::PipelineError;
use crate::ops::Filter;

use super::clamp_coord;
use crate::gpu_shaders::spatial;

/// Median filter — replaces each pixel with median of its neighborhood.
///
/// Effective for salt-and-pepper noise removal while preserving edges.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "median", category = "spatial", cost = "O(n * r^2 * log r)")]
pub struct Median {
    #[param(min = 0, max = 50, default = 1)]
    pub radius: u32,
}

impl Filter for Median {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let mut out = vec![0.0f32; w * h * 4];
        let mut neighborhood = Vec::new();

        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    if c == 3 {
                        // Alpha: pass through
                        out[(y * w + x) * 4 + 3] = input[(y * w + x) * 4 + 3];
                        continue;
                    }
                    neighborhood.clear();
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sx = clamp_coord(x as i32 + dx, w);
                            let sy = clamp_coord(y as i32 + dy, h);
                            neighborhood.push(input[(sy * w + sx) * 4 + c]);
                        }
                    }
                    neighborhood.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    out[(y * w + x) * 4 + c] = neighborhood[neighborhood.len() / 2];
                }
            }
        }

        Ok(out)
    }
}

// ── Median GPU (single-pass sorting network) ────────────────────────────────

gpu_filter!(Median,
    shader: spatial::MEDIAN,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [w, h, self_.radius, 0u32]
);
