use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::{rgb_to_hsl, hsl_to_rgb};
use super::ClutOp;

/// Vibrance — perceptually-weighted saturation boost.
/// Boosts less-saturated colors more than already-saturated ones.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "vibrance", category = "color", cost = "O(n)")]
pub struct Vibrance {
    /// Amount: -100 to 100.
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub amount: f32,
}

impl Filter for Vibrance {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let amt = self.amount / 100.0;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > 1e-7 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let scale = amt * (1.0 - sat);
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let ns = (s * (1.0 + scale)).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsl_to_rgb(h, ns, l);
            pixel[0] = nr;
            pixel[1] = ng;
            pixel[2] = nb;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for Vibrance {
    fn build_clut(&self) -> Clut3D {
        let amt = self.amount / 100.0;
        Clut3D::from_fn(33, move |r, g, b| {
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > 1e-7 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let scale = amt * (1.0 - sat);
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let ns = (s * (1.0 + scale)).clamp(0.0, 1.0);
            hsl_to_rgb(h, ns, l)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vibrance_neutral_on_saturated() {
        let input = vec![1.0, 0.0, 0.0, 1.0]; // fully saturated red
        let f = Vibrance { amount: 50.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Should change minimally (already saturated)
        assert!(
            (out[0] - 1.0).abs() < 0.1,
            "Vibrance should barely affect saturated color"
        );
    }
}
