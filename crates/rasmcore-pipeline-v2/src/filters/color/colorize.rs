use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::luminance;
use super::ClutOp;

/// Colorize — tint image with a target color using W3C luma blend.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "colorize", category = "color", cost = "O(n)")]
pub struct Colorize {
    /// Target color RGB [0,1].
    #[param(min = 0.0, max = 1.0, default = 1.0)]
    pub target_r: f32,
    #[param(min = 0.0, max = 1.0, default = 0.8)]
    pub target_g: f32,
    #[param(min = 0.0, max = 1.0, default = 0.6)]
    pub target_b: f32,
    /// Blend amount: 0=none, 1=full.
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub amount: f32,
}

impl Filter for Colorize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (tr, tg, tb) = (self.target_r, self.target_g, self.target_b);
        let amt = self.amount;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let luma = luminance(pixel[0], pixel[1], pixel[2]);
            pixel[0] = pixel[0] + (luma * tr - pixel[0]) * amt;
            pixel[1] = pixel[1] + (luma * tg - pixel[1]) * amt;
            pixel[2] = pixel[2] + (luma * tb - pixel[2]) * amt;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for Colorize {
    fn build_clut(&self) -> Clut3D {
        let (tr, tg, tb, amt) = (self.target_r, self.target_g, self.target_b, self.amount);
        Clut3D::from_fn(33, move |r, g, b| {
            let luma = luminance(r, g, b);
            (
                r + (luma * tr - r) * amt,
                g + (luma * tg - g) * amt,
                b + (luma * tb - b) * amt,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colorize_preserves_structure() {
        let input = vec![0.2, 0.5, 0.8, 1.0];
        let f = Colorize {
            target_r: 1.0,
            target_g: 0.8,
            target_b: 0.2,
            amount: 0.5,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[3] == 1.0, "Alpha preserved");
    }
}
