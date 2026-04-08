use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::{rgb_to_hsl, hsl_to_rgb};
use super::ClutOp;

/// Saturation adjustment in HSL space (legacy — prefer perceptual Saturate).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "saturate_hsl", category = "color", cost = "O(n)")]
pub struct SaturateHsl {
    /// Factor: 0=grayscale, 1=unchanged, 2=double.
    #[param(min = 0.0, max = 3.0, default = 1.0)]
    pub factor: f32,
}

impl Filter for SaturateHsl {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let factor = self.factor;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let (r, g, b) = hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for SaturateHsl {
    fn build_clut(&self) -> Clut3D {
        let factor = self.factor;
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l)
        })
    }
}
