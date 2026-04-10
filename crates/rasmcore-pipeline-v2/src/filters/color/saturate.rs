use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::ClutOp;

/// Perceptual saturation adjustment using OKLCH (OKLab cylindrical).
///
/// Scales chroma in OKLCH space — perceptually uniform across hues.
/// Equal factor changes produce visually equal saturation changes
/// for red, green, blue, and all intermediate hues.
///
/// Reference: Ottosson, B. (2020). "A perceptual color space for image processing."
/// Also: W3C CSS Color Level 4, Section 8 (OKLCH).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "saturate", category = "color", cost = "O(n)")]
pub struct Saturate {
    /// Chroma scale factor: 0=grayscale, 1=unchanged, 2=double saturation.
    #[param(min = 0.0, max = 3.0, default = 1.0)]
    pub factor: f32,
}

impl Filter for Saturate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        use crate::color_math::{
            linear_srgb_to_oklab, oklab_to_linear_srgb, oklab_to_oklch, oklch_to_oklab,
        };
        let factor = self.factor;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (l, a, b) = linear_srgb_to_oklab(pixel[0], pixel[1], pixel[2]);
            let (l2, c, h) = oklab_to_oklch(l, a, b);
            let (l3, a2, b2) = oklch_to_oklab(l2, c * factor, h);
            let (r, g, b3) = oklab_to_linear_srgb(l3, a2, b2);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b3;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for Saturate {
    fn build_clut(&self) -> Clut3D {
        use crate::color_math::{
            linear_srgb_to_oklab, oklab_to_linear_srgb, oklab_to_oklch, oklch_to_oklab,
        };
        let factor = self.factor;
        Clut3D::from_fn(33, move |r, g, b| {
            let (l, a, ob) = linear_srgb_to_oklab(r, g, b);
            let (l2, c, h) = oklab_to_oklch(l, a, ob);
            let (l3, a2, b2) = oklch_to_oklab(l2, c * factor, h);
            oklab_to_linear_srgb(l3, a2, b2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturate_zero_is_grayscale() {
        let input = vec![0.8, 0.2, 0.4, 1.0];
        let f = Saturate { factor: 0.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // All channels should be equal (grayscale)
        assert!(
            (out[0] - out[1]).abs() < 0.02 && (out[1] - out[2]).abs() < 0.02,
            "Expected grayscale, got ({:.3}, {:.3}, {:.3})",
            out[0],
            out[1],
            out[2]
        );
    }
}
