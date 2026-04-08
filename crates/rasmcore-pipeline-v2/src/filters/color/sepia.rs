use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::ClutOp;

/// Sepia tone — warm brownish tint via standard matrix blend.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sepia", category = "color", cost = "O(n)")]
pub struct Sepia {
    /// Intensity: 0=none, 1=full sepia.
    #[param(min = 0.0, max = 1.0, default = 1.0)]
    pub intensity: f32,
}

impl Filter for Sepia {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let t = self.intensity;
        let inv = 1.0 - t;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
            let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
            let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);
            pixel[0] = inv * r + t * sr;
            pixel[1] = inv * g + t * sg;
            pixel[2] = inv * b + t * sb;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for Sepia {
    fn build_clut(&self) -> Clut3D {
        let t = self.intensity;
        let inv = 1.0 - t;
        Clut3D::from_fn(33, move |r, g, b| {
            let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
            let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
            let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);
            (inv * r + t * sr, inv * g + t * sg, inv * b + t * sb)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pixel(r: f32, g: f32, b: f32) -> Vec<f32> {
        vec![r, g, b, 1.0]
    }

    fn assert_rgb_close(actual: &[f32], expected: (f32, f32, f32), tol: f32, label: &str) {
        assert!(
            (actual[0] - expected.0).abs() < tol
                && (actual[1] - expected.1).abs() < tol
                && (actual[2] - expected.2).abs() < tol,
            "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4})",
            expected.0,
            expected.1,
            expected.2,
            actual[0],
            actual[1],
            actual[2]
        );
    }

    #[test]
    fn sepia_full_intensity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = Sepia { intensity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Sepia: R > G > B
        assert!(out[0] > out[1] && out[1] > out[2], "Sepia should be warm-toned");
    }

    #[test]
    fn sepia_zero_is_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = Sepia { intensity: 0.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "sepia 0");
    }
}
