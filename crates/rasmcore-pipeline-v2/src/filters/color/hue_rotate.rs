use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::ClutOp;
use crate::filters::helpers::{hsl_to_rgb, rgb_to_hsl};

/// Hue rotation in HSL space.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "hue_rotate", category = "color", cost = "O(n)")]
pub struct HueRotate {
    /// Rotation in degrees (0-360).
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub degrees: f32,
}

impl Filter for HueRotate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let deg = self.degrees;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let nh = (h + deg) % 360.0;
            let (r, g, b) = hsl_to_rgb(if nh < 0.0 { nh + 360.0 } else { nh }, s, l);
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

impl ClutOp for HueRotate {
    fn build_clut(&self) -> Clut3D {
        let deg = self.degrees;
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let nh = (h + deg) % 360.0;
            hsl_to_rgb(if nh < 0.0 { nh + 360.0 } else { nh }, s, l)
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
    fn hue_rotate_180_inverts_hue() {
        let input = test_pixel(1.0, 0.0, 0.0); // red
        let f = HueRotate { degrees: 180.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Red rotated 180° = cyan (0, 1, 1)
        assert_rgb_close(&out, (0.0, 1.0, 1.0), 0.02, "hue_rotate 180");
    }

    #[test]
    fn hue_rotate_clut_matches_compute() {
        let f = HueRotate { degrees: 90.0 };
        let input = test_pixel(0.8, 0.2, 0.4);
        let computed = f.compute(&input, 1, 1).unwrap();
        let clut = f.build_clut();
        let (cr, cg, cb) = clut.sample(0.8, 0.2, 0.4);
        assert!(
            (computed[0] - cr).abs() < 0.05
                && (computed[1] - cg).abs() < 0.05
                && (computed[2] - cb).abs() < 0.05,
            "CLUT mismatch: compute=({:.3},{:.3},{:.3}) clut=({cr:.3},{cg:.3},{cb:.3})",
            computed[0],
            computed[1],
            computed[2]
        );
    }
}
