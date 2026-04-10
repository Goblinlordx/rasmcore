use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::ClutOp;
use crate::filters::helpers::{hsl_to_rgb, rgb_to_hsl};

/// Modulate — combined brightness/saturation/hue in HSL space.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "modulate", category = "color", cost = "O(n)")]
pub struct Modulate {
    /// Brightness factor (1.0=unchanged, 0=black, 2=double).
    #[param(min = 0.0, max = 3.0, default = 1.0)]
    pub brightness: f32,
    /// Saturation factor (1.0=unchanged, 0=grayscale).
    #[param(min = 0.0, max = 3.0, default = 1.0)]
    pub saturation: f32,
    /// Hue rotation in degrees.
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub hue: f32,
}

impl Filter for Modulate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (bri, sat, hue) = (self.brightness, self.saturation, self.hue);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let nl = (l * bri).clamp(0.0, 1.0);
            let ns = (s * sat).clamp(0.0, 1.0);
            let mut nh = (h + hue) % 360.0;
            if nh < 0.0 {
                nh += 360.0;
            }
            let (r, g, b) = hsl_to_rgb(nh, ns, nl);
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

impl ClutOp for Modulate {
    fn build_clut(&self) -> Clut3D {
        let (bri, sat, hue) = (self.brightness, self.saturation, self.hue);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let nl = (l * bri).clamp(0.0, 1.0);
            let ns = (s * sat).clamp(0.0, 1.0);
            let mut nh = (h + hue) % 360.0;
            if nh < 0.0 {
                nh += 360.0;
            }
            hsl_to_rgb(nh, ns, nl)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn modulate_identity() {
        let input = vec![0.5, 0.3, 0.7, 1.0];
        let f = Modulate {
            brightness: 1.0,
            saturation: 1.0,
            hue: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.3, 0.7), 0.02, "modulate identity");
    }
}
