use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::{rgb_to_hsl, hsl_to_rgb};
use super::ClutOp;

/// Selective color — adjust pixels matching a hue range in HSL space.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "selective_color", category = "color", cost = "O(n)")]
pub struct SelectiveColor {
    /// Target hue center (0-360).
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub target_hue: f32,
    /// Hue range (1-180).
    #[param(min = 1.0, max = 180.0, default = 30.0)]
    pub hue_range: f32,
    /// Hue shift (-180 to 180).
    #[param(min = -180.0, max = 180.0, default = 0.0)]
    pub hue_shift: f32,
    /// Saturation factor.
    #[param(min = 0.0, max = 3.0, default = 1.0)]
    pub saturation: f32,
    /// Lightness offset (-1 to 1).
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub lightness: f32,
}

impl Filter for SelectiveColor {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = selective_color_pixel(
                pixel[0],
                pixel[1],
                pixel[2],
                self.target_hue,
                self.hue_range,
                self.hue_shift,
                self.saturation,
                self.lightness,
            );
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

fn selective_color_pixel(
    r: f32,
    g: f32,
    b: f32,
    target_hue: f32,
    hue_range: f32,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
) -> (f32, f32, f32) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    let half = hue_range * 0.5;
    let mut diff = (h - target_hue).abs();
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    if diff > half {
        return (r, g, b);
    }
    // Cosine taper for smooth falloff
    let weight = 0.5 * (1.0 + (std::f32::consts::PI * diff / half).cos());
    let mut nh = h + hue_shift * weight;
    if nh < 0.0 {
        nh += 360.0;
    }
    if nh >= 360.0 {
        nh -= 360.0;
    }
    let ns = (s * (1.0 + (saturation - 1.0) * weight)).clamp(0.0, 1.0);
    let nl = (l + lightness * weight).clamp(0.0, 1.0);
    hsl_to_rgb(nh, ns, nl)
}

impl ClutOp for SelectiveColor {
    fn build_clut(&self) -> Clut3D {
        let (th, hr, hs, sat, lig) = (
            self.target_hue,
            self.hue_range,
            self.hue_shift,
            self.saturation,
            self.lightness,
        );
        Clut3D::from_fn(33, move |r, g, b| selective_color_pixel(r, g, b, th, hr, hs, sat, lig))
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
    fn selective_color_out_of_range() {
        // Blue pixel, targeting red hue — should be unchanged
        let input = vec![0.0, 0.0, 1.0, 1.0];
        let f = SelectiveColor {
            target_hue: 0.0,
            hue_range: 30.0,
            hue_shift: 90.0,
            saturation: 2.0,
            lightness: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.0, 0.0, 1.0), 0.02, "selective_color no match");
    }
}
