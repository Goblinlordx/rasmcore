use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::{rgb_to_hsl, hsl_to_rgb};
use super::ClutOp;

/// Replace color — select pixels by HSL ranges and shift them.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "replace_color", category = "color", cost = "O(n)")]
pub struct ReplaceColor {
    /// Center hue (0-360).
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub center_hue: f32,
    /// Hue range (1-180).
    #[param(min = 1.0, max = 180.0, default = 30.0)]
    pub hue_range: f32,
    /// Saturation range [min, max].
    #[param(min = 0.0, max = 1.0, default = 0.0)]
    pub sat_min: f32,
    #[param(min = 0.0, max = 1.0, default = 1.0)]
    pub sat_max: f32,
    /// Lightness range [min, max].
    #[param(min = 0.0, max = 1.0, default = 0.0)]
    pub lum_min: f32,
    #[param(min = 0.0, max = 1.0, default = 1.0)]
    pub lum_max: f32,
    /// Shift amounts.
    #[param(min = -180.0, max = 180.0, default = 0.0)]
    pub hue_shift: f32,
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub sat_shift: f32,
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub lum_shift: f32,
}

impl Filter for ReplaceColor {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = replace_color_pixel(pixel[0], pixel[1], pixel[2], self);
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

fn replace_color_pixel(r: f32, g: f32, b: f32, p: &ReplaceColor) -> (f32, f32, f32) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    // Check saturation and lightness ranges
    if s < p.sat_min || s > p.sat_max || l < p.lum_min || l > p.lum_max {
        return (r, g, b);
    }
    // Check hue range with cosine falloff
    let half = p.hue_range * 0.5;
    let mut diff = (h - p.center_hue).abs();
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    if diff > half {
        return (r, g, b);
    }
    let weight = 0.5 * (1.0 + (std::f32::consts::PI * diff / half).cos());
    let mut nh = h + p.hue_shift * weight;
    if nh < 0.0 {
        nh += 360.0;
    }
    if nh >= 360.0 {
        nh -= 360.0;
    }
    let ns = (s + p.sat_shift * weight).clamp(0.0, 1.0);
    let nl = (l + p.lum_shift * weight).clamp(0.0, 1.0);
    hsl_to_rgb(nh, ns, nl)
}

impl ClutOp for ReplaceColor {
    fn build_clut(&self) -> Clut3D {
        let p = self.clone();
        Clut3D::from_fn(33, move |r, g, b| replace_color_pixel(r, g, b, &p))
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
    fn replace_color_no_match() {
        // Gray pixel (no hue) should not be affected by hue-targeted replacement
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let f = ReplaceColor {
            center_hue: 0.0,
            hue_range: 30.0,
            sat_min: 0.5,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 90.0,
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.01, "replace_color no match");
    }
}
