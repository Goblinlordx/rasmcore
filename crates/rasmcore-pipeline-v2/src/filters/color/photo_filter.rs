use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use crate::filters::helpers::luminance;
use super::ClutOp;

/// Photo filter — color overlay with optional luminosity preservation.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "photo_filter", category = "color", cost = "O(n)")]
pub struct PhotoFilter {
    /// Filter color RGB [0,1].
    #[param(min = 0.0, max = 1.0, default = 1.0)]
    pub color_r: f32,
    #[param(min = 0.0, max = 1.0, default = 0.6)]
    pub color_g: f32,
    #[param(min = 0.0, max = 1.0, default = 0.2)]
    pub color_b: f32,
    /// Density: 0=none, 1=full overlay.
    #[param(min = 0.0, max = 1.0, default = 0.25)]
    pub density: f32,
    /// Preserve original luminosity.
    #[param(default = false)]
    pub preserve_luminosity: bool,
}

impl Filter for PhotoFilter {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (fr, fg, fb) = (self.color_r, self.color_g, self.color_b);
        let d = self.density;
        let preserve = self.preserve_luminosity;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let mut nr = r + (fr - r) * d;
            let mut ng = g + (fg - g) * d;
            let mut nb = b + (fb - b) * d;
            if preserve {
                let orig_luma = luminance(r, g, b);
                let new_luma = luminance(nr, ng, nb);
                if new_luma > 1e-7 {
                    let scale = orig_luma / new_luma;
                    nr *= scale;
                    ng *= scale;
                    nb *= scale;
                }
            }
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

impl ClutOp for PhotoFilter {
    fn build_clut(&self) -> Clut3D {
        let (fr, fg, fb, d, preserve) = (
            self.color_r,
            self.color_g,
            self.color_b,
            self.density,
            self.preserve_luminosity,
        );
        Clut3D::from_fn(33, move |r, g, b| {
            let mut nr = r + (fr - r) * d;
            let mut ng = g + (fg - g) * d;
            let mut nb = b + (fb - b) * d;
            if preserve {
                let orig_luma = luminance(r, g, b);
                let new_luma = luminance(nr, ng, nb);
                if new_luma > 1e-7 {
                    let scale = orig_luma / new_luma;
                    nr *= scale;
                    ng *= scale;
                    nb *= scale;
                }
            }
            (nr, ng, nb)
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
    fn photo_filter_zero_density_is_identity() {
        let input = vec![0.3, 0.5, 0.7, 1.0];
        let f = PhotoFilter {
            color_r: 1.0,
            color_g: 0.0,
            color_b: 0.0,
            density: 0.0,
            preserve_luminosity: false,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "photo_filter 0 density");
    }
}
