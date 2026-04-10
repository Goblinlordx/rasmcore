use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::{ClutOp, lab_to_rgb, rgb_to_lab};

/// Lab adjust — shift a* and b* channels in CIE Lab space.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "lab_adjust", category = "color", cost = "O(n)")]
pub struct LabAdjust {
    /// Green-red shift (-128 to 127).
    #[param(min = -128.0, max = 127.0, default = 0.0)]
    pub a_offset: f32,
    /// Blue-yellow shift (-128 to 127).
    #[param(min = -128.0, max = 127.0, default = 0.0)]
    pub b_offset: f32,
}

impl Filter for LabAdjust {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (ao, bo) = (self.a_offset, self.b_offset);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (l, a, b) = rgb_to_lab(pixel[0], pixel[1], pixel[2]);
            let na = (a + ao).clamp(-128.0, 127.0);
            let nb = (b + bo).clamp(-128.0, 127.0);
            let (r, g, bi) = lab_to_rgb(l, na, nb);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = bi;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for LabAdjust {
    fn build_clut(&self) -> Clut3D {
        let (ao, bo) = (self.a_offset, self.b_offset);
        Clut3D::from_fn(33, move |r, g, b| {
            let (l, a, bi) = rgb_to_lab(r, g, b);
            let na = (a + ao).clamp(-128.0, 127.0);
            let nb = (bi + bo).clamp(-128.0, 127.0);
            lab_to_rgb(l, na, nb)
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
    fn lab_adjust_identity() {
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let f = LabAdjust {
            a_offset: 0.0,
            b_offset: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.02, "lab_adjust identity");
    }
}
