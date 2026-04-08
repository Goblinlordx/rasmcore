use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::ClutOp;

/// 3x3 channel mixing matrix in RGB space.
#[derive(Clone)]
pub struct ChannelMixer {
    /// Row-major 3x3: [rr, rg, rb, gr, gg, gb, br, bg, bb].
    pub matrix: [f32; 9],
}

impl Filter for ChannelMixer {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let m = &self.matrix;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            pixel[0] = m[0] * r + m[1] * g + m[2] * b;
            pixel[1] = m[3] * r + m[4] * g + m[5] * b;
            pixel[2] = m[6] * r + m[7] * g + m[8] * b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for ChannelMixer {
    fn build_clut(&self) -> Clut3D {
        let m = self.matrix;
        Clut3D::from_fn(33, move |r, g, b| {
            (
                m[0] * r + m[1] * g + m[2] * b,
                m[3] * r + m[4] * g + m[5] * b,
                m[6] * r + m[7] * g + m[8] * b,
            )
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
    fn channel_mixer_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = ChannelMixer {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "identity mixer");
    }

    #[test]
    fn channel_mixer_swap_rb() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = ChannelMixer {
            matrix: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.7, 0.5, 0.3), 1e-6, "swap R<->B");
    }
}
