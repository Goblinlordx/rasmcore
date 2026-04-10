use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Brightness adjustment — additive offset in perceptual (log) space.
///
/// Declares preferred_color_space = ACEScct. The fusion optimizer auto-inserts
/// upstream_cs→ACEScct before and ACEScct→upstream_cs after, so the additive
/// offset operates in log space — perceptually uniform, matching consumer
/// editors (GIMP, Photoshop legacy, phone apps).
///
/// For physically-accurate exposure control in linear light, use `Exposure`.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "brightness",
    category = "adjustment",
    cost = "O(n)",
    doc = "docs/operations/filters/adjustment/brightness.adoc"
)]
pub struct Brightness {
    /// Additive offset applied to each RGB channel (in ACEScct log space).
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
    pub amount: f32,
}

impl Brightness {
    pub fn to_lmt(&self) -> Lmt {
        lmt::analytical_uniform(PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.amount)),
        ))
    }
}

impl Filter for Brightness {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.to_lmt().apply(input))
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.to_lmt().to_analytical()
    }

    fn preferred_color_space(&self) -> Option<crate::color_space::ColorSpace> {
        Some(crate::color_space::ColorSpace::AcesCct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::adjustment::tests::test_pixels;

    #[test]
    fn brightness_adds_offset() {
        let input = test_pixels();
        let b = Brightness { amount: 0.1 };
        let out = b.compute(&input, 2, 2).unwrap();
        assert!((out[0] - 0.1).abs() < 1e-6); // 0.0 + 0.1
        assert!((out[1] - 0.35).abs() < 1e-6); // 0.25 + 0.1
        assert_eq!(out[3], 1.0); // alpha unchanged
    }

    #[test]
    fn brightness_expression_matches_compute() {
        let b = Brightness { amount: 0.2 };
        let expr = b.analytic_expression_per_channel().unwrap();
        let input = test_pixels();
        let computed = b.compute(&input, 2, 2).unwrap();
        for i in (0..input.len()).step_by(4) {
            for c in 0..3 {
                let from_expr = expr[c].evaluate(input[i + c] as f64) as f32;
                assert!(
                    (from_expr - computed[i + c]).abs() < 1e-5,
                    "mismatch at pixel {}, channel {}: expr={from_expr} compute={}",
                    i / 4,
                    c,
                    computed[i + c]
                );
            }
        }
    }
}
