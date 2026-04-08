use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Exposure adjustment — pure EV stops.
///
/// `output = input * 2^ev`
///
/// Professional-grade exposure control matching camera EV behavior:
/// +1 EV doubles brightness, -1 EV halves it. No offset, no gamma —
/// those are separate filters (Brightness for offset, Gamma for curve).
/// ev=0 is exact identity. The expression `Mul(Input, Constant)` is
/// trivially fusable with any other point op in the pipeline.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "exposure", category = "adjustment", cost = "O(n)", doc = "docs/operations/filters/adjustment/exposure.adoc")]
pub struct Exposure {
    /// Exposure value in stops. 0 = unchanged, +1 = 2x brighter, -1 = half.
    #[param(min = -10.0, max = 10.0, step = 0.1, default = 0.0)]
    pub ev: f32,
}

impl Exposure {
    pub fn to_lmt(&self) -> Lmt {
        let multiplier = 2.0f32.powf(self.ev);
        lmt::analytical_uniform(PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(multiplier)),
        ))
    }
}

impl Filter for Exposure {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.to_lmt().apply(input))
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.to_lmt().to_analytical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::adjustment::tests::test_pixels;

    #[test]
    fn exposure_ev_zero_is_exact_identity() {
        let input = test_pixels();
        let exp = Exposure { ev: 0.0 };
        let out = exp.compute(&input, 2, 2).unwrap();
        assert_eq!(out, input, "ev=0 must be pixel-exact identity");
    }

    #[test]
    fn exposure_ev_plus_one_doubles() {
        let input = vec![0.25, 0.5, 0.125, 1.0];
        let exp = Exposure { ev: 1.0 };
        let out = exp.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);   // 0.25 * 2
        assert!((out[1] - 1.0).abs() < 1e-6);   // 0.5 * 2
        assert!((out[2] - 0.25).abs() < 1e-6);  // 0.125 * 2
        assert_eq!(out[3], 1.0);                  // alpha unchanged
    }

    #[test]
    fn exposure_ev_minus_one_halves() {
        let input = vec![0.5, 1.0, 0.25, 1.0];
        let exp = Exposure { ev: -1.0 };
        let out = exp.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.25).abs() < 1e-6);  // 0.5 / 2
        assert!((out[1] - 0.5).abs() < 1e-6);   // 1.0 / 2
        assert!((out[2] - 0.125).abs() < 1e-6); // 0.25 / 2
    }

    #[test]
    fn exposure_smooth_transition() {
        // Verify no discontinuities: output should be monotonically increasing
        // and the ratio between adjacent steps should be constant (2^0.1 ~ 1.072)
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let mut prev = 0.0f32;
        for i in -20..=20 {
            let ev = i as f32 * 0.1;
            let exp = Exposure { ev };
            let out = exp.compute(&input, 1, 1).unwrap();
            if i > -20 {
                assert!(out[0] > prev, "output must increase with ev at ev={ev}");
                // Ratio should be ~ 2^0.1 ~ 1.072 for each 0.1 EV step
                let ratio = out[0] / prev;
                assert!(
                    (ratio - 1.0717734).abs() < 0.01,
                    "non-smooth ratio at ev={ev}: {ratio}"
                );
            }
            prev = out[0];
        }
    }

    #[test]
    fn exposure_expression_matches_compute() {
        let exp = Exposure { ev: 1.5 };
        let expr = exp.analytic_expression_per_channel().unwrap();
        let input = test_pixels();
        let computed = exp.compute(&input, 2, 2).unwrap();
        for i in (0..input.len()).step_by(4) {
            for c in 0..3 {
                let from_expr = expr[c].evaluate(input[i + c] as f64) as f32;
                assert!(
                    (from_expr - computed[i + c]).abs() < 1e-5,
                    "mismatch at pixel {}, channel {c}",
                    i / 4
                );
            }
        }
    }

    #[test]
    fn exposure_expression_is_simple_mul() {
        // Verify the expression is just Mul(Input, Constant) — trivially fusable
        let exp = Exposure { ev: 2.0 };
        let exprs = exp.analytic_expression_per_channel().unwrap();
        match &exprs[0] {
            PointOpExpr::Mul(lhs, rhs) => {
                assert!(matches!(**lhs, PointOpExpr::Input));
                if let PointOpExpr::Constant(c) = **rhs {
                    assert!((c - 4.0).abs() < 1e-6, "2^2 = 4.0, got {c}");
                } else {
                    panic!("expected Constant, got {:?}", rhs);
                }
            }
            other => panic!("expected Mul(Input, Constant), got {:?}", other),
        }
    }
}
