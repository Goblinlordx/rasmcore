use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Sigmoidal contrast — S-curve.
///
/// Uses the sigmoidal transfer function for more natural contrast
/// than linear multiplication.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sigmoidal_contrast", category = "adjustment", cost = "O(n)")]
pub struct SigmoidalContrast {
    #[param(min = 0.0, max = 20.0, step = 0.5, default = 3.0)]
    pub strength: f32,
    /// Midpoint [0, 1]
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub midpoint: f32,
    /// true = increase contrast, false = decrease
    #[param(default = true)]
    pub sharpen: bool,
}

impl SigmoidalContrast {
    pub fn to_lmt(&self) -> Lmt {
        lmt::analytical_uniform(self.build_expr())
    }

    fn build_expr(&self) -> PointOpExpr {
        let a = self.strength;
        let b = self.midpoint;
        if a.abs() < 1e-6 {
            return PointOpExpr::Input;
        }
        if self.sharpen {
            // sig(x) = 1/(1+exp(-a*(x-b)))
            // num = sig(x) - sig(0) = 1/(1+exp(-a*(x-b))) - 1/(1+exp(a*b))
            // den = sig(1) - sig(0) = 1/(1+exp(-a*(1-b))) - 1/(1+exp(a*b))
            // out = num / den
            let sig_0 = 1.0 / (1.0 + (a * b).exp());
            let sig_1 = 1.0 / (1.0 + (-a * (1.0 - b)).exp());
            let den = sig_1 - sig_0;
            if den.abs() < 1e-10 {
                return PointOpExpr::Input;
            }
            // 1/(1+exp(-a*(v-b))) expressed as tree
            let neg_a_v_minus_b = PointOpExpr::Mul(
                Box::new(PointOpExpr::Constant(-a)),
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(b)),
                )),
            );
            let sigmoid = PointOpExpr::Div(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Add(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(PointOpExpr::Exp(Box::new(neg_a_v_minus_b))),
                )),
            );
            // (sigmoid - sig_0) / den
            PointOpExpr::Div(
                Box::new(PointOpExpr::Sub(
                    Box::new(sigmoid),
                    Box::new(PointOpExpr::Constant(sig_0)),
                )),
                Box::new(PointOpExpr::Constant(den)),
            )
        } else {
            // Inverse sigmoidal
            let sig_mid = 1.0 / (1.0 + (-a * b).exp());
            let sig_range = 1.0 / (1.0 + (-a * (1.0 - b)).exp()) - sig_mid;
            if sig_range.abs() < 1e-10 {
                return PointOpExpr::Input;
            }
            // t = sig_mid + v * sig_range
            // out = b - ln(1/t - 1) / a
            let t = PointOpExpr::Add(
                Box::new(PointOpExpr::Constant(sig_mid)),
                Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(sig_range)),
                )),
            );
            // ln(1/t - 1) = ln((1-t)/t)
            let inv_t_minus_1 = PointOpExpr::Sub(
                Box::new(PointOpExpr::Div(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(t),
                )),
                Box::new(PointOpExpr::Constant(1.0)),
            );
            PointOpExpr::Sub(
                Box::new(PointOpExpr::Constant(b)),
                Box::new(PointOpExpr::Div(
                    Box::new(PointOpExpr::Ln(Box::new(inv_t_minus_1))),
                    Box::new(PointOpExpr::Constant(a)),
                )),
            )
        }
    }
}

impl Filter for SigmoidalContrast {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.to_lmt().apply(input))
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.to_lmt().to_analytical()
    }
}

#[allow(dead_code)] // Retained for reference — compute() now uses Lmt::apply(PointOpExpr)
fn sigmoidal(v: f32, strength: f32, midpoint: f32, sharpen: bool) -> f32 {
    if strength.abs() < 1e-6 {
        return v;
    }
    if sharpen {
        let num = 1.0 / (1.0 + (-strength * (v - midpoint)).exp())
            - 1.0 / (1.0 + (strength * midpoint).exp());
        let den = 1.0 / (1.0 + (-strength * (1.0 - midpoint)).exp())
            - 1.0 / (1.0 + (strength * midpoint).exp());
        if den.abs() < 1e-10 { v } else { num / den }
    } else {
        let sig_mid = 1.0 / (1.0 + (-strength * midpoint).exp());
        let sig_range = 1.0 / (1.0 + (-strength * (1.0 - midpoint)).exp()) - sig_mid;
        if sig_range.abs() < 1e-10 {
            return v;
        }
        let t = sig_mid + v * sig_range;
        midpoint - (1.0 / t - 1.0).ln() / strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoidal_expression_matches_compute() {
        let sc = SigmoidalContrast {
            strength: 5.0,
            midpoint: 0.5,
            sharpen: true,
        };
        let expr = sc.analytic_expression_per_channel().unwrap();
        let input = vec![0.3, 0.5, 0.7, 1.0];
        let computed = sc.compute(&input, 1, 1).unwrap();
        for ch in 0..3 {
            let from_expr = expr[ch].evaluate(input[ch] as f64) as f32;
            assert!(
                (from_expr - computed[ch]).abs() < 0.01,
                "sigmoidal mismatch ch {ch}: expr={from_expr:.4} compute={:.4}",
                computed[ch]
            );
        }
    }
}
