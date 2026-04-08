use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Dodge — brighten shadows.
///
/// `output = input / (1 - amount)` (simplified dodge)
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dodge", category = "adjustment", cost = "O(n)")]
pub struct Dodge {
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub amount: f32,
}

impl Dodge {
    pub fn to_lmt(&self) -> Lmt {
        let divisor = (1.0 - self.amount).max(1e-6);
        lmt::analytical_uniform(PointOpExpr::Div(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(divisor)),
        ))
    }
}

impl Filter for Dodge {
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
    use crate::filters::adjustment::Burn;

    #[test]
    fn dodge_burn_are_inverses_at_half() {
        let dodge = Dodge { amount: 0.5 };
        let _burn = Burn { amount: 0.5 };
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let dodged = dodge.compute(&input, 1, 1).unwrap();
        // dodge: 0.5 / (1-0.5) = 1.0
        assert!((dodged[0] - 1.0).abs() < 1e-5);
    }
}
