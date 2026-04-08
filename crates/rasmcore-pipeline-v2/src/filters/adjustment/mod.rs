//! Adjustment filters — point operations on f32 pixel data.
//!
//! All implement `Filter` (f32 compute) and `analytic_expression()` (expression tree
//! for fusion optimizer). These are per-channel operations: f(v) applied
//! independently to R, G, B. Alpha is preserved unchanged.

pub mod brightness;
pub mod burn;
pub mod contrast;
pub mod dodge;
pub mod exposure;
pub mod gamma;
pub mod invert;
pub mod levels;
pub mod posterize;
pub mod sigmoidal_contrast;
pub mod solarize;

pub use brightness::Brightness;
pub use burn::Burn;
pub use contrast::Contrast;
pub use dodge::Dodge;
pub use exposure::Exposure;
pub use gamma::Gamma;
pub use invert::Invert;
pub use levels::Levels;
pub use posterize::Posterize;
pub use sigmoidal_contrast::SigmoidalContrast;
pub use solarize::Solarize;

// All adjustment filters are auto-registered via #[derive(V2Filter)] on their structs.

#[cfg(test)]
mod tests {
    use crate::ops::PointOpExpr;

    use super::*;

    /// Shared test helper: 2x2 RGBA with varied values.
    pub(crate) fn test_pixels() -> Vec<f32> {
        vec![
            0.0, 0.25, 0.5, 1.0, // pixel 0
            0.5, 0.75, 1.0, 1.0, // pixel 1
            0.1, 0.2, 0.3, 0.5, // pixel 2
            1.0, 0.0, 0.5, 1.0, // pixel 3
        ]
    }

    #[test]
    fn brightness_contrast_fusion() {
        use crate::ops::Filter;
        // Compose brightness(+0.1) -> contrast(1.5)
        let b = Brightness { amount: 0.1 };
        let c = Contrast { amount: 0.5 }; // factor = 1.5
        let b_expr = b.analytic_expression_per_channel().unwrap();
        let c_expr = c.analytic_expression_per_channel().unwrap();
        // Compose per-channel (uniform, so just test channel 0)
        let fused = PointOpExpr::compose(&c_expr[0], &b_expr[0]);

        // For v=0.5: brightness gives 0.6, contrast gives (0.6-0.5)*1.5+0.5 = 0.65
        let result = fused.evaluate(0.5);
        assert!((result - 0.65).abs() < 1e-6);
    }

    #[test]
    fn sigmoidal_solarize_fusion_generates_wgsl() {
        use crate::fusion::lower_to_wgsl;
        use crate::ops::Filter;
        let sc = SigmoidalContrast {
            strength: 3.0,
            midpoint: 0.5,
            sharpen: true,
        };
        let sol = Solarize { threshold: 0.5 };
        let sol_exprs = sol.analytic_expression_per_channel().unwrap();
        let sc_exprs = sc.analytic_expression_per_channel().unwrap();
        let composed = PointOpExpr::compose(&sol_exprs[0], &sc_exprs[0]);
        let wgsl = lower_to_wgsl(&composed);
        assert!(wgsl.contains("exp("), "Fused WGSL should contain exp()");
        assert!(wgsl.contains("select("), "Fused WGSL should contain select()");
    }
}
