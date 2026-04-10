//! Mask operation filters — alpha channel generation, manipulation, and blending.
//!
//! All filters operate on f32 RGBA. Mask operations primarily affect the alpha
//! channel or use it for blending. GPU shaders are per-pixel (no neighborhood).

mod color_range;
mod feather;
mod gradient_mask;
mod luminance_range;
mod mask_apply;
mod mask_combine;
mod masked_blend;

pub use color_range::ColorRange;
pub use feather::Feather;
pub use gradient_mask::GradientMask;
pub use luminance_range::LuminanceRange;
pub use mask_apply::MaskApply;
pub use mask_combine::MaskCombine;
pub use masked_blend::MaskedBlend;

// ─── Helpers ───────────────────────────────────────────────────────────────

#[inline(always)]
pub(crate) fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    #[test]
    fn all_mask_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &[
            "color_range",
            "luminance_range",
            "feather",
            "gradient_mask",
            "mask_apply",
            "masked_blend",
            "mask_combine",
        ] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn mask_apply_premultiplies() {
        let input = vec![0.8, 0.6, 0.4, 0.5, 1.0, 1.0, 1.0, 0.0];
        let f = MaskApply;
        let out = f.compute(&input, 2, 1).unwrap();
        assert!((out[0] - 0.4).abs() < 0.001); // 0.8 * 0.5
        assert!((out[4] - 0.0).abs() < 0.001); // 1.0 * 0.0
    }

    #[test]
    fn luminance_range_creates_mask() {
        let input = vec![0.1, 0.1, 0.1, 1.0, 0.9, 0.9, 0.9, 1.0];
        let f = LuminanceRange {
            low: 0.3,
            high: 0.7,
            softness: 0.0,
        };
        let out = f.compute(&input, 2, 1).unwrap();
        // Dark pixel (luma ~0.1) → outside range → alpha ≈ 0
        assert!(out[3] < 0.1);
        // Bright pixel (luma ~0.9) → outside range → alpha ≈ 0
        assert!(out[7] < 0.1);
    }
}
