//! Porter-Duff "over" alpha compositing — dual-input compositor.
//!
//! Composites foreground (A) over background (B) using premultiplied alpha blending.
//! Both inputs are f32 RGBA with straight (non-premultiplied) alpha.
//!
//! Formula (premultiplied internally for SIMD-friendly math):
//!   out_a = fg_a + bg_a * (1 - fg_a)
//!   out_rgb = fg_rgb * fg_a + bg_rgb * bg_a * (1 - fg_a)
//!   out_rgb /= out_a  (un-premultiply for straight alpha output)
//!
//! Reference: Porter & Duff, "Compositing Digital Images" (1984), SIGGRAPH.

use crate::node::PipelineError;
use crate::ops::Compositor;

/// Porter-Duff "over" alpha compositing.
///
/// Composites foreground over background using standard alpha blending.
/// `opacity` controls the effective foreground opacity (0 = bg only, 1 = normal composite).
#[derive(Clone)]
pub struct PorterDuffOver {
    /// Effective foreground opacity multiplier (0.0 = transparent fg, 1.0 = normal).
    pub opacity: f32,
}

impl Compositor for PorterDuffOver {
    fn compute(
        &self,
        fg: &[f32],
        bg: &[f32],
        _w: u32,
        _h: u32,
    ) -> Result<Vec<f32>, PipelineError> {
        let mut out = Vec::with_capacity(fg.len());
        let opacity = self.opacity;

        for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
            let fg_a = fg_px[3] * opacity;
            let bg_a = bg_px[3];
            let inv_fg_a = 1.0 - fg_a;

            let out_a = fg_a + bg_a * inv_fg_a;

            if out_a > 1e-7 {
                let inv_out_a = 1.0 / out_a;
                out.push((fg_px[0] * fg_a + bg_px[0] * bg_a * inv_fg_a) * inv_out_a);
                out.push((fg_px[1] * fg_a + bg_px[1] * bg_a * inv_fg_a) * inv_out_a);
                out.push((fg_px[2] * fg_a + bg_px[2] * bg_a * inv_fg_a) * inv_out_a);
            } else {
                out.push(0.0);
                out.push(0.0);
                out.push(0.0);
            }
            out.push(out_a);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn opaque_fg_covers_bg() {
        let fg = px(1.0, 0.0, 0.0, 1.0); // opaque red
        let bg = px(0.0, 0.0, 1.0, 1.0); // opaque blue
        let comp = PorterDuffOver { opacity: 1.0 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5, "R: {}", out[0]);
        assert!(out[1].abs() < 1e-5, "G: {}", out[1]);
        assert!(out[2].abs() < 1e-5, "B: {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "A: {}", out[3]);
    }

    #[test]
    fn transparent_fg_shows_bg() {
        let fg = px(1.0, 0.0, 0.0, 0.0); // transparent red
        let bg = px(0.0, 0.0, 1.0, 1.0); // opaque blue
        let comp = PorterDuffOver { opacity: 1.0 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!(out[0].abs() < 1e-5, "R: {}", out[0]);
        assert!(out[1].abs() < 1e-5, "G: {}", out[1]);
        assert!((out[2] - 1.0).abs() < 1e-5, "B: {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "A: {}", out[3]);
    }

    #[test]
    fn half_alpha_blends() {
        let fg = px(1.0, 0.0, 0.0, 0.5); // 50% red
        let bg = px(0.0, 0.0, 1.0, 1.0); // opaque blue
        let comp = PorterDuffOver { opacity: 1.0 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // out_a = 0.5 + 1.0 * 0.5 = 1.0
        // out_r = (1.0 * 0.5 + 0.0 * 1.0 * 0.5) / 1.0 = 0.5
        // out_b = (0.0 * 0.5 + 1.0 * 1.0 * 0.5) / 1.0 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5, "R: {}", out[0]);
        assert!((out[2] - 0.5).abs() < 1e-5, "B: {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "A: {}", out[3]);
    }

    #[test]
    fn opacity_zero_returns_bg() {
        let fg = px(1.0, 0.0, 0.0, 1.0);
        let bg = px(0.0, 1.0, 0.0, 1.0);
        let comp = PorterDuffOver { opacity: 0.0 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn both_transparent_yields_transparent() {
        let fg = px(1.0, 0.0, 0.0, 0.0);
        let bg = px(0.0, 0.0, 1.0, 0.0);
        let comp = PorterDuffOver { opacity: 1.0 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        assert!(out[3].abs() < 1e-5, "A should be 0: {}", out[3]);
    }

    #[test]
    fn opacity_half_attenuates_fg() {
        let fg = px(1.0, 0.0, 0.0, 1.0); // opaque red
        let bg = px(0.0, 0.0, 1.0, 1.0); // opaque blue
        let comp = PorterDuffOver { opacity: 0.5 };
        let out = comp.compute(&fg, &bg, 1, 1).unwrap();
        // effective fg_a = 0.5, bg mixes in at 0.5
        // out_a = 0.5 + 1.0 * 0.5 = 1.0
        // out_r = (1.0 * 0.5) / 1.0 = 0.5
        // out_b = (1.0 * 1.0 * 0.5) / 1.0 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5, "R: {}", out[0]);
        assert!((out[2] - 0.5).abs() < 1e-5, "B: {}", out[2]);
    }
}
