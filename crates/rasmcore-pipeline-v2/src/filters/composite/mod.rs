//! Alpha and compositing filters — premultiply, unpremultiply, blend modes, blend-if.
//!
//! All filters operate on f32 RGBA data. GPU shaders provided for all.
//!
//! ## Blend mode formulas
//!
//! Separable modes (per-channel) follow **ISO 32000-2:2020 Section 11.3.5**
//! (PDF 2.0 specification), which is the canonical source. The W3C Compositing
//! and Blending Level 1 spec and Adobe Photoshop both derive from it.
//!
//! Non-separable modes (hue, saturation, color, luminosity) use the
//! **SetLum/SetSat/ClipColor** helpers from ISO 32000-2 Section 11.3.5.4.
//! Luminance coefficients are derived from the working color space's primaries
//! via `ColorSpace::luma_coefficients()` — NOT hardcoded.

pub mod blend;
pub mod blend_if;
pub mod premultiply;
pub mod unpremultiply;

pub use blend::Blend;
pub use blend_if::BlendIf;
pub use premultiply::Premultiply;
pub use unpremultiply::Unpremultiply;

// ─── ISO 32000-2 compositing helpers (non-separable blend modes) ───────────
//
// These implement the SetLum/SetSat/ClipColor functions from
// ISO 32000-2:2020 Section 11.3.5.4. Luminance uses color-space-derived
// coefficients passed as a parameter, NOT hardcoded constants.

/// Luminance from color-space-derived coefficients (Y row of RGB->XYZ matrix).
fn luma(r: f32, g: f32, b: f32, coeffs: [f32; 3]) -> f32 {
    coeffs[0] * r + coeffs[1] * g + coeffs[2] * b
}

/// ISO 32000-2 ClipColor — clamp color to [0,1] while preserving luminance.
fn clip_color(r: f32, g: f32, b: f32, coeffs: [f32; 3]) -> (f32, f32, f32) {
    let l = luma(r, g, b, coeffs);
    let n = r.min(g).min(b);
    let (mut ro, mut go, mut bo) = (r, g, b);
    if n < 0.0 {
        let d = (l - n).max(1e-7);
        ro = l + (ro - l) * l / d;
        go = l + (go - l) * l / d;
        bo = l + (bo - l) * l / d;
    }
    let x2 = ro.max(go).max(bo);
    if x2 > 1.0 {
        let l2 = luma(ro, go, bo, coeffs);
        let d2 = (x2 - l2).max(1e-7);
        ro = l2 + (ro - l2) * (1.0 - l2) / d2;
        go = l2 + (go - l2) * (1.0 - l2) / d2;
        bo = l2 + (bo - l2) * (1.0 - l2) / d2;
    }
    (ro, go, bo)
}

/// ISO 32000-2 SetLum — adjust color to target luminance.
fn set_lum(r: f32, g: f32, b: f32, l: f32, coeffs: [f32; 3]) -> (f32, f32, f32) {
    let d = l - luma(r, g, b, coeffs);
    clip_color(r + d, g + d, b + d, coeffs)
}

/// ISO 32000-2 Sat — chroma range of a color.
fn sat(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

/// ISO 32000-2 SetSat — scale color to target saturation.
fn set_sat(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
    let cmin = r.min(g).min(b);
    let cmax = r.max(g).max(b);
    let range = cmax - cmin;
    if range < 1e-7 {
        return (0.0, 0.0, 0.0);
    }
    let scale = s / range;
    ((r - cmin) * scale, (g - cmin) * scale, (b - cmin) * scale)
}

/// ISO 32000-2 D(x) helper for SoftLight blend mode (Section 11.3.5).
fn soft_light_d(x: f32) -> f32 {
    if x <= 0.25 {
        ((16.0 * x - 12.0) * x + 4.0) * x
    } else {
        x.sqrt()
    }
}

// Simple hash for dissolve mode.
fn pcg_hash(v: u32) -> u32 {
    let mut x = v.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    (x >> 22) ^ x
}

#[cfg(test)]
mod tests {
    #[test]
    fn filters_registered() {
        let names: Vec<&str> = crate::registered_operations()
            .into_iter()
            .filter(|op| op.category == "composite")
            .map(|op| op.name)
            .collect();
        assert!(names.contains(&"premultiply"));
        assert!(names.contains(&"unpremultiply"));
        assert!(names.contains(&"blend"));
        assert!(names.contains(&"blend_if"));
    }
}
