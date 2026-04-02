//! Generator: mask_gradient_radial (category: mask)
//!
//! Radial gradient mask — elliptical falloff from center.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a radial gradient mask.
///
/// Output is RGB8 with R=G=B=gray. White at center, black at edges.
/// `cx`, `cy` are center coordinates as fractions [0,1].
/// `rx`, `ry` are radii as fractions of width/height.
/// `feather` controls edge softness (0 = hard edge, 1 = full falloff).
#[rasmcore_macros::register_generator(
    name = "mask_gradient_radial",
    category = "mask",
    group = "mask",
    variant = "gradient_radial",
    reference = "radial/elliptical gradient mask with feather"
)]
pub fn mask_gradient_radial(
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    feather: f32,
) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let mut pixels = vec![0u8; w * h * 3];

    let center_x = cx.clamp(0.0, 1.0) * w as f32;
    let center_y = cy.clamp(0.0, 1.0) * h as f32;
    let radius_x = (rx.clamp(0.01, 2.0) * w as f32).max(1.0);
    let radius_y = (ry.clamp(0.01, 2.0) * h as f32).max(1.0);
    let feath = feather.clamp(0.0, 1.0);

    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - center_x) / radius_x;
            let dy = (y as f32 - center_y) / radius_y;
            let dist = (dx * dx + dy * dy).sqrt();

            // Inner radius = 1.0 - feather, outer = 1.0
            let inner = 1.0 - feath;
            let gray = if dist <= inner {
                1.0
            } else if dist >= 1.0 {
                0.0
            } else if feath > 0.0 {
                // Cosine interpolation for smooth falloff
                let t = (dist - inner) / feath;
                0.5 * (1.0 + (t * std::f32::consts::PI).cos())
            } else {
                if dist <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            };

            let v = (gray * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            let idx = (y * w + x) * 3;
            pixels[idx] = v;
            pixels[idx + 1] = v;
            pixels[idx + 2] = v;
        }
    }
    pixels
}
