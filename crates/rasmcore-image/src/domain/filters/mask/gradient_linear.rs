//! Generator: mask_gradient_linear (category: mask)
//!
//! Linear gradient mask — grayscale ramp at a given angle with feathered edges.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a linear gradient mask.
///
/// Output is RGB8 with R=G=B=gray (compatible with mask_apply luminance extraction).
/// `position` controls the center of the transition (0.0 = start, 1.0 = end).
/// `feather` controls the width of the transition (0.0 = hard edge, 1.0 = full span).
#[rasmcore_macros::register_generator(
    name = "mask_gradient_linear",
    category = "mask",
    group = "mask",
    variant = "gradient_linear",
    reference = "linear gradient mask with angle and feather"
)]
pub fn mask_gradient_linear(
    width: u32,
    height: u32,
    angle: f32,
    position: f32,
    feather: f32,
) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let mut pixels = vec![0u8; w * h * 3];

    let rad = angle.to_radians();
    let dx = rad.cos();
    let dy = rad.sin();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let half_extent = (cx * dx.abs() + cy * dy.abs()).max(1.0);

    let pos = position.clamp(0.0, 1.0);
    let feath = feather.clamp(0.01, 1.0);
    let half_feath = feath / 2.0;

    for y in 0..h {
        for x in 0..w {
            let proj = (x as f32 - cx) * dx + (y as f32 - cy) * dy;
            let t = proj / half_extent * 0.5 + 0.5; // [0, 1] along gradient axis

            // Smoothstep transition centered at `position` with `feather` width
            let gray = if half_feath > 0.0 {
                let norm = (t - pos) / half_feath;
                (norm * 0.5 + 0.5).clamp(0.0, 1.0)
            } else {
                if t >= pos {
                    1.0
                } else {
                    0.0
                }
            };

            let v = (gray * 255.0 + 0.5) as u8;
            let idx = (y * w + x) * 3;
            pixels[idx] = v;
            pixels[idx + 1] = v;
            pixels[idx + 2] = v;
        }
    }
    pixels
}
