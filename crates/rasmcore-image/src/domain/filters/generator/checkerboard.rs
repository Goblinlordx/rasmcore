//! Filter: checkerboard (category: generator)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a checkerboard pattern image.
///
/// IM equivalent: `magick -size WxH pattern:checkerboard`
#[rasmcore_macros::register_generator(
    name = "checkerboard",
    category = "generator",
    reference = "alternating two-color grid pattern"
)]
pub fn checkerboard(
    width: u32,
    height: u32,
    cell_size: u32,
    color1_r: u32,
    color1_g: u32,
    color1_b: u32,
    color2_r: u32,
    color2_g: u32,
    color2_b: u32,
) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let cell = cell_size.max(1) as usize;
    let mut pixels = vec![0u8; w * h * 3];

    let c1 = [
        color1_r.min(255) as u8,
        color1_g.min(255) as u8,
        color1_b.min(255) as u8,
    ];
    let c2 = [
        color2_r.min(255) as u8,
        color2_g.min(255) as u8,
        color2_b.min(255) as u8,
    ];

    for y in 0..h {
        for x in 0..w {
            let color = if ((x / cell) + (y / cell)).is_multiple_of(2) {
                &c1
            } else {
                &c2
            };
            let idx = (y * w + x) * 3;
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
        }
    }
    pixels
}
