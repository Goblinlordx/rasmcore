//! Filter: gradient_radial (category: generator)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a radial gradient image between two colors from center outward.
///
/// IM equivalent: `magick -size WxH radial-gradient:color1-color2`
#[rasmcore_macros::register_generator(
    name = "gradient_radial",
    category = "generator",
    group = "gradient",
    variant = "radial",
    reference = "radial interpolation from center to edges"
)]
pub fn gradient_radial(
    width: u32,
    height: u32,
    color1_r: u32,
    color1_g: u32,
    color1_b: u32,
    color2_r: u32,
    color2_g: u32,
    color2_b: u32,
    center_x: f32,
    center_y: f32,
) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let mut pixels = vec![0u8; w * h * 3];

    let c1 = [
        color1_r.min(255) as f32,
        color1_g.min(255) as f32,
        color1_b.min(255) as f32,
    ];
    let c2 = [
        color2_r.min(255) as f32,
        color2_g.min(255) as f32,
        color2_b.min(255) as f32,
    ];

    let cx = center_x * w as f32;
    let cy = center_y * h as f32;
    // Max radius: distance from center to farthest corner
    let max_r = [
        (cx * cx + cy * cy).sqrt(),
        ((w as f32 - cx).powi(2) + cy * cy).sqrt(),
        (cx * cx + (h as f32 - cy).powi(2)).sqrt(),
        ((w as f32 - cx).powi(2) + (h as f32 - cy).powi(2)).sqrt(),
    ]
    .iter()
    .cloned()
    .fold(0.0f32, f32::max)
    .max(1.0);

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let t = ((dx * dx + dy * dy).sqrt() / max_r).clamp(0.0, 1.0);
            let idx = (y * w + x) * 3;
            pixels[idx] = (c1[0] + (c2[0] - c1[0]) * t + 0.5) as u8;
            pixels[idx + 1] = (c1[1] + (c2[1] - c1[1]) * t + 0.5) as u8;
            pixels[idx + 2] = (c1[2] + (c2[2] - c1[2]) * t + 0.5) as u8;
        }
    }
    pixels
}
