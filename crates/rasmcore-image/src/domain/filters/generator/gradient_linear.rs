//! Filter: gradient_linear (category: generator)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a linear gradient image between two colors at a given angle.
///
/// Angle 0 = left-to-right, 90 = top-to-bottom, etc.
/// IM equivalent: `magick -size WxH gradient:color1-color2 -rotate angle`
#[rasmcore_macros::register_generator(
    name = "gradient_linear",
    category = "generator",
    group = "gradient",
    variant = "linear",
    reference = "linear interpolation between two endpoint colors"
)]
pub fn gradient_linear(
    width: u32,
    height: u32,
    color1_r: u32,
    color1_g: u32,
    color1_b: u32,
    color2_r: u32,
    color2_g: u32,
    color2_b: u32,
    angle: f32,
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

    let rad = angle.to_radians();
    let dx = rad.cos();
    let dy = rad.sin();
    // Project image corners onto gradient axis to find extent
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let half_extent = (cx * dx.abs() + cy * dy.abs()).max(1.0);

    for y in 0..h {
        for x in 0..w {
            let proj = (x as f32 - cx) * dx + (y as f32 - cy) * dy;
            let t = ((proj / half_extent) * 0.5 + 0.5).clamp(0.0, 1.0);
            let idx = (y * w + x) * 3;
            pixels[idx] = (c1[0] + (c2[0] - c1[0]) * t + 0.5) as u8;
            pixels[idx + 1] = (c1[1] + (c2[1] - c1[1]) * t + 0.5) as u8;
            pixels[idx + 2] = (c1[2] + (c2[2] - c1[2]) * t + 0.5) as u8;
        }
    }
    pixels
}
