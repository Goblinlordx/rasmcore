//! Mapper: mask_color_range (category: mask)
//!
//! Generate a mask from image color — isolate pixels by hue and saturation.

use crate::domain::color_grading::rgb_to_hsl;
#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a mask from image color range.
///
/// Pixels matching the target hue (within `hue_range`) and above `sat_min`
/// are white; others are black. `feather` controls edge softness.
#[rasmcore_macros::register_mapper(
    name = "mask_color_range",
    category = "mask",
    reference = "hue/saturation-based color range mask",
    output_format = "Gray8"
)]
pub fn mask_color_range(
    pixels: &[u8],
    info: &ImageInfo,
    target_hue: f32,
    hue_range: f32,
    sat_min: f32,
    feather: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "mask_color_range requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let n = (info.width as usize) * (info.height as usize);
    let half_range = hue_range.clamp(1.0, 180.0) / 2.0;
    let feath = feather.clamp(0.0, 90.0);
    let sat_threshold = sat_min.clamp(0.0, 1.0);

    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f32 / 255.0;
        let g = pixels[pi + 1] as f32 / 255.0;
        let b = pixels[pi + 2] as f32 / 255.0;

        let (h, s, _l) = rgb_to_hsl(r, g, b);

        // Hue distance (wrapping)
        let hue_diff = ((h - target_hue + 180.0).rem_euclid(360.0)) - 180.0;
        let hue_dist = hue_diff.abs();

        // Saturation gate
        if s < sat_threshold {
            mask.push(0);
            continue;
        }

        let v = if feath > 0.0 {
            let outer = half_range + feath;
            if hue_dist <= half_range {
                255
            } else if hue_dist >= outer {
                0
            } else {
                let t = (hue_dist - half_range) / feath;
                let smooth = 0.5 * (1.0 + (t * std::f32::consts::PI).cos());
                (smooth * 255.0 + 0.5) as u8
            }
        } else {
            if hue_dist <= half_range {
                255
            } else {
                0
            }
        };
        mask.push(v);
    }

    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((mask, out_info))
}
