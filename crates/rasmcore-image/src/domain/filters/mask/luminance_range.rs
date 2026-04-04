//! Mapper: mask_luminance_range (category: mask)
//!
//! Generate a mask from image luminance — isolate highlights, shadows, or midtones.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate a mask from image luminance.
///
/// Pixels with luminance between `black` and `white` are white in the mask;
/// pixels outside are black. `feather` controls the softness of the transition.
///
/// BT.709 luminance: Y = 0.2126*R + 0.7152*G + 0.0722*B
#[rasmcore_macros::register_mapper(
    name = "mask_luminance_range",
    category = "mask",
    reference = "luminance-based mask with range and feather",
    output_format = "Gray8"
)]
pub fn mask_luminance_range(
    pixels: &[u8],
    info: &ImageInfo,
    black: f32,
    white: f32,
    feather: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "mask_luminance_range requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let n = (info.width as usize) * (info.height as usize);
    let black_val = black.clamp(0.0, 1.0) * 255.0;
    let white_val = (white.clamp(0.0, 1.0) * 255.0).max(black_val);
    let feath = feather.clamp(0.0, 128.0);

    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f32;
        let g = pixels[pi + 1] as f32;
        let b = pixels[pi + 2] as f32;
        let lum = r * 0.2126 + g * 0.7152 + b * 0.0722;

        let v = if feath > 0.0 {
            // Smooth ramp at both boundaries
            let lo = smoothstep_ramp(lum, black_val - feath, black_val);
            let hi = smoothstep_ramp(white_val + feath - lum, 0.0, feath);
            (lo * hi * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        } else {
            if lum >= black_val && lum <= white_val {
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

/// Smooth ramp: 0 when val <= edge0, 1 when val >= edge1, smooth in between.
#[inline]
fn smoothstep_ramp(val: f32, edge0: f32, edge1: f32) -> f32 {
    if edge1 <= edge0 {
        return if val >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((val - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t) // Hermite smoothstep
}
