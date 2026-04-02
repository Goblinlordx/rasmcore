//! Filter: chromatic_split (category: effect)
//!
//! Separate RGB channels and apply independent spatial offsets, then recombine.
//! Creates a "prism" or "RGB split" look popular in social/consumer apps.
//! Reference: PicsArt/Pixlr RGB split effect.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for the chromatic split effect.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ChromaticSplitParams {
    /// Red channel X offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 5.0, hint = "rc.signed_slider")]
    pub red_dx: f32,
    /// Red channel Y offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub red_dy: f32,
    /// Green channel X offset in pixels (usually 0)
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub green_dx: f32,
    /// Green channel Y offset in pixels (usually 0)
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub green_dy: f32,
    /// Blue channel X offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = -5.0, hint = "rc.signed_slider")]
    pub blue_dx: f32,
    /// Blue channel Y offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub blue_dy: f32,
}

#[rasmcore_macros::register_filter(
    name = "chromatic_split",
    category = "effect",
    reference = "PicsArt/Pixlr RGB channel split"
)]
pub fn chromatic_split(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ChromaticSplitParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            chromatic_split(r, &mut u, i8, config)
        });
    }
    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            chromatic_split(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "chromatic_split requires RGB".into(),
        ));
    }

    let offsets = [
        (config.red_dx, config.red_dy),
        (config.green_dx, config.green_dy),
        (config.blue_dx, config.blue_dy),
    ];

    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            let dst_idx = (y * w + x) * ch;

            // Sample each RGB channel from its offset position
            for c in 0..3 {
                let (dx, dy) = offsets[c];
                let sx = (x as f32 + dx).round() as isize;
                let sy = (y as f32 + dy).round() as isize;

                // Clamp to image bounds
                let sx = sx.clamp(0, w as isize - 1) as usize;
                let sy = sy.clamp(0, h as isize - 1) as usize;

                let src_idx = (sy * w + sx) * ch;
                out[dst_idx + c] = pixels[src_idx + c];
            }

            // Copy alpha if present
            if ch == 4 {
                out[dst_idx + 3] = pixels[(y * w + x) * ch + 3];
            }
        }
    }

    Ok(out)
}
