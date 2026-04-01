//! Filter: blend (category: composite)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Blend two same-size RGB8 or RGBA8 images using the given blend mode.
///
/// `fg` is the "top" layer, `bg` is the "bottom" layer.
/// Both must have the same format and dimensions.
/// For RGBA8, alpha is preserved from `bg` (bottom layer).
#[rasmcore_macros::register_compositor(
    name = "blend",
    category = "composite",
    group = "composite",
    variant = "blend",
    reference = "27-mode photographic blend (multiply, screen, overlay, hue, saturation, etc.)"
)]
pub fn blend(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    bg_pixels: &[u8],
    bg_info: &ImageInfo,
    mode: BlendMode,
) -> Result<Vec<u8>, ImageError> {
    if fg_info.format != bg_info.format {
        return Err(ImageError::InvalidInput("format mismatch".into()));
    }
    if fg_info.width != bg_info.width || fg_info.height != bg_info.height {
        return Err(ImageError::InvalidInput("dimension mismatch".into()));
    }
    validate_format(fg_info.format)?;

    let bpp = match fg_info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "blend requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let pixel_level = matches!(
        mode,
        BlendMode::Dissolve
            | BlendMode::DarkerColor
            | BlendMode::LighterColor
            | BlendMode::Hue
            | BlendMode::Saturation
            | BlendMode::Color
            | BlendMode::Luminosity
    );

    let mut result = bg_pixels.to_vec();
    if pixel_level {
        for (px_idx, (fg_chunk, bg_chunk)) in fg_pixels
            .chunks_exact(bpp)
            .zip(result.chunks_exact_mut(bpp))
            .enumerate()
        {
            let (r, g, b) = blend_pixel(fg_chunk, bg_chunk, mode, px_idx as u32);
            bg_chunk[0] = r;
            bg_chunk[1] = g;
            bg_chunk[2] = b;
            // Alpha stays from bg (bottom layer) for RGBA8
        }
    } else {
        for (fg_chunk, bg_chunk) in fg_pixels
            .chunks_exact(bpp)
            .zip(result.chunks_exact_mut(bpp))
        {
            bg_chunk[0] = blend_channel(fg_chunk[0], bg_chunk[0], mode);
            bg_chunk[1] = blend_channel(fg_chunk[1], bg_chunk[1], mode);
            bg_chunk[2] = blend_channel(fg_chunk[2], bg_chunk[2], mode);
            // Alpha stays from bg (bottom layer) for RGBA8
        }
    }
    Ok(result)
}
