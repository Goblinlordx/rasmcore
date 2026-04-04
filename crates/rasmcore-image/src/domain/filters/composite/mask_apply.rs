//! Filter: mask_apply (category: composite)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply a grayscale mask to the image's alpha channel.
///
/// The mask's luminance values become the output alpha. White = fully opaque,
/// black = fully transparent. If the image is RGB8, it's promoted to RGBA8.
/// Mask is resized to match image dimensions if they differ.
///
/// IM equivalent: `magick image mask -compose CopyOpacity -composite`
#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Mask apply parameters.
pub struct MaskApplyParams {
    /// Invert mask (0 = normal, 1 = inverted)
    #[param(min = 0, max = 1, step = 1, default = 0, hint = "rc.toggle")]
    pub invert: u32,
}

#[rasmcore_macros::register_compositor(
    name = "mask_apply",
    category = "composite",
    group = "composite",
    variant = "mask",
    reference = "alpha mask application via luminance"
)]
pub fn mask_apply(
    pixels: &[u8],
    info: &ImageInfo,
    mask_data: &[u8],
    mask_width: u32,
    mask_height: u32,
    invert: u32,
) -> Result<Vec<u8>, ImageError> {
    // Ensure image is RGBA
    let (mut rgba, out_info) = if info.format == PixelFormat::Rgb8 {
        let (r, i) = add_alpha(pixels, info, 255)?;
        (r, i)
    } else if info.format == PixelFormat::Rgba8 {
        (pixels.to_vec(), info.clone())
    } else {
        return Err(ImageError::UnsupportedFormat(
            "mask_apply requires RGB8 or RGBA8".into(),
        ));
    };

    let w = out_info.width as usize;
    let h = out_info.height as usize;
    let mw = mask_width as usize;
    let mh = mask_height as usize;
    let invert_mask = invert != 0;

    // Determine mask bytes-per-pixel (Gray8 = 1, RGB8 = 3, RGBA8 = 4)
    let mask_bpp = if mask_data.len() == mw * mh {
        1
    } else if mask_data.len() == mw * mh * 3 {
        3
    } else if mask_data.len() == mw * mh * 4 {
        4
    } else {
        return Err(ImageError::InvalidInput(format!(
            "mask data length {} doesn't match {}x{} at any known format",
            mask_data.len(),
            mw,
            mh
        )));
    };

    for y in 0..h {
        for x in 0..w {
            // Map to mask coordinates (nearest-neighbor resize)
            let mx = (x * mw / w).min(mw - 1);
            let my = (y * mh / h).min(mh - 1);
            let mi = my * mw + mx;

            let gray = match mask_bpp {
                1 => mask_data[mi],
                3 => {
                    let base = mi * 3;
                    // BT.709 luminance
                    let r = mask_data[base] as u32;
                    let g = mask_data[base + 1] as u32;
                    let b = mask_data[base + 2] as u32;
                    ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
                }
                4 => {
                    let base = mi * 4;
                    let r = mask_data[base] as u32;
                    let g = mask_data[base + 1] as u32;
                    let b = mask_data[base + 2] as u32;
                    ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
                }
                _ => 0,
            };

            let alpha = if invert_mask { 255 - gray } else { gray };
            rgba[(y * w + x) * 4 + 3] = alpha;
        }
    }

    Ok(rgba)
}
