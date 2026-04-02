//! Compositor: masked_blend (category: mask)
//!
//! Blend original and adjusted images using a grayscale mask.
//! output = adjusted * mask + original * (1 - mask)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Blend adjusted and original images using a grayscale mask.
///
/// This is the core of selective/masked adjustment:
/// `output[i] = adjusted[i] * (mask_gray/255) + original[i] * (1 - mask_gray/255)`
///
/// The mask auto-detects format (Gray8/RGB8/RGBA8) and extracts luminance.
/// Original and adjusted must have the same format and dimensions.
#[rasmcore_macros::register_compositor(
    name = "masked_blend",
    category = "mask",
    group = "mask",
    variant = "blend",
    reference = "mask-weighted blend for selective adjustments"
)]
pub fn masked_blend(
    original: &[u8],
    original_info: &ImageInfo,
    adjusted: &[u8],
    adjusted_info: &ImageInfo,
    mask_data: &[u8],
    mask_width: u32,
    mask_height: u32,
) -> Result<Vec<u8>, ImageError> {
    if original_info.format != adjusted_info.format {
        return Err(ImageError::InvalidInput(
            "original and adjusted format mismatch".into(),
        ));
    }
    if original_info.width != adjusted_info.width || original_info.height != adjusted_info.height {
        return Err(ImageError::InvalidInput(
            "original and adjusted dimension mismatch".into(),
        ));
    }

    let bpp = match original_info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "masked_blend requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let w = original_info.width as usize;
    let h = original_info.height as usize;
    let mw = mask_width as usize;
    let mh = mask_height as usize;

    // Determine mask bytes-per-pixel
    let mask_bpp = if mask_data.len() == mw * mh {
        1
    } else if mask_data.len() == mw * mh * 3 {
        3
    } else if mask_data.len() == mw * mh * 4 {
        4
    } else {
        return Err(ImageError::InvalidInput(format!(
            "mask data length {} doesn't match {}x{} at any format",
            mask_data.len(),
            mw,
            mh
        )));
    };

    let mut result = vec![0u8; original.len()];
    let n = w * h;

    for i in 0..n {
        let y = i / w;
        let x = i % w;

        // Map to mask coordinates (nearest-neighbor)
        let mx = (x * mw / w).min(mw - 1);
        let my = (y * mh / h).min(mh - 1);
        let mi = my * mw + mx;

        let mask_val = match mask_bpp {
            1 => mask_data[mi],
            3 => {
                let base = mi * 3;
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

        let alpha = mask_val as u32;
        let inv_alpha = 255 - alpha;
        let pi = i * bpp;

        // Blend RGB channels
        result[pi] =
            ((adjusted[pi] as u32 * alpha + original[pi] as u32 * inv_alpha + 127) / 255) as u8;
        result[pi + 1] =
            ((adjusted[pi + 1] as u32 * alpha + original[pi + 1] as u32 * inv_alpha + 127) / 255)
                as u8;
        result[pi + 2] =
            ((adjusted[pi + 2] as u32 * alpha + original[pi + 2] as u32 * inv_alpha + 127) / 255)
                as u8;

        // Alpha channel: preserve from original
        if bpp == 4 {
            result[pi + 3] = original[pi + 3];
        }
    }

    Ok(result)
}
