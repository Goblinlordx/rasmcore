//! Compositor: mask_combine (category: mask)
//!
//! Combine two masks using add, subtract, or intersect modes.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Combine two masks.
///
/// Both masks must be the same format and dimensions.
/// Modes: 0 = add (union), 1 = subtract (difference), 2 = intersect (min).
///
/// For Gray8: operates directly on gray values.
/// For RGB8: operates on BT.709 luminance of each pixel.
#[rasmcore_macros::register_compositor(
    name = "mask_combine",
    category = "mask",
    group = "mask",
    variant = "combine",
    reference = "add/subtract/intersect mask combination"
)]
pub fn mask_combine(
    mask_a: &[u8],
    mask_a_info: &ImageInfo,
    mask_b: &[u8],
    mask_b_info: &ImageInfo,
    mode: u32,
) -> Result<Vec<u8>, ImageError> {
    if mask_a_info.width != mask_b_info.width || mask_a_info.height != mask_b_info.height {
        return Err(ImageError::InvalidInput("mask dimension mismatch".into()));
    }
    if mask_a_info.format != mask_b_info.format {
        return Err(ImageError::InvalidInput("mask format mismatch".into()));
    }

    let n = (mask_a_info.width as usize) * (mask_a_info.height as usize);
    let bpp = match mask_a_info.format {
        PixelFormat::Gray8 => 1usize,
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "mask_combine requires Gray8, RGB8, or RGBA8".into(),
            ));
        }
    };

    let mut result = vec![0u8; mask_a.len()];

    for i in 0..n {
        let a_val = gray_from_pixel(mask_a, i, bpp);
        let b_val = gray_from_pixel(mask_b, i, bpp);

        let combined = match mode {
            0 => (a_val as u16 + b_val as u16).min(255) as u8, // Add (union)
            1 => (a_val as i16 - b_val as i16).max(0) as u8,   // Subtract
            2 => a_val.min(b_val),                             // Intersect (min)
            _ => a_val,
        };

        set_gray_pixel(&mut result, i, bpp, combined);
    }

    Ok(result)
}

#[inline]
fn gray_from_pixel(data: &[u8], pixel_idx: usize, bpp: usize) -> u8 {
    match bpp {
        1 => data[pixel_idx],
        3 => {
            let base = pixel_idx * 3;
            let r = data[base] as u32;
            let g = data[base + 1] as u32;
            let b = data[base + 2] as u32;
            ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
        }
        4 => {
            let base = pixel_idx * 4;
            let r = data[base] as u32;
            let g = data[base + 1] as u32;
            let b = data[base + 2] as u32;
            ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
        }
        _ => 0,
    }
}

#[inline]
fn set_gray_pixel(data: &mut [u8], pixel_idx: usize, bpp: usize, val: u8) {
    match bpp {
        1 => data[pixel_idx] = val,
        3 => {
            let base = pixel_idx * 3;
            data[base] = val;
            data[base + 1] = val;
            data[base + 2] = val;
        }
        4 => {
            let base = pixel_idx * 4;
            data[base] = val;
            data[base + 1] = val;
            data[base + 2] = val;
            data[base + 3] = 255;
        }
        _ => {}
    }
}
