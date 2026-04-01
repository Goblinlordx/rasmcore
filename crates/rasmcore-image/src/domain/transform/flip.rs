use super::super::error::ImageError;
use super::super::types::{DecodedImage, FlipDirection, ImageInfo};
use super::bytes_per_pixel;

/// Flip an image horizontally or vertically using raw buffer ops.
pub fn flip(
    pixels: &[u8],
    info: &ImageInfo,
    direction: FlipDirection,
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = w * bpp;
    let mut result = vec![0u8; pixels.len()];

    match direction {
        FlipDirection::Horizontal => {
            for y in 0..h {
                for x in 0..w {
                    let src_off = y * stride + x * bpp;
                    let dst_off = y * stride + (w - 1 - x) * bpp;
                    result[dst_off..dst_off + bpp].copy_from_slice(&pixels[src_off..src_off + bpp]);
                }
            }
        }
        FlipDirection::Vertical => {
            for y in 0..h {
                let src_off = y * stride;
                let dst_off = (h - 1 - y) * stride;
                result[dst_off..dst_off + stride]
                    .copy_from_slice(&pixels[src_off..src_off + stride]);
            }
        }
    }

    Ok(DecodedImage {
        pixels: result,
        info: info.clone(),
        icc_profile: None,
    })
}
