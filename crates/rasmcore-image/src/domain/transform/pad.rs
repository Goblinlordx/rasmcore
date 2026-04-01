use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo};
use super::{bytes_per_pixel, validate_pixel_buffer};

/// Extend the canvas by adding padding around the image.
///
/// `fill_color` should match the pixel format (3 bytes for RGB8, 4 for RGBA8, etc.).
pub fn pad(
    pixels: &[u8],
    info: &ImageInfo,
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    fill_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = w + left as usize + right as usize;
    let out_h = h + top as usize + bottom as usize;

    // Fill output with background color
    let mut output = Vec::with_capacity(out_w * out_h * bpp);
    for _ in 0..out_w * out_h {
        for c in 0..bpp {
            output.push(fill_color.get(c).copied().unwrap_or(0));
        }
    }

    // Blit original image at (left, top)
    for y in 0..h {
        let src_start = y * w * bpp;
        let dst_start = ((top as usize + y) * out_w + left as usize) * bpp;
        output[dst_start..dst_start + w * bpp]
            .copy_from_slice(&pixels[src_start..src_start + w * bpp]);
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_w as u32,
            height: out_h as u32,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}
