use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo};
use super::bytes_per_pixel;

/// Crop a region from an image using raw row-slice copies.
pub fn crop(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, ImageError> {
    if x + width > info.width || y + height > info.height {
        return Err(ImageError::InvalidParameters(format!(
            "crop region ({x},{y},{width},{height}) exceeds image bounds ({},{})",
            info.width, info.height
        )));
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidParameters(
            "crop dimensions must be > 0".into(),
        ));
    }
    let bpp = bytes_per_pixel(info.format)?;
    let src_stride = info.width as usize * bpp;
    let dst_stride = width as usize * bpp;
    let mut result = vec![0u8; height as usize * dst_stride];

    for row in 0..height as usize {
        let src_offset = (y as usize + row) * src_stride + x as usize * bpp;
        let dst_offset = row * dst_stride;
        result[dst_offset..dst_offset + dst_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + dst_stride]);
    }

    Ok(DecodedImage {
        pixels: result,
        info: ImageInfo {
            width,
            height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}
