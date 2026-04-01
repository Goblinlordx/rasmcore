use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo};
use super::bytes_per_pixel;

/// Trim uniform borders from an image.
///
/// Scans inward from each edge, comparing pixels against the top-left corner pixel.
/// Pixels within `threshold` (per-channel absolute difference) are considered border.
pub fn trim(pixels: &[u8], info: &ImageInfo, threshold: u8) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    super::validate_pixel_buffer(pixels, info, bpp)?;

    if w == 0 || h == 0 {
        return Err(ImageError::InvalidParameters("empty image".into()));
    }

    // Reference color: top-left pixel
    let ref_color = &pixels[0..bpp];

    let pixel_matches = |x: usize, y: usize| -> bool {
        let idx = (y * w + x) * bpp;
        for c in 0..bpp {
            if (pixels[idx + c] as i16 - ref_color[c] as i16).unsigned_abs() > threshold as u16 {
                return false;
            }
        }
        true
    };

    // Scan from each edge
    let mut top = 0;
    'top: for y in 0..h {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'top;
            }
        }
        top = y + 1;
    }

    let mut bottom = h;
    'bottom: for y in (top..h).rev() {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'bottom;
            }
        }
        bottom = y;
    }

    let mut left = 0;
    'left: for x in 0..w {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'left;
            }
        }
        left = x + 1;
    }

    let mut right = w;
    'right: for x in (left..w).rev() {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'right;
            }
        }
        right = x;
    }

    if left >= right || top >= bottom {
        // Entire image is uniform border — return 1x1
        return Ok(DecodedImage {
            pixels: ref_color.to_vec(),
            info: ImageInfo {
                width: 1,
                height: 1,
                format: info.format,
                color_space: info.color_space,
            },
            icc_profile: None,
        });
    }

    super::crop(
        pixels,
        info,
        left as u32,
        top as u32,
        (right - left) as u32,
        (bottom - top) as u32,
    )
}
