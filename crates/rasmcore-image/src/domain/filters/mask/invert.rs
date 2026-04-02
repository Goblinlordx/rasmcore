//! Filter: mask_invert (category: mask)
//!
//! Invert a grayscale mask (255 - value).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Mask invert — no parameters needed.
pub struct MaskInvertParams {}

/// Invert mask values: 0 becomes 255, 255 becomes 0.
///
/// Works on Gray8, RGB8 (R=G=B), or RGBA8.
#[rasmcore_macros::register_filter(
    name = "mask_invert",
    category = "mask",
    reference = "grayscale mask inversion"
)]
pub fn mask_invert(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    _config: &MaskInvertParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };

    match info.format {
        PixelFormat::Gray8 => {
            let mut result = pixels;
            for v in result.iter_mut() {
                *v = 255 - *v;
            }
            Ok(result)
        }
        PixelFormat::Rgb8 => {
            let mut result = pixels;
            for chunk in result.chunks_exact_mut(3) {
                chunk[0] = 255 - chunk[0];
                chunk[1] = 255 - chunk[1];
                chunk[2] = 255 - chunk[2];
            }
            Ok(result)
        }
        PixelFormat::Rgba8 => {
            let mut result = pixels;
            for chunk in result.chunks_exact_mut(4) {
                chunk[0] = 255 - chunk[0];
                chunk[1] = 255 - chunk[1];
                chunk[2] = 255 - chunk[2];
                // Alpha preserved
            }
            Ok(result)
        }
        _ => Err(ImageError::UnsupportedFormat(
            "mask_invert requires Gray8, RGB8, or RGBA8".into(),
        )),
    }
}
