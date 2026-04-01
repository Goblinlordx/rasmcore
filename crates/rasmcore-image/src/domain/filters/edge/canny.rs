//! Filter: canny (category: edge)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Canny edge detection — produces binary edge map (0 or 255).
///
/// Steps: 1) Gaussian blur, 2) Sobel gradient + direction,
/// 3) Non-maximum suppression, 4) Hysteresis thresholding.
/// Canny edge detection — registered as mapper (outputs Gray8).
#[rasmcore_macros::register_mapper(
    name = "canny",
    category = "edge",
    group = "edge_detect",
    variant = "canny",
    reference = "Canny 1986 multi-stage edge detector",
    output_format = "Gray8"
)]
pub fn canny_mapper(
    pixels: &[u8],
    info: &ImageInfo,
    config: &CannyParams,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let gray_pixels = canny(pixels, info, config)?;
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((gray_pixels, out_info))
}
