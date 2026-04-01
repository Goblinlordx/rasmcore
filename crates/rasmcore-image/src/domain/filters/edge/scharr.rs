//! Filter: scharr (category: edge)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Scharr edge detection — more rotationally symmetric than Sobel.
///
/// Uses 3x3 Scharr kernels: Gx = [[-3,0,3],[-10,0,10],[-3,0,3]]
/// Returns gradient magnitude (L2 norm of Gx and Gy).
/// Reference: cv2.Scharr (OpenCV 4.13).
/// Scharr edge detection — registered as mapper (outputs Gray8).
#[rasmcore_macros::register_mapper(
    name = "scharr",
    category = "edge",
    group = "edge_detect",
    variant = "scharr",
    reference = "Scharr 2000 rotationally symmetric gradient",
    output_format = "Gray8"
)]
pub fn scharr_mapper(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let gray_pixels = scharr(pixels, info)?;
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((gray_pixels, out_info))
}
