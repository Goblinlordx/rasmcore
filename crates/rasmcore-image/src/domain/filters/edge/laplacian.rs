//! Filter: laplacian (category: edge)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Laplacian — second-order derivative edge detection.
///
/// Uses 3x3 kernel: [[0,1,0],[1,-4,1],[0,1,0]].
/// Returns absolute value of Laplacian, clamped to [0, 255].
/// Reference: cv2.Laplacian (OpenCV 4.13).
/// Laplacian edge detection — registered as mapper (outputs Gray8).
#[rasmcore_macros::register_mapper(
    name = "laplacian",
    category = "edge",
    group = "edge_detect",
    variant = "laplacian",
    reference = "second-order derivative operator",
    output_format = "Gray8"
)]
pub fn laplacian_mapper(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let gray_pixels = laplacian(pixels, info)?;
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((gray_pixels, out_info))
}
