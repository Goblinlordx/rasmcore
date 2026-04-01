//! Filter: sobel (category: edge)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Sobel edge detection — produces grayscale gradient magnitude image.
///
/// Uses unrolled 3x3 Sobel with padded input — no inner loop or
/// match-based weight lookup. Direct coefficient access gives ~3x speedup.
/// Sobel edge detection — registered as mapper (outputs Gray8).
/// The underlying `sobel_inner` is also used internally by charcoal.
#[rasmcore_macros::register_mapper(
    name = "sobel",
    category = "edge",
    group = "edge_detect",
    variant = "sobel",
    reference = "Sobel 1968 gradient operator",
    output_format = "Gray8"
)]
pub fn sobel_mapper(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let gray_pixels = sobel(pixels, info)?;
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((gray_pixels, out_info))
}
