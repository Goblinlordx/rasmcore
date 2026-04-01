//! Filter: grayscale (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Convert to grayscale using BT.709 weights.
/// Registered as mapper because it changes pixel format (RGB8/RGBA8 → Gray8).
#[rasmcore_macros::register_mapper(
    name = "grayscale",
    category = "color",
    reference = "luminance-weighted desaturation",
    output_format = "Gray8"
)]
pub fn grayscale_registered(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let decoded = grayscale(pixels, info)?;
    Ok((decoded.pixels, decoded.info))
}
