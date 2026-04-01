//! Filter: flatten (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Flatten RGBA to RGB by compositing onto a solid background color.
/// Registered as mapper because it changes pixel format (RGBA8 → RGB8).
#[rasmcore_macros::register_mapper(
    name = "flatten",
    category = "alpha",
    group = "alpha",
    variant = "flatten",
    reference = "composite onto background color",
    output_format = "Rgb8"
)]
pub fn flatten_registered(
    pixels: &[u8],
    info: &ImageInfo,
    config: &FlattenParams,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let bg_r = config.bg_r;
    let bg_g = config.bg_g;
    let bg_b = config.bg_b;

    flatten(pixels, info, [bg_r, bg_g, bg_b])
}
