//! Filter: dither_ordered (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Ordered (Bayer) dithering with median-cut palette.
#[rasmcore_macros::register_filter(
    name = "dither_ordered",
    category = "color",
    group = "quantize",
    variant = "dither_ordered",
    reference = "Bayer matrix ordered dithering"
)]
pub fn dither_ordered_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DitherOrderedParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let max_colors = config.max_colors;
    let map_size = config.map_size;

    let palette = crate::domain::quantize::median_cut(pixels, info, max_colors as usize)?;
    crate::domain::quantize::dither_ordered(pixels, info, &palette, map_size as usize)
}
