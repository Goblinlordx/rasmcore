//! Filter: quantize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Color quantization via median-cut palette reduction.
#[rasmcore_macros::register_filter(
    name = "quantize",
    category = "color",
    group = "quantize",
    reference = "median cut palette quantization"
)]
pub fn quantize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &QuantizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let max_colors = config.max_colors;

    let palette = crate::domain::quantize::median_cut(pixels, info, max_colors as usize)?;
    crate::domain::quantize::quantize(pixels, info, &palette)
}
