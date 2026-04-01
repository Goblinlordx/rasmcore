//! Filter: dither_floyd_steinberg (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "dither_floyd_steinberg",
    category = "color",
    group = "quantize",
    variant = "dither_floyd_steinberg",
    reference = "Floyd & Steinberg 1976 error diffusion"
)]
pub fn dither_floyd_steinberg_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DitherFloydSteinbergParams,
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
    crate::domain::quantize::dither_floyd_steinberg(pixels, info, &palette)
}
