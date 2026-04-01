//! Filter: erode (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological erosion (user-facing wrapper).
#[rasmcore_macros::register_filter(
    name = "erode",
    category = "morphology",
    group = "morphology",
    variant = "erode",
    reference = "binary erosion"
)]
pub fn erode_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ErodeParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.ksize / 2;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let ksize = config.ksize;
    let shape = config.shape;

    let result = erode(pixels, info, ksize, morph_shape_from_u32(shape))?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
