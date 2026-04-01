//! Filter: morph_open (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological opening (user-facing wrapper).
#[rasmcore_macros::register_filter(
    name = "morph_open",
    category = "morphology",
    group = "morphology",
    variant = "open",
    reference = "erosion then dilation"
)]
pub fn morph_open_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MorphOpenParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let ksize = config.ksize;
    let shape = config.shape;

    morph_open(pixels, info, ksize, morph_shape_from_u32(shape))
}
