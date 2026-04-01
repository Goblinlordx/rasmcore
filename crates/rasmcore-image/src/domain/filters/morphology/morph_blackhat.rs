//! Filter: morph_blackhat (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological black-hat (user-facing wrapper).
#[rasmcore_macros::register_filter(
    name = "morph_blackhat",
    category = "morphology",
    group = "morphology",
    variant = "blackhat",
    reference = "closing minus input"
)]
pub fn morph_blackhat_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MorphBlackhatParams,
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

    morph_blackhat(pixels, info, ksize, morph_shape_from_u32(shape))
}
