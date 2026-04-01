//! Filter: morph_gradient (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological gradient (user-facing wrapper).
#[rasmcore_macros::register_filter(
    name = "morph_gradient",
    category = "morphology",
    group = "morphology",
    variant = "gradient",
    reference = "dilation minus erosion"
)]
pub fn morph_gradient_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MorphGradientParams,
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

    morph_gradient(pixels, info, ksize, morph_shape_from_u32(shape))
}
