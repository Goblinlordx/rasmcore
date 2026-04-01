//! Filter: morph_close (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological closing (user-facing wrapper).

/// Parameters for morphological closing.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MorphCloseParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

#[rasmcore_macros::register_filter(
    name = "morph_close",
    category = "morphology",
    group = "morphology",
    variant = "close",
    reference = "dilation then erosion"
)]
pub fn morph_close_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MorphCloseParams,
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

    morph_close(pixels, info, ksize, morph_shape_from_u32(shape))
}
