//! Filter: dilate (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological dilation (user-facing wrapper).

/// Parameters for morphological dilation.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct DilateParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

#[rasmcore_macros::register_filter(
    name = "dilate",
    category = "morphology",
    group = "morphology",
    variant = "dilate",
    reference = "binary dilation"
)]
pub fn dilate_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DilateParams,
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

    let result = dilate(pixels, info, ksize, morph_shape_from_u32(shape))?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
