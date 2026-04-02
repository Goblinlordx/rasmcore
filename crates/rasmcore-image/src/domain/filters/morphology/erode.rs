//! Filter: erode (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Morphological erosion (user-facing wrapper).

/// Parameters for morphological erosion.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ErodeParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

impl InputRectProvider for ErodeParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.ksize / 2;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

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
