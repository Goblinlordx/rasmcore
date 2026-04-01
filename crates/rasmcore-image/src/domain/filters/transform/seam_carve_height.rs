//! Filter: seam_carve_height (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Content-aware height resize via seam carving (output height changes)
pub struct SeamCarveHeightParams {
    /// Target height in pixels (must be less than current height)
    #[param(min = 1, max = 65535, step = 1, default = 256, hint = "rc.pixels")]
    pub target_height: u32,
}

#[rasmcore_macros::register_filter(
    name = "seam_carve_height",
    category = "transform",
    group = "seam_carve",
    variant = "height",
    reference = "Avidan & Shamir 2007 content-aware height resize"
)]
pub fn seam_carve_height_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SeamCarveHeightParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_height = config.target_height;

    let (data, _new_info) = crate::domain::content_aware::seam_carve_height(pixels, info, target_height)?;
    Ok(data)
}
