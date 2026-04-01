//! Filter: seam_carve_width (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "seam_carve_width",
    category = "transform",
    group = "seam_carve",
    variant = "width",
    reference = "Avidan & Shamir 2007 content-aware width resize"
)]
pub fn seam_carve_width_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SeamCarveWidthParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_width = config.target_width;

    let (data, _new_info) = crate::domain::content_aware::seam_carve_width(pixels, info, target_width)?;
    Ok(data)
}
