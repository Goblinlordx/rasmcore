//! Filter: smart_crop (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "smart_crop",
    category = "transform",
    reference = "saliency-based automatic crop"
)]
pub fn smart_crop_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SmartCropParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_width = config.target_width;
    let target_height = config.target_height;

    let result = crate::domain::smart_crop::smart_crop(
        pixels,
        info,
        target_width,
        target_height,
        crate::domain::smart_crop::SmartCropStrategy::Attention,
    )?;
    Ok(result.pixels)
}
