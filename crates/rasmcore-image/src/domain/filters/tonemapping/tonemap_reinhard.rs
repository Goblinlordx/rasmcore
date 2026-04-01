//! Reinhard photographic tone reproduction (Reinhard et al. 2002).

use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "tonemap_reinhard",
    category = "tonemapping",
    group = "tonemap",
    variant = "reinhard",
    reference = "Reinhard et al. 2002 photographic tone reproduction"
)]
pub fn tonemap_reinhard_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    crate::domain::color_grading::tonemap_reinhard(pixels, info)
}
