//! Filter: gamma (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Gamma correction (user-facing, LUT-collapsible).
#[rasmcore_macros::register_filter(
    name = "gamma",
    category = "adjustment",
    reference = "power-law gamma correction",
    point_op = "true"
)]
pub fn gamma_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GammaParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let gamma_value = config.gamma_value;

    crate::domain::point_ops::gamma(pixels, info, gamma_value)
}
