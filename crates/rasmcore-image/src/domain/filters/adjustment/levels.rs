//! Filter: levels (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Levels adjustment: remap [black, white] input range with gamma curve.
/// Matches ImageMagick `-level black%,white%,gamma`.
#[rasmcore_macros::register_filter(
    name = "levels",
    category = "adjustment",
    reference = "input/output level remapping",
    point_op = "true"
)]
pub fn levels(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LevelsParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let black_point = config.black_point;
    let white_point = config.white_point;
    let gamma = config.gamma;

    // Convert percentage to fraction
    crate::domain::point_ops::levels(
        pixels,
        info,
        black_point / 100.0,
        white_point / 100.0,
        gamma,
    )
}
