//! Filter: sigmoidal_contrast (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Sigmoidal contrast: S-curve contrast adjustment.
/// Matches ImageMagick `-sigmoidal-contrast strengthxmidpoint%`.
#[rasmcore_macros::register_filter(
    name = "sigmoidal_contrast",
    category = "adjustment",
    reference = "sigmoidal transfer function contrast",
    point_op = "true"
)]
pub fn sigmoidal_contrast(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SigmoidalContrastParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let strength = config.strength;
    let midpoint = config.midpoint;
    let sharpen = config.sharpen;

    crate::domain::point_ops::sigmoidal_contrast(pixels, info, strength, midpoint / 100.0, sharpen)
}
