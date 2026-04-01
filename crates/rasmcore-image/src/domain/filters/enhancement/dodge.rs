//! Filter: dodge (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Dodge: lighten (increase exposure) selectively in shadows, midtones, or highlights.
///
/// Equivalent to Photoshop's Dodge tool applied uniformly.
/// Formula: `output = pixel + pixel * exposure * range_weight(luma)`
///
/// Range weights:
/// - shadows: peaks at dark values, fades at midtones
/// - midtones: peaks at mid-gray, fades at extremes
/// - highlights: peaks at bright values, fades at midtones
///
/// Validated: pixel-exact match against reference formula (max_diff=0).
#[rasmcore_macros::register_filter(name = "dodge", category = "enhancement")]
pub fn dodge(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DodgeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let exposure = config.exposure;
    let range = config.range;

    dodge_burn_impl(pixels, info, exposure / 100.0, range, true)
}
