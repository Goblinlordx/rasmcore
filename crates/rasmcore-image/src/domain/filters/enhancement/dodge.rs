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

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Dodge — lighten exposure in a selected tonal range
pub struct DodgeParams {
    /// Exposure increase (0-100%)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 50.0)]
    pub exposure: f32,
    /// Tonal range: 0=shadows, 1=midtones, 2=highlights
    #[param(min = 0, max = 2, step = 1, default = 1)]
    pub range: u32,
}

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
