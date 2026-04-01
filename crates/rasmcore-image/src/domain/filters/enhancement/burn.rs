//! Filter: burn (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Burn: darken (decrease exposure) selectively in shadows, midtones, or highlights.
///
/// Equivalent to Photoshop's Burn tool applied uniformly.
/// Formula: `output = pixel * (1 - exposure * range_weight(luma))`
///
/// Validated: pixel-exact match against reference formula (max_diff=0).

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Burn — darken exposure in a selected tonal range
pub struct BurnParams {
    /// Exposure decrease (0-100%)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 50.0)]
    pub exposure: f32,
    /// Tonal range: 0=shadows, 1=midtones, 2=highlights
    #[param(min = 0, max = 2, step = 1, default = 1)]
    pub range: u32,
}

#[rasmcore_macros::register_filter(name = "burn", category = "enhancement")]
pub fn burn(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BurnParams,
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

    dodge_burn_impl(pixels, info, exposure / 100.0, range, false)
}
