//! Filter: burn (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Burn: darken (decrease exposure) selectively in shadows, midtones, or highlights.
///
/// Equivalent to Photoshop's Burn tool applied uniformly.
/// Formula: `output = pixel * (1 - exposure * range_weight(luma))`
///
/// Validated: pixel-exact match against reference formula (max_diff=0).
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
