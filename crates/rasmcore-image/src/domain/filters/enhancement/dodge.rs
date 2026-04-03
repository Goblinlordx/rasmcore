//! Filter: dodge (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

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

#[derive(rasmcore_macros::Filter, Clone)]
/// Dodge — lighten exposure in a selected tonal range
#[filter(name = "dodge", category = "enhancement")]
pub struct DodgeParams {
    /// Exposure increase (0-100%)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 50.0)]
    pub exposure: f32,
    /// Tonal range: 0=shadows, 1=midtones, 2=highlights
    #[param(min = 0, max = 2, step = 1, default = 1)]
    pub range: u32,
}

impl CpuFilter for DodgeParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let exposure = self.exposure;
    let range = self.range;

    dodge_burn_impl(pixels, info, exposure / 100.0, range, true)
}
}

