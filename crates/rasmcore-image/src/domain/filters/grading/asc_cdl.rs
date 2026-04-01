//! Filter: asc_cdl (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "asc_cdl",
    category = "grading",
    reference = "ASC CDL slope/offset/power color decision list",
    color_op = "true"
)]
#[allow(clippy::too_many_arguments)]
pub fn asc_cdl_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &AscCdlParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let slope_r = config.slope_r;
    let slope_g = config.slope_g;
    let slope_b = config.slope_b;
    let offset_r = config.offset_r;
    let offset_g = config.offset_g;
    let offset_b = config.offset_b;
    let power_r = config.power_r;
    let power_g = config.power_g;
    let power_b = config.power_b;

    let cdl = crate::domain::color_grading::AscCdl {
        slope: [slope_r, slope_g, slope_b],
        offset: [offset_r, offset_g, offset_b],
        power: [power_r, power_g, power_b],
        saturation: 1.0,
    };
    crate::domain::color_grading::asc_cdl(pixels, info, &cdl)
}
