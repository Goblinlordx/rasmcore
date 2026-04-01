//! Filter: sat_vs_sat (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "sat_vs_sat",
    category = "grading",
    group = "hue_curves",
    variant = "sat_vs_sat",
    reference = "DaVinci Resolve Sat vs Sat curve"
)]
pub fn sat_vs_sat(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    points: String,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let pts = parse_curve_points(&points)?;
    let lut = crate::domain::color_grading::build_norm_curve_lut(&pts);
    crate::domain::color_grading::sat_vs_sat(pixels, info, &lut)
}
