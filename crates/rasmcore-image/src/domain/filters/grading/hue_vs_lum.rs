//! Filter: hue_vs_lum (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Adjust luminance based on hue (DaVinci Resolve Hue vs Lum curve)
pub struct HueVsLumParams {
    /// Control points as JSON array [[x,y],...] in [0,1]. x=hue position, y=0.5 is no change.
    #[param(
        min = "null",
        max = "null",
        step = "null",
        default = "[[0,0],[1,1]]",
        hint = "rc.curve"
    )]
    pub points: String,
}

#[rasmcore_macros::register_filter(
    name = "hue_vs_lum",
    category = "grading",
    group = "hue_curves",
    variant = "hue_vs_lum",
    reference = "DaVinci Resolve Hue vs Lum curve"
)]
pub fn hue_vs_lum(
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
    let lut = crate::domain::color_grading::build_hue_curve_lut(&pts);
    crate::domain::color_grading::hue_vs_lum(pixels, info, &lut)
}
