//! Filter: curves_master (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Tone curve applied to all RGB channels
pub struct CurvesMasterParams {
    /// Control points as JSON array [[x,y],...] in [0,1]
    #[param(
        min = "null",
        max = "null",
        step = "null",
        default = "[[0,0],[1,1]]",
        hint = "rc.text"
    )]
    pub points: String,
}

#[rasmcore_macros::register_filter(
    name = "curves_master",
    category = "grading",
    group = "curves",
    variant = "master",
    reference = "spline-interpolated tone curve (all channels)"
)]
pub fn curves_master(
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
    let tc = crate::domain::color_grading::ToneCurves {
        r: pts.clone(),
        g: pts.clone(),
        b: pts,
    };
    crate::domain::color_grading::curves(pixels, info, &tc)
}
