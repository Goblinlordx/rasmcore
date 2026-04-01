//! Filter: curves_blue (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Tone curve applied to blue channel only
pub struct CurvesBlueParams {
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
    name = "curves_blue",
    category = "grading",
    group = "curves",
    variant = "blue",
    reference = "spline-interpolated tone curve (blue channel)"
)]
pub fn curves_blue(
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
    let identity = vec![(0.0, 0.0), (1.0, 1.0)];
    let tc = crate::domain::color_grading::ToneCurves {
        r: identity.clone(),
        g: identity,
        b: pts,
    };
    crate::domain::color_grading::curves(pixels, info, &tc)
}
