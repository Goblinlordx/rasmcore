//! Filter: curves_red (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "curves_red",
    category = "grading",
    group = "curves",
    variant = "red",
    reference = "spline-interpolated tone curve (red channel)"
)]
pub fn curves_red(
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
        r: pts,
        g: identity.clone(),
        b: identity,
    };
    crate::domain::color_grading::curves(pixels, info, &tc)
}
