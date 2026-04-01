//! Filter: curves_blue (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

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
