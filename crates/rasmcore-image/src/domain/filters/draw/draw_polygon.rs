//! Filter: draw_polygon (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw a polygon on the image. Points are `[{x, y}, ...]` coordinates.
/// Parameters for draw_polygon.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct DrawPolygonParams {
    /// Fill color
    pub fill_color: crate::domain::param_types::ColorRgba,
    /// Stroke color
    pub stroke_color: crate::domain::param_types::ColorRgba,
    /// Stroke width (0 = no stroke)
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 0.0)]
    pub stroke_width: f32,
    /// Fill the polygon
    #[param(default = true, hint = "rc.toggle")]
    pub filled: bool,
}

#[rasmcore_macros::register_filter(
    name = "draw_polygon",
    category = "draw",
    group = "draw",
    variant = "polygon",
    reference = "filled/outlined polygon"
)]
pub fn draw_polygon_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    points: &[crate::domain::param_types::Point2D],
    config: &DrawPolygonParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let fill_color = [
        config.fill_color.r,
        config.fill_color.g,
        config.fill_color.b,
        config.fill_color.a,
    ];
    let stroke_color = [
        config.stroke_color.r,
        config.stroke_color.g,
        config.stroke_color.b,
        config.stroke_color.a,
    ];
    let (result, _) = crate::domain::draw::draw_polygon(
        pixels,
        info,
        points,
        fill_color,
        stroke_color,
        config.stroke_width,
        config.filled,
    )?;
    Ok(result)
}
