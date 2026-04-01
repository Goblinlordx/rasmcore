//! Filter: draw_polygon (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw a polygon on the image. Points are `[{x, y}, ...]` coordinates.
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
