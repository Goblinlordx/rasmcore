//! Filter: draw_arc (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw an arc (partial ellipse outline) on the image.
#[rasmcore_macros::register_filter(
    name = "draw_arc",
    category = "draw",
    group = "draw",
    variant = "arc",
    reference = "elliptical arc stroke"
)]
pub fn draw_arc_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DrawArcParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let color = [
        config.color.r,
        config.color.g,
        config.color.b,
        config.color.a,
    ];
    let (result, _) = crate::domain::draw::draw_arc(
        pixels,
        info,
        config.cx,
        config.cy,
        config.rx,
        config.ry,
        config.start_angle,
        config.end_angle,
        color,
        config.stroke_width,
    )?;
    Ok(result)
}
