//! Filter: draw_ellipse (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw an ellipse on the image.
#[rasmcore_macros::register_filter(
    name = "draw_ellipse",
    category = "draw",
    group = "draw",
    variant = "ellipse",
    reference = "filled/outlined ellipse"
)]
pub fn draw_ellipse_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DrawEllipseParams,
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
    let (result, _) = crate::domain::draw::draw_ellipse(
        pixels,
        info,
        config.cx,
        config.cy,
        config.rx,
        config.ry,
        color,
        config.stroke_width,
        config.filled,
    )?;
    Ok(result)
}
