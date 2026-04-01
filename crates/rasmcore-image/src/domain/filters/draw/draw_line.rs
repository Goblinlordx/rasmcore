//! Filter: draw_line (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw a line on the image. Color components are 0-255.

/// Parameters for draw_line.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct DrawLineParams {
    /// Start X coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.pixels"
    )]
    pub x1: f32,
    /// Start Y coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.pixels"
    )]
    pub y1: f32,
    /// End X coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 100.0,
        hint = "rc.pixels"
    )]
    pub x2: f32,
    /// End Y coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 100.0,
        hint = "rc.pixels"
    )]
    pub y2: f32,
    /// Line color
    pub color: crate::domain::param_types::ColorRgba,
    /// Line width in pixels
    #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
    pub width: f32,
}

#[rasmcore_macros::register_filter(
    name = "draw_line",
    category = "draw",
    group = "draw",
    variant = "line",
    reference = "Bresenham/anti-aliased line"
)]
#[allow(clippy::too_many_arguments)]
pub fn draw_line_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DrawLineParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let x1 = config.x1;
    let y1 = config.y1;
    let x2 = config.x2;
    let y2 = config.y2;
    let color_r = config.color.r as u32;
    let color_g = config.color.g as u32;
    let color_b = config.color.b as u32;
    let color_a = config.color.a as u32;
    let width = config.width;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) = crate::domain::draw::draw_line(pixels, info, x1, y1, x2, y2, color, width)?;
    Ok(result)
}
