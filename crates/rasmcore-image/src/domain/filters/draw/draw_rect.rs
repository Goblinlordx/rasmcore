//! Filter: draw_rect (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw a rectangle on the image. Set filled=true for solid fill.

/// Parameters for draw_rect.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct DrawRectParams {
    /// Rectangle X position
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 10.0,
        hint = "rc.pixels"
    )]
    pub x: f32,
    /// Rectangle Y position
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 10.0,
        hint = "rc.pixels"
    )]
    pub y: f32,
    /// Rectangle width
    #[param(
        min = 1.0,
        max = 65535.0,
        step = 1.0,
        default = 100.0,
        hint = "rc.pixels"
    )]
    pub rect_width: f32,
    /// Rectangle height
    #[param(
        min = 1.0,
        max = 65535.0,
        step = 1.0,
        default = 100.0,
        hint = "rc.pixels"
    )]
    pub rect_height: f32,
    /// Shape color
    pub color: crate::domain::param_types::ColorRgba,
    /// Stroke width in pixels (outline mode)
    #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
    pub stroke_width: f32,
    /// Fill the rectangle (true) or draw outline only (false)
    #[param(default = true, hint = "rc.toggle")]
    pub filled: bool,
}

#[rasmcore_macros::register_filter(
    name = "draw_rect",
    category = "draw",
    group = "draw",
    variant = "rect",
    reference = "filled/outlined rectangle"
)]
#[allow(clippy::too_many_arguments)]
pub fn draw_rect_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DrawRectParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let x = config.x;
    let y = config.y;
    let rect_width = config.rect_width;
    let rect_height = config.rect_height;
    let color_r = config.color.r as u32;
    let color_g = config.color.g as u32;
    let color_b = config.color.b as u32;
    let color_a = config.color.a as u32;
    let stroke_width = config.stroke_width;
    let filled = config.filled;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) = crate::domain::draw::draw_rect(
        pixels,
        info,
        x,
        y,
        rect_width,
        rect_height,
        color,
        stroke_width,
        filled,
    )?;
    Ok(result)
}
