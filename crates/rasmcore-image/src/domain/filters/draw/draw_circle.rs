//! Filter: draw_circle (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Draw a circle on the image. Set filled=true for solid fill.

/// Parameters for draw_circle.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct DrawCircleParams {
    /// Center X coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 50.0,
        hint = "rc.point"
    )]
    pub cx: f32,
    /// Center Y coordinate
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 50.0,
        hint = "rc.point"
    )]
    pub cy: f32,
    /// Circle radius
    #[param(
        min = 1.0,
        max = 65535.0,
        step = 1.0,
        default = 25.0,
        hint = "rc.log_slider"
    )]
    pub radius: f32,
    /// Shape color
    pub color: crate::domain::param_types::ColorRgba,
    /// Stroke width in pixels (outline mode)
    #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
    pub stroke_width: f32,
    /// Fill the circle (true) or draw outline only (false)
    #[param(default = true, hint = "rc.toggle")]
    pub filled: bool,
}

#[rasmcore_macros::register_filter(
    name = "draw_circle",
    category = "draw",
    group = "draw",
    variant = "circle",
    reference = "filled/outlined circle"
)]
#[allow(clippy::too_many_arguments)]
pub fn draw_circle_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &DrawCircleParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let cx = config.cx;
    let cy = config.cy;
    let radius = config.radius;
    let color_r = config.color.r as u32;
    let color_g = config.color.g as u32;
    let color_b = config.color.b as u32;
    let color_a = config.color.a as u32;
    let stroke_width = config.stroke_width;
    let filled = config.filled;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) =
        crate::domain::draw::draw_circle(pixels, info, cx, cy, radius, color, stroke_width, filled)?;
    Ok(result)
}
