//! Filter: draw_circle (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Draw a circle on the image. Set filled=true for solid fill.
/// Parameters for draw_circle.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "draw_circle", category = "draw", group = "draw", variant = "circle", reference = "filled/outlined circle")]
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

impl CpuFilter for DrawCircleParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let cx = self.cx;
    let cy = self.cy;
    let radius = self.radius;
    let color_r = self.color.r as u32;
    let color_g = self.color.g as u32;
    let color_b = self.color.b as u32;
    let color_a = self.color.a as u32;
    let stroke_width = self.stroke_width;
    let filled = self.filled;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) =
        crate::domain::draw::draw_circle(pixels, info, cx, cy, radius, color, stroke_width, filled)?;
    Ok(result)
}
}

