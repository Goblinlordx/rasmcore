//! Filter: draw_line (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Draw a line on the image. Color components are 0-255.
/// Parameters for draw_line.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "draw_line", category = "draw", group = "draw", variant = "line", reference = "Bresenham/anti-aliased line")]
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

impl CpuFilter for DrawLineParams {
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
    let x1 = self.x1;
    let y1 = self.y1;
    let x2 = self.x2;
    let y2 = self.y2;
    let color_r = self.color.r as u32;
    let color_g = self.color.g as u32;
    let color_b = self.color.b as u32;
    let color_a = self.color.a as u32;
    let width = self.width;

    let color = [color_r as u8, color_g as u8, color_b as u8, color_a as u8];
    let (result, _) = crate::domain::draw::draw_line(pixels, info, x1, y1, x2, y2, color, width)?;
    Ok(result)
}
}

