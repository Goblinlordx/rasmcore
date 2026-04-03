//! Filter: draw_ellipse (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Draw an ellipse on the image.

/// Parameters for draw_ellipse.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "draw_ellipse", category = "draw", group = "draw", variant = "ellipse", reference = "filled/outlined ellipse")]
pub struct DrawEllipseParams {
    /// Center X
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 50.0,
        hint = "rc.point"
    )]
    pub cx: f32,
    /// Center Y
    #[param(
        min = 0.0,
        max = 65535.0,
        step = 1.0,
        default = 50.0,
        hint = "rc.point"
    )]
    pub cy: f32,
    /// Radius X
    #[param(
        min = 1.0,
        max = 65535.0,
        step = 1.0,
        default = 40.0,
        hint = "rc.log_slider"
    )]
    pub rx: f32,
    /// Radius Y
    #[param(
        min = 1.0,
        max = 65535.0,
        step = 1.0,
        default = 25.0,
        hint = "rc.log_slider"
    )]
    pub ry: f32,
    /// Color
    pub color: crate::domain::param_types::ColorRgba,
    /// Stroke width (for outline mode)
    #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
    pub stroke_width: f32,
    /// Fill the ellipse
    #[param(default = true, hint = "rc.toggle")]
    pub filled: bool,
}

impl CpuFilter for DrawEllipseParams {
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
    let color = [
        self.color.r,
        self.color.g,
        self.color.b,
        self.color.a,
    ];
    let (result, _) = crate::domain::draw::draw_ellipse(
        pixels,
        info,
        self.cx,
        self.cy,
        self.rx,
        self.ry,
        color,
        self.stroke_width,
        self.filled,
    )?;
    Ok(result)
}
}

