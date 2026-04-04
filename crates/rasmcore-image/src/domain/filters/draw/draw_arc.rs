//! Filter: draw_arc (category: draw)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Draw an arc (partial ellipse outline) on the image.
/// Parameters for draw_arc.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "draw_arc", category = "draw", group = "draw", variant = "arc", reference = "elliptical arc stroke")]
pub struct DrawArcParams {
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
    /// Start angle in degrees (0 = right, counter-clockwise)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub start_angle: f32,
    /// End angle in degrees
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 180.0,
        hint = "rc.angle_deg"
    )]
    pub end_angle: f32,
    /// Stroke color
    pub color: crate::domain::param_types::ColorRgba,
    /// Stroke width
    #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
    pub stroke_width: f32,
}

impl CpuFilter for DrawArcParams {
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
    let (result, _) = crate::domain::draw::draw_arc(
        pixels,
        info,
        self.cx,
        self.cy,
        self.rx,
        self.ry,
        self.start_angle,
        self.end_angle,
        color,
        self.stroke_width,
    )?;
    Ok(result)
}
}

