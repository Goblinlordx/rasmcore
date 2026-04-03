//! Filter: hue_rotate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Rotate hue by `degrees` (0-360). Works on RGB8 and RGBA8 images.

/// Parameters for hue_rotate.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct HueRotateParams {
    /// Hue rotation in degrees (0-360)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub degrees: f32,
}
impl ColorLutOp for HueRotateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::HueRotate(self.degrees).to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "hue_rotate",
    category = "color",
    reference = "HSV hue rotation",
    color_op = "true"
)]
pub fn hue_rotate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &HueRotateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let degrees = config.degrees;

    apply_color_op(pixels, info, &ColorOp::HueRotate(degrees))
}
