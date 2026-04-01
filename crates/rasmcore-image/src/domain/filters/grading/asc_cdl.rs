//! Filter: asc_cdl (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// ASC CDL color grading (slope/offset/power per RGB channel)
pub struct AscCdlParams {
    /// Red slope
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub slope_r: f32,
    /// Green slope
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub slope_g: f32,
    /// Blue slope
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub slope_b: f32,
    /// Red offset
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub offset_r: f32,
    /// Green offset
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub offset_g: f32,
    /// Blue offset
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub offset_b: f32,
    /// Red power
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub power_r: f32,
    /// Green power
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub power_g: f32,
    /// Blue power
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub power_b: f32,
}
impl ColorLutOp for AscCdlParams {
    fn build_clut(&self) -> ColorLut3D {
        let cdl = crate::domain::color_grading::AscCdl {
            slope: [self.slope_r, self.slope_g, self.slope_b],
            offset: [self.offset_r, self.offset_g, self.offset_b],
            power: [self.power_r, self.power_g, self.power_b],
            saturation: 1.0,
        };
        ColorLut3D::from_fn(DEFAULT_CLUT_GRID, move |r, g, b| {
            crate::domain::color_grading::asc_cdl_pixel(r, g, b, &cdl)
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "asc_cdl",
    category = "grading",
    reference = "ASC CDL slope/offset/power color decision list",
    color_op = "true"
)]
#[allow(clippy::too_many_arguments)]
pub fn asc_cdl_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &AscCdlParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let slope_r = config.slope_r;
    let slope_g = config.slope_g;
    let slope_b = config.slope_b;
    let offset_r = config.offset_r;
    let offset_g = config.offset_g;
    let offset_b = config.offset_b;
    let power_r = config.power_r;
    let power_g = config.power_g;
    let power_b = config.power_b;

    let cdl = crate::domain::color_grading::AscCdl {
        slope: [slope_r, slope_g, slope_b],
        offset: [offset_r, offset_g, offset_b],
        power: [power_r, power_g, power_b],
        saturation: 1.0,
    };
    crate::domain::color_grading::asc_cdl(pixels, info, &cdl)
}
