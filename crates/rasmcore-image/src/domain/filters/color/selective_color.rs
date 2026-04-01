//! Filter: selective_color (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Selective color — adjust pixels within a specific hue range
pub struct SelectiveColorParams {
    /// Target center hue in degrees (0-360)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub target_hue: f32,
    /// Hue range width in degrees
    #[param(min = 1.0, max = 180.0, step = 1.0, default = 30.0)]
    pub hue_range: f32,
    /// Hue shift in degrees
    #[param(min = -180.0, max = 180.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub hue_shift: f32,
    /// Saturation multiplier (0 = desaturate, 1 = unchanged, 2 = double)
    #[param(min = 0.0, max = 4.0, step = 0.01, default = 1.0)]
    pub saturation: f32,
    /// Lightness offset
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub lightness: f32,
}

#[rasmcore_macros::register_filter(
    name = "selective_color",
    category = "color",
    reference = "hue-range-targeted color adjustment"
)]
pub fn selective_color_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SelectiveColorParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_hue = config.target_hue;
    let hue_range = config.hue_range;
    let hue_shift = config.hue_shift;
    let saturation = config.saturation;
    let lightness = config.lightness;

    let params = crate::domain::content_aware::SelectiveColorParams {
        hue_range: crate::domain::content_aware::HueRange {
            center: target_hue,
            width: hue_range,
        },
        hue_shift,
        saturation,
        lightness,
    };
    crate::domain::content_aware::selective_color(pixels, info, &params)
}
