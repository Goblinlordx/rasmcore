//! Filter: white_balance_temperature (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Temperature-based white balance adjustment.

/// Parameters for white balance temperature adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct WhiteBalanceTemperatureParams {
    /// Color temperature in Kelvin
    #[param(
        min = 2000.0,
        max = 12000.0,
        step = 100.0,
        default = 6500.0,
        hint = "rc.temperature_k"
    )]
    pub temperature: f32,
    /// Tint adjustment
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub tint: f32,
}

#[rasmcore_macros::register_filter(
    name = "white_balance_temperature",
    category = "color",
    group = "white_balance",
    variant = "temperature",
    reference = "Planckian locus color temperature"
)]
pub fn white_balance_temperature_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &WhiteBalanceTemperatureParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let temperature = config.temperature;
    let tint = config.tint;

    crate::domain::color_spaces::white_balance_temperature(pixels, info, temperature as f64, tint as f64)
}
