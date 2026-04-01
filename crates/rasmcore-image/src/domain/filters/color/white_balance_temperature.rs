//! Filter: white_balance_temperature (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Temperature-based white balance adjustment.
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
