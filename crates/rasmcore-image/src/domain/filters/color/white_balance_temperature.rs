//! Filter: white_balance_temperature (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Temperature-based white balance adjustment.

/// Parameters for white balance temperature adjustment.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "white_balance_temperature", category = "color", group = "white_balance", variant = "temperature", reference = "Planckian locus color temperature")]
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

impl CpuFilter for WhiteBalanceTemperatureParams {
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
    let temperature = self.temperature;
    let tint = self.tint;

    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            crate::domain::color_spaces::white_balance_temperature(p8, i8, temperature as f64, tint as f64)
        });
    }
    crate::domain::color_spaces::white_balance_temperature(pixels, info, temperature as f64, tint as f64)
}
}

