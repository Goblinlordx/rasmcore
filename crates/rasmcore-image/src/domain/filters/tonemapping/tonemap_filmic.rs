//! Filmic/ACES tone mapping (Narkowicz 2015 / Hable 2010).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Filmic/ACES tone mapping (Narkowicz 2015)
pub struct TonemapFilmicParams {
    /// Shoulder strength (a coefficient)
    #[param(min = 0.0, max = 10.0, step = 0.01, default = 2.51)]
    pub shoulder_strength: f32,
    /// Linear strength (b coefficient)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.03)]
    pub linear_strength: f32,
    /// Linear angle (c coefficient)
    #[param(min = 0.0, max = 10.0, step = 0.01, default = 2.43)]
    pub linear_angle: f32,
    /// Toe strength (d coefficient)
    #[param(min = 0.0, max = 2.0, step = 0.01, default = 0.59)]
    pub toe_strength: f32,
    /// Toe numerator (e coefficient)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.14)]
    pub toe_numerator: f32,
}

#[rasmcore_macros::register_filter(
    name = "tonemap_filmic",
    category = "tonemapping",
    group = "tonemap",
    variant = "filmic",
    reference = "Hable 2010 Uncharted 2 filmic curve"
)]
pub fn tonemap_filmic_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &TonemapFilmicParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let shoulder_strength = config.shoulder_strength;
    let linear_strength = config.linear_strength;
    let linear_angle = config.linear_angle;
    let toe_strength = config.toe_strength;
    let toe_numerator = config.toe_numerator;

    let params = crate::domain::color_grading::FilmicParams {
        a: shoulder_strength,
        b: linear_strength,
        c: linear_angle,
        d: toe_strength,
        e: toe_numerator,
    };
    crate::domain::color_grading::tonemap_filmic(pixels, info, &params)
}
