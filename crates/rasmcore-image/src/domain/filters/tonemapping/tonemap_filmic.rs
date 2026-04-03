//! Filmic/ACES tone mapping (Narkowicz 2015 / Hable 2010).

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

#[derive(rasmcore_macros::Filter, Clone)]
/// Filmic/ACES tone mapping (Narkowicz 2015)
#[filter(name = "tonemap_filmic", category = "tonemapping", group = "tonemap", variant = "filmic", reference = "Hable 2010 Uncharted 2 filmic curve")]
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

impl CpuFilter for TonemapFilmicParams {
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
    let shoulder_strength = self.shoulder_strength;
    let linear_strength = self.linear_strength;
    let linear_angle = self.linear_angle;
    let toe_strength = self.toe_strength;
    let toe_numerator = self.toe_numerator;

    let params = crate::domain::color_grading::FilmicParams {
        a: shoulder_strength,
        b: linear_strength,
        c: linear_angle,
        d: toe_strength,
        e: toe_numerator,
    };
    crate::domain::color_grading::tonemap_filmic(pixels, info, &params)
}
}

