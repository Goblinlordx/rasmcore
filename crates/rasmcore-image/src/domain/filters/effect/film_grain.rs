//! Filter: film_grain (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Film grain simulation
pub struct FilmGrainParams {
    /// Grain amount (0 = none, 1 = heavy)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.3)]
    pub amount: f32,
    /// Grain size in pixels (1 = fine, 4+ = coarse)
    #[param(min = 0.5, max = 8.0, step = 0.1, default = 1.5)]
    pub size: f32,
    /// Random seed for deterministic output
    #[param(min = 0, max = 4294967295, step = 1, default = 0, hint = "rc.seed")]
    pub seed: u32,
}

#[rasmcore_macros::register_filter(
    name = "film_grain",
    category = "effect",
    reference = "photographic film grain overlay"
)]
pub fn film_grain_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &FilmGrainParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;
    let size = config.size;
    let seed = config.seed;

    let params = crate::domain::color_grading::FilmGrainParams {
        amount,
        size,
        color: false,
        seed,
    };
    crate::domain::color_grading::film_grain(pixels, info, &params)
}
