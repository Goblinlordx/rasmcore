//! Filter: film_grain (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

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
