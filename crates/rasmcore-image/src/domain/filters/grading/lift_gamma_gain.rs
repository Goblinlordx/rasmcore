//! Filter: lift_gamma_gain (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "lift_gamma_gain",
    category = "grading",
    reference = "three-way color corrector (shadows/midtones/highlights)",
    color_op = "true"
)]
#[allow(clippy::too_many_arguments)]
pub fn lift_gamma_gain_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LiftGammaGainParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let lift_r = config.lift_r;
    let lift_g = config.lift_g;
    let lift_b = config.lift_b;
    let gamma_r = config.gamma_r;
    let gamma_g = config.gamma_g;
    let gamma_b = config.gamma_b;
    let gain_r = config.gain_r;
    let gain_g = config.gain_g;
    let gain_b = config.gain_b;

    let lgg = crate::domain::color_grading::LiftGammaGain {
        lift: [lift_r, lift_g, lift_b],
        gamma: [gamma_r, gamma_g, gamma_b],
        gain: [gain_r, gain_g, gain_b],
    };
    crate::domain::color_grading::lift_gamma_gain(pixels, info, &lgg)
}
