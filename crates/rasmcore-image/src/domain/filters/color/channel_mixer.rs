//! Filter: channel_mixer (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Mix RGB channels via a 3x3 matrix for creative color grading.
///
/// Identity matrix (1,0,0, 0,1,0, 0,0,1) produces unchanged output.
/// IM equivalent: `-color-matrix "rr rg rb 0 / gr gg gb 0 / br bg bb 0 / 0 0 0 1"`
#[rasmcore_macros::register_filter(
    name = "channel_mixer",
    category = "color",
    reference = "RGB channel matrix multiplication",
    color_op = "true"
)]
#[allow(clippy::too_many_arguments)]
pub fn channel_mixer(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ChannelMixerParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let rr = config.rr;
    let rg = config.rg;
    let rb = config.rb;
    let gr = config.gr;
    let gg = config.gg;
    let gb = config.gb;
    let br = config.br;
    let bg = config.bg;
    let bb = config.bb;

    apply_color_op(
        pixels,
        info,
        &ColorOp::ChannelMix([rr, rg, rb, gr, gg, gb, br, bg, bb]),
    )
}
