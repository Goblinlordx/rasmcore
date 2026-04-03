//! Filter: channel_mixer (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Mix RGB channels via a 3x3 matrix for creative color grading.
///
/// Identity matrix (1,0,0, 0,1,0, 0,0,1) produces unchanged output.
/// IM equivalent: `-color-matrix "rr rg rb 0 / gr gg gb 0 / br bg bb 0 / 0 0 0 1"`

#[derive(rasmcore_macros::Filter, Clone)]
/// Channel mixer — cross-mix RGB channels via a 3x3 matrix.
#[filter(name = "channel_mixer", category = "color", reference = "RGB channel matrix multiplication", color_op = "true")]
pub struct ChannelMixerParams {
    /// Red-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub rr: f32,
    /// Red-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub rg: f32,
    /// Red-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub rb: f32,
    /// Green-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub gr: f32,
    /// Green-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub gg: f32,
    /// Green-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub gb: f32,
    /// Blue-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub br: f32,
    /// Blue-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub bg: f32,
    /// Blue-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub bb: f32,
}
impl ColorLutOp for ChannelMixerParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::ChannelMix([
            self.rr, self.rg, self.rb, self.gr, self.gg, self.gb, self.br, self.bg, self.bb,
        ])
        .to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for ChannelMixerParams {
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
    let rr = self.rr;
    let rg = self.rg;
    let rb = self.rb;
    let gr = self.gr;
    let gg = self.gg;
    let gb = self.gb;
    let br = self.br;
    let bg = self.bg;
    let bb = self.bb;

    apply_color_op(
        pixels,
        info,
        &ColorOp::ChannelMix([rr, rg, rb, gr, gg, gb, br, bg, bb]),
    )
}
}

