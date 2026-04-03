//! Filter: lift_gamma_gain (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
/// Lift/Gamma/Gain 3-way color corrector
#[filter(name = "lift_gamma_gain", category = "grading", reference = "three-way color corrector (shadows/midtones/highlights)", color_op = "true")]
pub struct LiftGammaGainParams {
    /// Red lift
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub lift_r: f32,
    /// Green lift
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub lift_g: f32,
    /// Blue lift
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.color_rgb")]
    pub lift_b: f32,
    /// Red gamma
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gamma_r: f32,
    /// Green gamma
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gamma_g: f32,
    /// Blue gamma
    #[param(
        min = 0.1,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gamma_b: f32,
    /// Red gain
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gain_r: f32,
    /// Green gain
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gain_g: f32,
    /// Blue gain
    #[param(
        min = 0.0,
        max = 4.0,
        step = 0.01,
        default = 1.0,
        hint = "rc.color_rgb"
    )]
    pub gain_b: f32,
}
impl ColorLutOp for LiftGammaGainParams {
    fn build_clut(&self) -> ColorLut3D {
        let lgg = crate::domain::color_grading::LiftGammaGain {
            lift: [self.lift_r, self.lift_g, self.lift_b],
            gamma: [self.gamma_r, self.gamma_g, self.gamma_b],
            gain: [self.gain_r, self.gain_g, self.gain_b],
        };
        ColorLut3D::from_fn(DEFAULT_CLUT_GRID, move |r, g, b| {
            crate::domain::color_grading::lift_gamma_gain_pixel(r, g, b, &lgg)
        })
    }
}

impl CpuFilter for LiftGammaGainParams {
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
    let lift_r = self.lift_r;
    let lift_g = self.lift_g;
    let lift_b = self.lift_b;
    let gamma_r = self.gamma_r;
    let gamma_g = self.gamma_g;
    let gamma_b = self.gamma_b;
    let gain_r = self.gain_r;
    let gain_g = self.gain_g;
    let gain_b = self.gain_b;

    let lgg = crate::domain::color_grading::LiftGammaGain {
        lift: [lift_r, lift_g, lift_b],
        gamma: [gamma_r, gamma_g, gamma_b],
        gain: [gain_r, gain_g, gain_b],
    };
    crate::domain::color_grading::lift_gamma_gain(pixels, info, &lgg)
}
}

