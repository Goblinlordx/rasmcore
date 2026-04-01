//! Filter: sigmoidal_contrast (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Sigmoidal contrast: S-curve contrast adjustment.
/// Matches ImageMagick `-sigmoidal-contrast strengthxmidpoint%`.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Sigmoidal contrast — S-curve contrast adjustment
pub struct SigmoidalContrastParams {
    /// Contrast strength (0-20, 0 = identity)
    #[param(min = 0.0, max = 20.0, step = 0.1, default = 3.0)]
    pub strength: f32,
    /// Midpoint percentage (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 50.0)]
    pub midpoint: f32,
    /// true = increase contrast (sharpen), false = decrease contrast (soften)
    #[param(default = true)]
    pub sharpen: bool,
}
impl LutPointOp for SigmoidalContrastParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::SigmoidalContrast {
            strength: self.strength,
            midpoint: self.midpoint / 100.0,
            sharpen: self.sharpen,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "sigmoidal_contrast",
    category = "adjustment",
    reference = "sigmoidal transfer function contrast",
    point_op = "true"
)]
pub fn sigmoidal_contrast(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SigmoidalContrastParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let strength = config.strength;
    let midpoint = config.midpoint;
    let sharpen = config.sharpen;

    crate::domain::point_ops::sigmoidal_contrast(pixels, info, strength, midpoint / 100.0, sharpen)
}
