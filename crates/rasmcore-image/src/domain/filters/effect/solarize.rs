//! Filter: solarize (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Solarize — invert pixels above threshold for a partial-negative effect
pub struct SolarizeParams {
    /// Threshold (0-255): pixels above this are inverted
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub threshold: u8,
}
impl LutPointOp for SolarizeParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Solarize(self.threshold))
    }
}

#[rasmcore_macros::register_filter(
    name = "solarize",
    category = "effect",
    reference = "Man Ray solarization effect",
    point_op = "true"
)]
pub fn solarize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SolarizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let threshold = config.threshold;

    crate::domain::point_ops::solarize(pixels, info, threshold)
}
