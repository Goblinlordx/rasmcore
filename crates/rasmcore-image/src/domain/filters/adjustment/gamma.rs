//! Filter: gamma (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Gamma correction (user-facing, LUT-collapsible).
/// Parameters for gamma correction.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct GammaParams {
    /// Gamma value (>1 brightens, <1 darkens)
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
    pub gamma_value: f32,
}
impl LutPointOp for GammaParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Gamma(self.gamma_value))
    }
}

#[rasmcore_macros::register_filter(
    name = "gamma",
    category = "adjustment",
    reference = "power-law gamma correction",
    point_op = "true"
)]
pub fn gamma_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GammaParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let gamma_value = config.gamma_value;

    crate::domain::point_ops::gamma(pixels, info, gamma_value)
}
