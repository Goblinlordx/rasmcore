//! Filter: exposure (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Photoshop-style exposure adjustment — logarithmic brightness with offset and gamma.
///
/// Uses the composable LUT infrastructure from `point_ops`. Fully LUT-collapsible.
/// Parameters for Photoshop-style exposure adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ExposureParams {
    /// Exposure value in stops (-5 to +5, 0 = unchanged)
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0, hint = "rc.signed_slider")]
    pub ev: f32,
    /// Offset applied before exposure scaling (-0.5 to 0.5)
    #[param(min = -0.5, max = 0.5, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub offset: f32,
    /// Gamma correction applied after exposure (0.01-9.99, 1.0 = linear)
    #[param(min = 0.01, max = 9.99, step = 0.01, default = 1.0)]
    pub gamma_correction: f32,
}
impl LutPointOp for ExposureParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Exposure {
            ev: self.ev,
            offset: self.offset,
            gamma_correction: self.gamma_correction,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "exposure",
    category = "adjustment",
    reference = "Photoshop exposure (EV stops + offset + gamma)",
    point_op = "true"
)]
pub fn exposure(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ExposureParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if config.gamma_correction <= 0.0 {
        return Err(ImageError::InvalidParameters(
            "exposure gamma_correction must be > 0".into(),
        ));
    }
    crate::domain::point_ops::exposure(
        pixels,
        info,
        config.ev,
        config.offset,
        config.gamma_correction,
    )
}
