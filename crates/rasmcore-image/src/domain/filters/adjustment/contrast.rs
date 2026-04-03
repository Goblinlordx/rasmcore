//! Filter: contrast (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adjust contrast (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.

/// Parameters for contrast adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ContrastParams {
    /// Contrast factor (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0, hint = "rc.signed_slider")]
    pub amount: f32,
}
impl LutPointOp for ContrastParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Contrast(self.amount))
    }
}

#[rasmcore_macros::register_filter(
    name = "contrast",
    category = "adjustment",
    reference = "multiplicative contrast",
    point_op = "true"
)]
pub fn contrast(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ContrastParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;

    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::Contrast(amount))
}
