//! Filter: levels (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Levels adjustment: remap [black, white] input range with gamma curve.
/// Matches ImageMagick `-level black%,white%,gamma`.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Levels adjustment — remap input black/white points with gamma
pub struct LevelsParams {
    /// Input black point (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 0.0)]
    pub black_point: f32,
    /// Input white point (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 100.0)]
    pub white_point: f32,
    /// Gamma correction (0.1-10.0, 1.0 = linear)
    #[param(min = 0.1, max = 10.0, step = 0.01, default = 1.0)]
    pub gamma: f32,
}
impl LutPointOp for LevelsParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Levels {
            black: self.black_point / 100.0,
            white: self.white_point / 100.0,
            gamma: self.gamma,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "levels",
    category = "adjustment",
    reference = "input/output level remapping",
    point_op = "true"
)]
pub fn levels(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LevelsParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let black_point = config.black_point;
    let white_point = config.white_point;
    let gamma = config.gamma;

    // Convert percentage to fraction
    crate::domain::point_ops::levels(
        pixels,
        info,
        black_point / 100.0,
        white_point / 100.0,
        gamma,
    )
}
