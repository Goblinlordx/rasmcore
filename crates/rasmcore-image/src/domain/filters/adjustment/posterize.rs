//! Filter: posterize (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Posterize to N discrete levels per channel (user-facing, LUT-collapsible).
/// Parameters for posterize.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct PosterizeParams {
    /// Number of discrete levels per channel
    #[param(min = 2, max = 255, step = 1, default = 8)]
    pub levels: u8,
}
impl LutPointOp for PosterizeParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Posterize(self.levels))
    }
}

#[rasmcore_macros::register_filter(
    name = "posterize",
    category = "adjustment",
    reference = "bit-depth reduction",
    point_op = "true"
)]
pub fn posterize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PosterizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let levels = config.levels;

    crate::domain::point_ops::posterize(pixels, info, levels)
}
