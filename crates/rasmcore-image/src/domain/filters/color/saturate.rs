//! Filter: saturate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Adjust saturation by `factor` (0=grayscale, 1=unchanged, 2=double).
/// Parameters for saturate.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "saturate", category = "color", reference = "HSV saturation scaling", color_op = "true")]
pub struct SaturateParams {
    /// Saturation factor (0=grayscale, 1=unchanged, 2=double)
    #[param(min = 0.0, max = 3.0, step = 0.1, default = 1.0)]
    pub factor: f32,
}
impl ColorLutOp for SaturateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Saturate(self.factor).to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for SaturateParams {
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
    let factor = self.factor;

    apply_color_op(pixels, info, &ColorOp::Saturate(factor))
}
}

