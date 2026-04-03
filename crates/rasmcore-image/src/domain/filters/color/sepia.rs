//! Filter: sepia (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Apply sepia tone with given `intensity` (0=none, 1=full sepia).

/// Parameters for sepia.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "sepia", category = "color", reference = "sepia tone matrix", color_op = "true")]
pub struct SepiaParams {
    /// Sepia intensity (0=none, 1=full)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub intensity: f32,
}
impl ColorLutOp for SepiaParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Sepia(self.intensity).to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for SepiaParams {
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
    let intensity = self.intensity;

    apply_color_op(pixels, info, &ColorOp::Sepia(intensity.clamp(0.0, 1.0)))
}
}

