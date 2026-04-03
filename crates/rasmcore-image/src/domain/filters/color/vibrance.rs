//! Filter: vibrance (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Perceptually weighted saturation: boosts low-saturation pixels more.
///
/// Unlike `saturate` which applies a uniform multiplier, vibrance weights
/// the boost inversely by current saturation — muted colors get more boost,
/// already-vivid colors get less. amount=0 is identity.

#[derive(rasmcore_macros::Filter, Clone)]
/// Vibrance — perceptually weighted saturation boost.
#[filter(name = "vibrance", category = "color", reference = "saturation-weighted chroma boost", color_op = "true")]
pub struct VibranceParams {
    /// Vibrance amount (-100 to 100). Positive boosts muted colors more.
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub amount: f32,
}
impl ColorLutOp for VibranceParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Vibrance(self.amount).to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for VibranceParams {
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
    let amount = self.amount;

    apply_color_op(pixels, info, &ColorOp::Vibrance(amount))
}
}

