//! Filter: vibrance (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Perceptually weighted saturation: boosts low-saturation pixels more.
///
/// Unlike `saturate` which applies a uniform multiplier, vibrance weights
/// the boost inversely by current saturation — muted colors get more boost,
/// already-vivid colors get less. amount=0 is identity.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Vibrance — perceptually weighted saturation boost.
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

#[rasmcore_macros::register_filter(
    name = "vibrance",
    category = "color",
    reference = "saturation-weighted chroma boost",
    color_op = "true"
)]
pub fn vibrance(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &VibranceParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;

    apply_color_op(pixels, info, &ColorOp::Vibrance(amount))
}
