//! Filter: vibrance (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Perceptually weighted saturation: boosts low-saturation pixels more.
///
/// Unlike `saturate` which applies a uniform multiplier, vibrance weights
/// the boost inversely by current saturation — muted colors get more boost,
/// already-vivid colors get less. amount=0 is identity.
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
