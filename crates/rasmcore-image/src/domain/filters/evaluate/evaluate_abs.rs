//! Filter: evaluate_abs

use crate::domain::filters::common::*;

/// Absolute value of channel values (identity for u8, included for pipeline symmetry).
#[rasmcore_macros::register_filter(
    name = "evaluate_abs",
    category = "evaluate",
    group = "evaluate",
    variant = "abs",
    reference = "ImageMagick -evaluate Abs"
)]
pub fn evaluate_abs(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    Ok(pixels.to_vec()) // identity for unsigned types
}
