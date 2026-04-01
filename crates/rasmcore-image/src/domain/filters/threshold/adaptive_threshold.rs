//! Filter: adaptive_threshold (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adaptive threshold (user-facing wrapper with u32 method param).
#[rasmcore_macros::register_filter(
    name = "adaptive_threshold",
    category = "threshold",
    group = "threshold",
    variant = "adaptive",
    reference = "local block-based adaptive threshold"
)]
pub fn adaptive_threshold_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &AdaptiveThresholdParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let max_value = config.max_value;
    let method = config.method;
    let block_size = config.block_size;
    let c = config.c;

    let m = match method {
        1 => AdaptiveMethod::Gaussian,
        _ => AdaptiveMethod::Mean,
    };
    adaptive_threshold(pixels, info, max_value, m, block_size, c as f64)
}
