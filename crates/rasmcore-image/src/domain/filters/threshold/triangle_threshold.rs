//! Filter: triangle_threshold (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Triangle auto-threshold — compute optimal threshold then binarize.
#[rasmcore_macros::register_filter(
    name = "triangle_threshold",
    category = "threshold",
    group = "threshold",
    variant = "triangle",
    reference = "Zack et al. 1977 triangle method"
)]
pub fn triangle_threshold_registered(
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
    let t = triangle_threshold(pixels, info)?;
    let r = Rect::new(0, 0, info.width, info.height);
    let mut u = |_: Rect| Ok(pixels.to_vec());
    threshold_binary(
        r,
        &mut u,
        info,
        &ThresholdBinaryParams {
            thresh: t,
            max_value: 255,
        },
    )
}
