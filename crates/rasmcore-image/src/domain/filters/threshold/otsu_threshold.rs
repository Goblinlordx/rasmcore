//! Filter: otsu_threshold (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Otsu auto-threshold — compute optimal threshold then binarize.
#[rasmcore_macros::register_filter(
    name = "otsu_threshold",
    category = "threshold",
    group = "threshold",
    variant = "otsu",
    reference = "Otsu 1979 automatic bimodal threshold"
)]
pub fn otsu_threshold_registered(
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
    let t = otsu_threshold(pixels, info)?;
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
