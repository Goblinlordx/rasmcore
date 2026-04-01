//! Filter: threshold_binary (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply binary threshold to a grayscale image.
///
/// Pixels >= threshold become max_value, pixels < threshold become 0.

/// Parameters for binary threshold.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ThresholdBinaryParams {
    /// Threshold value
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub thresh: u8,
    /// Maximum output value
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub max_value: u8,
}

#[rasmcore_macros::register_filter(
    name = "threshold_binary",
    category = "threshold",
    group = "threshold",
    variant = "binary",
    reference = "fixed-level binary threshold"
)]
pub fn threshold_binary(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ThresholdBinaryParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let thresh = config.thresh;
    let max_value = config.max_value;

    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "threshold requires Gray8 input".into(),
        ));
    }
    Ok(pixels
        .iter()
        .map(|&v| if v >= thresh { max_value } else { 0 })
        .collect())
}
