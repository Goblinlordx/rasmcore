//! Filter: blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply gaussian blur using our own convolve() function.
///
/// Uses separable gaussian convolution (auto-detected by convolve) with
/// WASM SIMD128 acceleration. Large sigma (>= 20) uses box blur
/// approximation for O(1) per-pixel performance.

/// Parameters for Gaussian blur.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BlurParams {
    /// Blur radius in pixels
    #[param(
        min = 0.0,
        max = 100.0,
        step = 0.5,
        default = 3.0,
        hint = "rc.log_slider"
    )]
    pub radius: f32,
}

#[rasmcore_macros::register_filter(
    name = "blur", gpu = "true",
    category = "spatial",
    group = "blur",
    reference = "Gaussian convolution"
)]
pub fn blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BlurParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;
    let overlap = if radius > 0.0 { ((radius * 3.0).ceil() as u32).max(1) } else { 0 };
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
    let result = blur_impl(&pixels, &expanded_info, config)?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
