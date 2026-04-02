//! Filter: sharpen (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply sharpening (unsharp mask).
///
/// Computes: output = original + amount * (original - blurred)
/// Uses the SIMD-optimized blur internally.

/// Parameters for unsharp mask sharpening.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SharpenParams {
    /// Sharpening amount
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 1.0)]
    pub amount: f32,
}

impl InputRectProvider for SharpenParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = 4u32;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

#[rasmcore_macros::register_filter(
    name = "sharpen", gpu = "true",
    category = "spatial",
    reference = "unsharp mask"
)]
pub fn sharpen(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SharpenParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = 4u32; // blur radius 1.0 → kernel ~7, half = 3, +1 safety
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
    let result = sharpen_impl(&pixels, &expanded_info, config)?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
