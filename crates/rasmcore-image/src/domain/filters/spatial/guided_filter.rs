//! Filter: guided_filter (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Edge-preserving guided filter.
///
/// O(N) complexity regardless of radius. Uses a guidance image (typically
/// the input itself) to compute local linear model a*I+b that smooths
/// while preserving edges in the guidance.
///
/// - `radius`: window radius (4-8 typical)
/// - `epsilon`: regularization parameter (0.01-0.1 typical; smaller = more edge-preserving)
///
/// For self-guided filtering, the input is used as both source and guide.

/// Parameters for guided filter.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct GuidedFilterParams {
    /// Window radius (4-8 typical)
    #[param(min = 1, max = 30, step = 1, default = 4, hint = "rc.log_slider")]
    pub radius: u32,
    /// Regularization parameter (smaller = more edge-preserving)
    #[param(min = 0.001, max = 1.0, step = 0.001, default = 0.01)]
    pub epsilon: f32,
}

#[rasmcore_macros::register_filter(
    name = "guided_filter", gpu = "true",
    category = "spatial",
    group = "denoise",
    variant = "guided",
    reference = "He et al. 2010 guided image filtering"
)]
pub fn guided_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GuidedFilterParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = 2 * config.radius;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
    let result = guided_filter_impl(&pixels, &expanded_info, config)?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
