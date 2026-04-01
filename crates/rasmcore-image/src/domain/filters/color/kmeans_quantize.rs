//! Filter: kmeans_quantize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct KmeansQuantizeParams {
    /// Number of output colors (k clusters)
    #[param(min = 2, max = 256, step = 1, default = 8)]
    pub k: u32,
    /// Maximum Lloyd iterations before returning best result
    #[param(min = 1, max = 200, step = 1, default = 30)]
    pub max_iterations: u32,
    /// Random seed for deterministic initialization
    #[param(min = 0, max = 4294967295, step = 1, default = 0, hint = "rc.seed")]
    pub seed: u32,
}

#[rasmcore_macros::register_filter(
    name = "kmeans_quantize",
    category = "color",
    group = "quantize",
    variant = "kmeans",
    reference = "Lloyd's k-means color clustering (not reference-validated)"
)]
pub fn kmeans_quantize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &KmeansQuantizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let palette = crate::domain::quantize::kmeans_palette(
        pixels,
        info,
        config.k as usize,
        config.max_iterations,
        config.seed as u64,
    )?;
    crate::domain::quantize::quantize(pixels, info, &palette)
}
