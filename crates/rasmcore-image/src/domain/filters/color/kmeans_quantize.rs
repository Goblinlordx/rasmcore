//! Filter: kmeans_quantize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "kmeans_quantize", category = "color", group = "quantize", variant = "kmeans", reference = "Lloyd's k-means color clustering (not reference-validated)")]
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

impl CpuFilter for KmeansQuantizeParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
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
        self.k as usize,
        self.max_iterations,
        self.seed as u64,
    )?;
    crate::domain::quantize::quantize(pixels, info, &palette)
}
}

