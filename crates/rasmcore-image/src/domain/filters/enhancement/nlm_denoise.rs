//! Filter: nlm_denoise (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Non-local means denoising (user-facing wrapper with scalar params).

/// Parameters for NLM denoising.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "nlm_denoise", category = "enhancement", group = "denoise", variant = "nlm", reference = "Buades et al. 2005 non-local means")]
pub struct NlmDenoiseParams {
    /// Filter strength (higher = more denoising)
    #[param(min = 1.0, max = 100.0, step = 1.0, default = 10.0)]
    pub h: f32,
    /// Patch size (must be odd)
    #[param(min = 3, max = 21, step = 2, default = 7, hint = "rc.log_slider")]
    pub patch_size: u32,
    /// Search window size (must be odd)
    #[param(min = 7, max = 51, step = 2, default = 21, hint = "rc.log_slider")]
    pub search_size: u32,
}

impl CpuFilter for NlmDenoiseParams {
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
    let h = self.h;
    let patch_size = self.patch_size;
    let search_size = self.search_size;

    nlm_denoise(
        pixels,
        info,
        &NlmParams {
            h,
            patch_size,
            search_size,
            algorithm: NlmAlgorithm::OpenCv,
        },
    )
}
}

