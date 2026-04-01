//! Filter: nlm_denoise (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Non-local means denoising (user-facing wrapper with scalar params).
#[rasmcore_macros::register_filter(
    name = "nlm_denoise",
    category = "enhancement",
    group = "denoise",
    variant = "nlm",
    reference = "Buades et al. 2005 non-local means"
)]
pub fn nlm_denoise_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &NlmDenoiseParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let h = config.h;
    let patch_size = config.patch_size;
    let search_size = config.search_size;

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
