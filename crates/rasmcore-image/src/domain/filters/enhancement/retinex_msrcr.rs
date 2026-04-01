//! Filter: retinex_msrcr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Multi-scale Retinex with color restoration (user-facing wrapper).
#[rasmcore_macros::register_filter(
    name = "retinex_msrcr",
    category = "enhancement",
    group = "retinex",
    variant = "msrcr",
    reference = "Jobson et al. 1997 multi-scale Retinex with color restoration"
)]
pub fn retinex_msrcr_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &RetinexMsrcrParams,
) -> Result<Vec<u8>, ImageError> {
    let sigma_small = config.sigma_small;
    let sigma_medium = config.sigma_medium;
    let sigma_large = config.sigma_large;
    let alpha = config.alpha;
    let beta = config.beta;

    retinex_msrcr(
        request,
        upstream,
        info,
        &[sigma_small, sigma_medium, sigma_large],
        alpha,
        beta,
    )
}
