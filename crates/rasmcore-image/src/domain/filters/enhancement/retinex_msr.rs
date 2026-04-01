//! Filter: retinex_msr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Multi-scale Retinex (user-facing wrapper with 3 fixed sigma scales).
#[rasmcore_macros::register_filter(
    name = "retinex_msr",
    category = "enhancement",
    group = "retinex",
    variant = "msr",
    reference = "Jobson et al. 1997 multi-scale Retinex"
)]
pub fn retinex_msr_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &RetinexMsrParams,
) -> Result<Vec<u8>, ImageError> {
    let sigma_small = config.sigma_small;
    let sigma_medium = config.sigma_medium;
    let sigma_large = config.sigma_large;

    retinex_msr(
        request,
        upstream,
        info,
        &[sigma_small, sigma_medium, sigma_large],
    )
}
