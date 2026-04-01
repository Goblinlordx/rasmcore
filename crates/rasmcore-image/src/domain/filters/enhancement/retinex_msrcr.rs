//! Filter: retinex_msrcr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Multi-scale Retinex with color restoration (user-facing wrapper).

/// Parameters for multi-scale Retinex with color restoration.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct RetinexMsrcrParams {
    /// Small-scale Gaussian sigma
    #[param(min = 1.0, max = 100.0, step = 1.0, default = 15.0)]
    pub sigma_small: f32,
    /// Medium-scale Gaussian sigma
    #[param(min = 10.0, max = 200.0, step = 5.0, default = 80.0)]
    pub sigma_medium: f32,
    /// Large-scale Gaussian sigma
    #[param(min = 50.0, max = 500.0, step = 10.0, default = 250.0)]
    pub sigma_large: f32,
    /// Color restoration nonlinearity
    #[param(min = 1.0, max = 300.0, step = 5.0, default = 125.0)]
    pub alpha: f32,
    /// Color restoration gain
    #[param(min = 1.0, max = 100.0, step = 1.0, default = 46.0)]
    pub beta: f32,
}

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
