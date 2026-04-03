//! Filter: retinex_msrcr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Multi-scale Retinex with color restoration (user-facing wrapper).

/// Parameters for multi-scale Retinex with color restoration.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "retinex_msrcr", category = "enhancement", group = "retinex", variant = "msrcr", reference = "Jobson et al. 1997 multi-scale Retinex with color restoration")]
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

impl CpuFilter for RetinexMsrcrParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let sigma_small = self.sigma_small;
    let sigma_medium = self.sigma_medium;
    let sigma_large = self.sigma_large;
    let alpha = self.alpha;
    let beta = self.beta;

    retinex_msrcr(
        request,
        upstream,
        info,
        &[sigma_small, sigma_medium, sigma_large],
        alpha,
        beta,
    )
}
}

