//! Filter: retinex_msr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Multi-scale Retinex (user-facing wrapper with 3 fixed sigma scales).

/// Parameters for multi-scale Retinex.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "retinex_msr", category = "enhancement", group = "retinex", variant = "msr", reference = "Jobson et al. 1997 multi-scale Retinex")]
pub struct RetinexMsrParams {
    /// Small-scale Gaussian sigma
    #[param(min = 1.0, max = 100.0, step = 1.0, default = 15.0)]
    pub sigma_small: f32,
    /// Medium-scale Gaussian sigma
    #[param(min = 10.0, max = 200.0, step = 5.0, default = 80.0)]
    pub sigma_medium: f32,
    /// Large-scale Gaussian sigma
    #[param(min = 50.0, max = 500.0, step = 10.0, default = 250.0)]
    pub sigma_large: f32,
}

impl CpuFilter for RetinexMsrParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let sigma_small = self.sigma_small;
    let sigma_medium = self.sigma_medium;
    let sigma_large = self.sigma_large;

    retinex_msr(
        request,
        upstream,
        info,
        &[sigma_small, sigma_medium, sigma_large],
    )
}
}

