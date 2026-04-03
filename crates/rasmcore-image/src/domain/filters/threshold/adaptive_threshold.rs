//! Filter: adaptive_threshold (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Adaptive threshold (user-facing wrapper with u32 method param).

/// Parameters for adaptive threshold.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "adaptive_threshold", category = "threshold", group = "threshold", variant = "adaptive", reference = "local block-based adaptive threshold")]
pub struct AdaptiveThresholdParams {
    /// Maximum output value
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub max_value: u8,
    /// Adaptive method: 0=mean, 1=gaussian
    #[param(min = 0, max = 1, step = 1, default = 0)]
    pub method: u32,
    /// Block size (must be odd, >= 3)
    #[param(min = 3, max = 51, step = 2, default = 11)]
    pub block_size: u32,
    /// Constant subtracted from mean
    #[param(min = -50.0, max = 50.0, step = 0.5, default = 2.0, hint = "rc.signed_slider")]
    pub c: f32,
}

impl CpuFilter for AdaptiveThresholdParams {
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
    let max_value = self.max_value;
    let method = self.method;
    let block_size = self.block_size;
    let c = self.c;

    let m = match method {
        1 => AdaptiveMethod::Gaussian,
        _ => AdaptiveMethod::Mean,
    };
    adaptive_threshold(pixels, info, max_value, m, block_size, c as f64)
}
}

