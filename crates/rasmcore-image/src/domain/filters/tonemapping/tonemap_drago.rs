//! Drago logarithmic HDR tone mapping (Drago et al. 2003).

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

#[derive(rasmcore_macros::Filter, Clone)]
/// Drago logarithmic HDR tone mapping
#[filter(name = "tonemap_drago", category = "tonemapping", group = "tonemap", variant = "drago", reference = "Drago et al. 2003 logarithmic tone mapping")]
pub struct TonemapDragoParams {
    /// Bias parameter (0.5 = low contrast, 1.0 = high contrast)
    #[param(min = 0.5, max = 1.0, step = 0.01, default = 0.85)]
    pub bias: f32,
}

impl CpuFilter for TonemapDragoParams {
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
    let bias = self.bias;

    let params = crate::domain::color_grading::DragoParams { l_max: 1.0, bias };
    crate::domain::color_grading::tonemap_drago(pixels, info, &params)
}
}

