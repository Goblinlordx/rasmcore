//! Filter: quantize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Color quantization via median-cut palette reduction.
/// Parameters for color quantization.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "quantize", category = "color", group = "quantize", reference = "median cut palette quantization")]
pub struct QuantizeParams {
    /// Maximum number of palette colors
    #[param(min = 2, max = 256, step = 1, default = 256)]
    pub max_colors: u32,
}

impl CpuFilter for QuantizeParams {
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
    let max_colors = self.max_colors;

    let palette = crate::domain::quantize::median_cut(pixels, info, max_colors as usize)?;
    crate::domain::quantize::quantize(pixels, info, &palette)
}
}

