//! Filter: dither_floyd_steinberg (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Parameters for Floyd-Steinberg dithering.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "dither_floyd_steinberg",
    category = "color",
    group = "quantize",
    variant = "dither_floyd_steinberg",
    reference = "Floyd & Steinberg 1976 error diffusion"
)]
pub struct DitherFloydSteinbergParams {
    /// Maximum number of palette colors
    #[param(min = 2, max = 256, step = 1, default = 256)]
    pub max_colors: u32,
}

impl CpuFilter for DitherFloydSteinbergParams {
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
        let max_colors = self.max_colors;
        let palette = crate::domain::quantize::median_cut(&pixels, info, max_colors as usize)?;
        crate::domain::quantize::dither_floyd_steinberg(&pixels, info, &palette)
    }
}
