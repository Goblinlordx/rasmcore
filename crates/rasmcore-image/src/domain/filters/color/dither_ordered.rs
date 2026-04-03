//! Filter: dither_ordered (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Ordered (Bayer) dithering with median-cut palette.

/// Parameters for ordered (Bayer) dithering.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "dither_ordered", category = "color", group = "quantize", variant = "dither_ordered", reference = "Bayer matrix ordered dithering")]
pub struct DitherOrderedParams {
    /// Maximum number of palette colors
    #[param(min = 2, max = 256, step = 1, default = 256)]
    pub max_colors: u32,
    /// Bayer matrix size (2, 4, 8, or 16)
    #[param(min = 2, max = 16, step = 2, default = 4)]
    pub map_size: u32,
}

impl CpuFilter for DitherOrderedParams {
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
    let map_size = self.map_size;

    let palette = crate::domain::quantize::median_cut(pixels, info, max_colors as usize)?;
    crate::domain::quantize::dither_ordered(pixels, info, &palette, map_size as usize)
}
}

