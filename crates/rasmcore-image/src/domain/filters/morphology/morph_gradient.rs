//! Filter: morph_gradient (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Morphological gradient (user-facing wrapper).
/// Parameters for morphological gradient.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "morph_gradient", category = "morphology", group = "morphology", variant = "gradient", reference = "dilation minus erosion")]
pub struct MorphGradientParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

impl CpuFilter for MorphGradientParams {
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
    let ksize = self.ksize;
    let shape = self.shape;

    morph_gradient(pixels, info, ksize, morph_shape_from_u32(shape))
}
}

