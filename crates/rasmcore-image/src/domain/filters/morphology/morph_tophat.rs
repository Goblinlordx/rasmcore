//! Filter: morph_tophat (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Morphological top-hat (user-facing wrapper).
/// Parameters for morphological top-hat.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "morph_tophat", category = "morphology", group = "morphology", variant = "tophat", reference = "input minus opening")]
pub struct MorphTophatParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

impl CpuFilter for MorphTophatParams {
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

    morph_tophat(pixels, info, ksize, morph_shape_from_u32(shape))
}
}

