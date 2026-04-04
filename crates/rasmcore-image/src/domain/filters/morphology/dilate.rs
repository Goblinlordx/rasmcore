//! Filter: dilate (category: morphology)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Morphological dilation (user-facing wrapper).
/// Parameters for morphological dilation.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "dilate", category = "morphology", group = "morphology", variant = "dilate", reference = "binary dilation")]
pub struct DilateParams {
    /// Kernel size (must be odd)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub ksize: u32,
    /// Structuring element shape: 0=rect, 1=ellipse, 2=cross
    #[param(min = 0, max = 2, step = 1, default = 0)]
    pub shape: u32,
}

impl CpuFilter for DilateParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let overlap = self.ksize / 2;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let ksize = self.ksize;
    let shape = self.shape;

    let result = dilate(pixels, info, ksize, morph_shape_from_u32(shape))?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.ksize / 2;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

