//! Filter: triangle_threshold (category: threshold)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Triangle auto-threshold — compute optimal threshold then binarize.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "triangle_threshold",
    category = "threshold",
    group = "threshold",
    variant = "triangle",
    reference = "Zack et al. 1977 triangle method"
)]
pub struct TriangleThresholdParams {
    /// Placeholder (Triangle method computes threshold automatically)
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub max_value: u8,
}

impl CpuFilter for TriangleThresholdParams {
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
        let t = triangle_threshold(&pixels, info)?;
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        let params = ThresholdBinaryParams {
            thresh: t,
            max_value: self.max_value,
        };
        params.compute(r, &mut u, info)
    }
}
