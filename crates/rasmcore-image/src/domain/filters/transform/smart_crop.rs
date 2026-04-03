//! Filter: smart_crop (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Content-aware smart crop — selects the most salient region.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "smart_crop",
    category = "transform",
    reference = "saliency-based automatic crop"
)]
pub struct SmartCrop {
    /// Target crop width in pixels
    #[param(min = 1, max = 65535, step = 1, default = 256, hint = "rc.box_select")]
    pub target_width: u32,
    /// Target crop height in pixels
    #[param(min = 1, max = 65535, step = 1, default = 256, hint = "rc.box_select")]
    pub target_height: u32,
}

impl CpuFilter for SmartCrop {
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
        let result = crate::domain::smart_crop::smart_crop(
            &pixels,
            info,
            self.target_width,
            self.target_height,
            crate::domain::smart_crop::SmartCropStrategy::Attention,
        )?;
        Ok(result.pixels)
    }
}
