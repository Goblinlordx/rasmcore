//! Filter: seam_carve_height (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Content-aware height resize via seam carving (output height changes).
///
/// Avidan & Shamir 2007 — removes lowest-energy horizontal seams to reduce height.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "seam_carve_height",
    category = "transform",
    group = "seam_carve",
    variant = "height",
    reference = "Avidan & Shamir 2007 content-aware height resize"
)]
pub struct SeamCarveHeight {
    /// Target height in pixels (must be less than current height)
    #[param(min = 1, max = 65535, step = 1, default = 256, hint = "rc.pixels")]
    pub target_height: u32,
}

impl CpuFilter for SeamCarveHeight {
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
        let (data, _new_info) =
            crate::domain::content_aware::seam_carve_height(&pixels, info, self.target_height)?;
        Ok(data)
    }
}
