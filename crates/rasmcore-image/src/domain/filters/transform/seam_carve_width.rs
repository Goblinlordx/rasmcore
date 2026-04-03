//! Filter: seam_carve_width (category: transform)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Content-aware width resize via seam carving (output width changes).
///
/// Avidan & Shamir 2007 — removes lowest-energy vertical seams to reduce width.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "seam_carve_width",
    category = "transform",
    group = "seam_carve",
    variant = "width",
    reference = "Avidan & Shamir 2007 content-aware width resize"
)]
pub struct SeamCarveWidth {
    /// Target width in pixels (must be less than current width)
    #[param(min = 1, max = 65535, step = 1, default = 256, hint = "rc.pixels")]
    pub target_width: u32,
}

impl CpuFilter for SeamCarveWidth {
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
            crate::domain::content_aware::seam_carve_width(&pixels, info, self.target_width)?;
        Ok(data)
    }
}
