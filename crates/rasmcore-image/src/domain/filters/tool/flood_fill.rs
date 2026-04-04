//! Filter: flood_fill (category: tool)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Flood fill (user-facing wrapper returning buffer only).
/// Parameters for flood fill.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "flood_fill", category = "tool", reference = "seed-based flood fill")]
pub struct FloodFillParams {
    /// Seed X coordinate
    #[param(min = 0, max = 8000, step = 1, default = 0, hint = "rc.point")]
    pub seed_x: u32,
    /// Seed Y coordinate
    #[param(min = 0, max = 8000, step = 1, default = 0, hint = "rc.point")]
    pub seed_y: u32,
    /// Fill value
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub new_val: u8,
    /// Intensity tolerance
    #[param(min = 0, max = 255, step = 1, default = 10)]
    pub tolerance: u8,
    /// Connectivity: 4 or 8
    #[param(min = 4, max = 8, step = 4, default = 8)]
    pub connectivity: u32,
}

impl CpuFilter for FloodFillParams {
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
    let seed_x = self.seed_x;
    let seed_y = self.seed_y;
    let new_val = self.new_val;
    let tolerance = self.tolerance;
    let connectivity = self.connectivity;

    let (result, _count) = flood_fill(
        pixels,
        info,
        seed_x,
        seed_y,
        new_val,
        tolerance,
        connectivity,
    )?;
    Ok(result)
}
}

