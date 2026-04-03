//! Filter: connected_components (category: analysis)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "connected_components", category = "analysis", group = "analysis", variant = "connected_components", reference = "two-pass connected component labeling")]
pub struct ConnectedComponentsParams {
    pub connectivity: u32,
}

impl CpuFilter for ConnectedComponentsParams {
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
    let connectivity = self.connectivity;

    let conn = if connectivity == 4 { 4 } else { 8 };
    let (labels, _count) = connected_components(pixels, info, conn)?;
    // Convert u32 labels to u8 (mod 256) for visualization
    Ok(labels.iter().map(|&l| (l % 256) as u8).collect())
}
}

