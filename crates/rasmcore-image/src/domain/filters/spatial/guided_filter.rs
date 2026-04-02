//! Filter: guided_filter (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Edge-preserving guided filter.
///
/// O(N) complexity regardless of radius. Uses a guidance image (typically
/// the input itself) to compute local linear model a*I+b that smooths
/// while preserving edges in the guidance.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "guided_filter", category = "spatial",
    group = "denoise",
    variant = "guided",
    reference = "He et al. 2010 guided image filtering"
)]
pub struct GuidedFilterParams {
    /// Window radius (4-8 typical)
    #[param(min = 1, max = 30, step = 1, default = 4, hint = "rc.log_slider")]
    pub radius: u32,
    /// Regularization parameter (smaller = more edge-preserving)
    #[param(min = 0.001, max = 1.0, step = 0.001, default = 0.01)]
    pub epsilon: f32,
}

impl CpuFilter for GuidedFilterParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = 2 * self.radius;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
        let result = guided_filter_impl(&pixels, &expanded_info, self)?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = 2 * self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for GuidedFilterParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static GUIDED_FILTER: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/guided_filter.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&self.epsilon.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: GUIDED_FILTER.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}
