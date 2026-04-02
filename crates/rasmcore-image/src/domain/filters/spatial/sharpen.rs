//! Filter: sharpen (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Unsharp mask sharpening.
///
/// Computes: output = original + amount * (original - blurred)
/// Uses the SIMD-optimized blur internally.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "sharpen", category = "spatial",
    reference = "unsharp mask"
)]
pub struct SharpenParams {
    /// Sharpening amount
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 1.0)]
    pub amount: f32,
}

impl CpuFilter for SharpenParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = 4u32;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
        let result = sharpen_impl(&pixels, &expanded_info, self)?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = 4u32;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for SharpenParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static SHARPEN: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/sharpen.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: SHARPEN.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}
