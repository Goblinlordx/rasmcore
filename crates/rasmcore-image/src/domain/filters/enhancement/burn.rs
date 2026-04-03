//! Filter: burn (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Burn: darken (decrease exposure) selectively in shadows, midtones, or highlights.
///
/// Equivalent to Photoshop's Burn tool applied uniformly.
/// Formula: `output = pixel * (1 - exposure * range_weight(luma))`
///
/// Validated: pixel-exact match against reference formula (max_diff=0).

#[derive(rasmcore_macros::Filter, Clone)]
/// Burn — darken exposure in a selected tonal range
#[filter(name = "burn", category = "enhancement")]
pub struct BurnParams {
    /// Exposure decrease (0-100%)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 50.0)]
    pub exposure: f32,
    /// Tonal range: 0=shadows, 1=midtones, 2=highlights
    #[param(min = 0, max = 2, step = 1, default = 1)]
    pub range: u32,
}

impl CpuFilter for BurnParams {
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
    let exposure = self.exposure;
    let range = self.range;

    dodge_burn_impl(pixels, info, exposure / 100.0, range, false)
}
}

impl GpuFilter for BurnParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/dodge_burn_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&(self.exposure / 100.0).to_le_bytes());
        params.extend_from_slice(&self.range.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // is_dodge = 0
        params.extend_from_slice(&0u32.to_le_bytes()); // pad
        params.extend_from_slice(&0u32.to_le_bytes()); // pad
        params.extend_from_slice(&0u32.to_le_bytes()); // pad

        Some(vec![GpuOp::Compute {
            shader: SHADER_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}

