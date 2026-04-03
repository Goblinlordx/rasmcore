//! Filter: posterize (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Posterize to N discrete levels per channel (user-facing, LUT-collapsible).

/// Parameters for posterize.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct PosterizeParams {
    /// Number of discrete levels per channel
    #[param(min = 2, max = 255, step = 1, default = 8)]
    pub levels: u8,
}
impl LutPointOp for PosterizeParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Posterize(self.levels))
    }
}

#[rasmcore_macros::register_filter(
    name = "posterize",
    category = "adjustment",
    reference = "bit-depth reduction",
    point_op = "true"
)]
pub fn posterize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PosterizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let levels = config.levels;

    crate::domain::point_ops::posterize(pixels, info, levels)
}

impl crate::domain::filter_traits::GpuFilter for PosterizeParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        None
    }

    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        if buffer_format != rasmcore_pipeline::gpu::BufferFormat::F32Vec4 {
            return None;
        }
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/posterize_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&(self.levels as f32).to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        Some(vec![GpuOp::Compute {
            shader: SHADER.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::F32Vec4,
        }])
    }
}
