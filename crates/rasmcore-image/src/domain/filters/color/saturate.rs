//! Filter: saturate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adjust saturation by `factor` (0=grayscale, 1=unchanged, 2=double).

/// Parameters for saturate.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SaturateParams {
    /// Saturation factor (0=grayscale, 1=unchanged, 2=double)
    #[param(min = 0.0, max = 3.0, step = 0.1, default = 1.0)]
    pub factor: f32,
}
impl ColorLutOp for SaturateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Saturate(self.factor).to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "saturate",
    category = "color",
    reference = "HSV saturation scaling",
    color_op = "true"
)]
pub fn saturate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SaturateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let factor = config.factor;

    apply_color_op(pixels, info, &ColorOp::Saturate(factor))
}

impl crate::domain::filter_traits::GpuFilter for SaturateParams {
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
            include_str!("../../../shaders/saturate_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.factor.to_le_bytes());
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
