//! Filter: sepia (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply sepia tone with given `intensity` (0=none, 1=full sepia).

/// Parameters for sepia.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SepiaParams {
    /// Sepia intensity (0=none, 1=full)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub intensity: f32,
}
impl ColorLutOp for SepiaParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Sepia(self.intensity).to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "sepia",
    category = "color",
    reference = "sepia tone matrix",
    color_op = "true"
)]
pub fn sepia(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SepiaParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let intensity = config.intensity;

    apply_color_op(pixels, info, &ColorOp::Sepia(intensity.clamp(0.0, 1.0)))
}

impl crate::domain::filter_traits::GpuFilter for SepiaParams {
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
            include_str!("../../../shaders/sepia_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.intensity.to_le_bytes());
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
