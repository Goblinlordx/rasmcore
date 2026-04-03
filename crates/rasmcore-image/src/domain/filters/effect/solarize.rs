//! Filter: solarize (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Solarize — invert pixels above threshold for a partial-negative effect
pub struct SolarizeParams {
    /// Threshold (0-255): pixels above this are inverted
    #[param(min = 0, max = 255, step = 1, default = 128)]
    pub threshold: u8,
}
impl LutPointOp for SolarizeParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Solarize(self.threshold))
    }
}

#[rasmcore_macros::register_filter(
    name = "solarize",
    category = "effect",
    reference = "Man Ray solarization effect",
    point_op = "true"
)]
pub fn solarize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SolarizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let threshold = config.threshold;

    crate::domain::point_ops::solarize(pixels, info, threshold)
}

impl crate::domain::filter_traits::GpuFilter for SolarizeParams {
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
            include_str!("../../../shaders/solarize_f32.wgsl").to_string()
        });
        let threshold_norm = self.threshold as f32 / 255.0;
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&threshold_norm.to_le_bytes());
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
