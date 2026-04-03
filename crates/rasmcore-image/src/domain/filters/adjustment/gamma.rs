//! Filter: gamma (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Gamma correction (user-facing, LUT-collapsible).

/// Parameters for gamma correction.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct GammaParams {
    /// Gamma value (>1 brightens, <1 darkens)
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
    pub gamma_value: f32,
}
impl LutPointOp for GammaParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Gamma(self.gamma_value))
    }
}

#[rasmcore_macros::register_filter(
    name = "gamma",
    category = "adjustment",
    reference = "power-law gamma correction",
    point_op = "true"
)]
pub fn gamma_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GammaParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let gamma_value = config.gamma_value;

    crate::domain::point_ops::gamma(pixels, info, gamma_value)
}

impl crate::domain::filter_traits::GpuFilter for GammaParams {
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
            include_str!("../../../shaders/gamma_f32.wgsl").to_string()
        });
        let inv_gamma = 1.0 / self.gamma_value.max(0.001);
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&inv_gamma.to_le_bytes());
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
