//! Filter: exposure (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Photoshop-style exposure adjustment — logarithmic brightness with offset and gamma.
///
/// Uses the composable LUT infrastructure from `point_ops`. Fully LUT-collapsible.

/// Parameters for Photoshop-style exposure adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ExposureParams {
    /// Exposure value in stops (-5 to +5, 0 = unchanged)
    #[param(min = -5.0, max = 5.0, step = 0.1, default = 0.0, hint = "rc.signed_slider")]
    pub ev: f32,
    /// Offset applied before exposure scaling (-0.5 to 0.5)
    #[param(min = -0.5, max = 0.5, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub offset: f32,
    /// Gamma correction applied after exposure (0.01-9.99, 1.0 = linear)
    #[param(min = 0.01, max = 9.99, step = 0.01, default = 1.0)]
    pub gamma_correction: f32,
}
impl LutPointOp for ExposureParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Exposure {
            ev: self.ev,
            offset: self.offset,
            gamma_correction: self.gamma_correction,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "exposure",
    category = "adjustment",
    reference = "Photoshop exposure (EV stops + offset + gamma)",
    point_op = "true"
)]
pub fn exposure(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ExposureParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if config.gamma_correction <= 0.0 {
        return Err(ImageError::InvalidParameters(
            "exposure gamma_correction must be > 0".into(),
        ));
    }
    crate::domain::point_ops::exposure(
        pixels,
        info,
        config.ev,
        config.offset,
        config.gamma_correction,
    )
}

impl crate::domain::filter_traits::GpuFilter for ExposureParams {
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
            include_str!("../../../shaders/exposure_f32.wgsl").to_string()
        });
        let inv_gamma = 1.0 / self.gamma_correction.max(0.001);
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.ev.to_le_bytes());
        params.extend_from_slice(&self.offset.to_le_bytes());
        params.extend_from_slice(&inv_gamma.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // pad
        params.extend_from_slice(&0u32.to_le_bytes()); // pad
        params.extend_from_slice(&0u32.to_le_bytes()); // pad
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
