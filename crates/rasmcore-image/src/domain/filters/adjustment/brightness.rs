//! Filter: brightness (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adjust brightness (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.

/// Parameters for brightness adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BrightnessParams {
    /// Brightness offset (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0, hint = "rc.signed_slider")]
    pub amount: f32,
}
impl LutPointOp for BrightnessParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Brightness(self.amount))
    }
}

#[rasmcore_macros::register_filter(
    name = "brightness",
    category = "adjustment",
    reference = "additive brightness offset",
    point_op = "true"
)]
pub fn brightness(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BrightnessParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;

    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "brightness must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::Brightness(amount))
}

impl crate::domain::filter_traits::GpuFilter for BrightnessParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        // Only f32 shader available — return None for legacy gpu_ops (U32Packed)
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
            include_str!("../../../shaders/brightness_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // padding
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
