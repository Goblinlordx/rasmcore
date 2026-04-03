//! Drago logarithmic HDR tone mapping (Drago et al. 2003).

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

#[derive(rasmcore_macros::Filter, Clone)]
/// Drago logarithmic HDR tone mapping
#[filter(name = "tonemap_drago", category = "tonemapping", group = "tonemap", variant = "drago", reference = "Drago et al. 2003 logarithmic tone mapping")]
pub struct TonemapDragoParams {
    /// Bias parameter (0.5 = low contrast, 1.0 = high contrast)
    #[param(min = 0.5, max = 1.0, step = 0.01, default = 0.85)]
    pub bias: f32,
}

impl CpuFilter for TonemapDragoParams {
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
    let bias = self.bias;

    let params = crate::domain::color_grading::DragoParams { l_max: 1.0, bias };
    crate::domain::color_grading::tonemap_drago(pixels, info, &params)
}
}

impl GpuFilter for TonemapDragoParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/tonemap_drago_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&1.0f32.to_le_bytes()); // l_max
        params.extend_from_slice(&self.bias.to_le_bytes());

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

