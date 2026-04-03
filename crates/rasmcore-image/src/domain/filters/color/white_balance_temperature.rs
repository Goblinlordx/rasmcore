//! Filter: white_balance_temperature (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Temperature-based white balance adjustment.

/// Parameters for white balance temperature adjustment.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "white_balance_temperature", category = "color", group = "white_balance", variant = "temperature", reference = "Planckian locus color temperature")]
pub struct WhiteBalanceTemperatureParams {
    /// Color temperature in Kelvin
    #[param(
        min = 2000.0,
        max = 12000.0,
        step = 100.0,
        default = 6500.0,
        hint = "rc.temperature_k"
    )]
    pub temperature: f32,
    /// Tint adjustment
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub tint: f32,
}

impl CpuFilter for WhiteBalanceTemperatureParams {
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
    let temperature = self.temperature;
    let tint = self.tint;

    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            crate::domain::color_spaces::white_balance_temperature(p8, i8, temperature as f64, tint as f64)
        });
    }
    crate::domain::color_spaces::white_balance_temperature(pixels, info, temperature as f64, tint as f64)
}
}

impl GpuFilter for WhiteBalanceTemperatureParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/white_balance_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.temperature.to_le_bytes());
        params.extend_from_slice(&self.tint.to_le_bytes());

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

