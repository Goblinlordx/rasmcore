//! Filmic/ACES tone mapping (Narkowicz 2015 / Hable 2010).

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

#[derive(rasmcore_macros::Filter, Clone)]
/// Filmic/ACES tone mapping (Narkowicz 2015)
#[filter(name = "tonemap_filmic", category = "tonemapping", group = "tonemap", variant = "filmic", reference = "Hable 2010 Uncharted 2 filmic curve")]
pub struct TonemapFilmicParams {
    /// Shoulder strength (a coefficient)
    #[param(min = 0.0, max = 10.0, step = 0.01, default = 2.51)]
    pub shoulder_strength: f32,
    /// Linear strength (b coefficient)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.03)]
    pub linear_strength: f32,
    /// Linear angle (c coefficient)
    #[param(min = 0.0, max = 10.0, step = 0.01, default = 2.43)]
    pub linear_angle: f32,
    /// Toe strength (d coefficient)
    #[param(min = 0.0, max = 2.0, step = 0.01, default = 0.59)]
    pub toe_strength: f32,
    /// Toe numerator (e coefficient)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.14)]
    pub toe_numerator: f32,
}

impl CpuFilter for TonemapFilmicParams {
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
    let shoulder_strength = self.shoulder_strength;
    let linear_strength = self.linear_strength;
    let linear_angle = self.linear_angle;
    let toe_strength = self.toe_strength;
    let toe_numerator = self.toe_numerator;

    let params = crate::domain::color_grading::FilmicParams {
        a: shoulder_strength,
        b: linear_strength,
        c: linear_angle,
        d: toe_strength,
        e: toe_numerator,
    };
    crate::domain::color_grading::tonemap_filmic(pixels, info, &params)
}
}

impl GpuFilter for TonemapFilmicParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/tonemap_filmic_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.shoulder_strength.to_le_bytes());
        params.extend_from_slice(&self.linear_strength.to_le_bytes());
        params.extend_from_slice(&self.linear_angle.to_le_bytes());
        params.extend_from_slice(&self.toe_strength.to_le_bytes());
        params.extend_from_slice(&self.toe_numerator.to_le_bytes());
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

