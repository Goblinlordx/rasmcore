//! Filter: film_grain (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Film grain simulation
pub struct FilmGrainParams {
    /// Grain amount (0 = none, 1 = heavy)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.3)]
    pub amount: f32,
    /// Grain size in pixels (1 = fine, 4+ = coarse)
    #[param(min = 0.5, max = 8.0, step = 0.1, default = 1.5)]
    pub size: f32,
    /// Random seed for deterministic output
    #[param(min = 0, max = 4294967295, step = 1, default = 0, hint = "rc.seed")]
    pub seed: u32,
}

#[rasmcore_macros::register_filter(
    name = "film_grain",
    category = "effect",
    reference = "photographic film grain overlay"
)]
pub fn film_grain_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &FilmGrainParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;
    let size = config.size;
    let seed = config.seed;

    let params = crate::domain::color_grading::FilmGrainParams {
        amount,
        size,
        color: false,
        seed,
    };
    crate::domain::color_grading::film_grain(pixels, info, &params)
}

impl crate::domain::filter_traits::GpuFilter for FilmGrainParams {
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
            include_str!("../../../shaders/film_grain_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
        params.extend_from_slice(&self.size.to_le_bytes());
        params.extend_from_slice(&self.seed.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
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
