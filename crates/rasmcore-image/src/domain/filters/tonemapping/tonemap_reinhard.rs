//! Reinhard photographic tone reproduction (Reinhard et al. 2002).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for Reinhard tone mapping (empty — no user controls).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct TonemapReinhardParams {}

impl rasmcore_pipeline::GpuCapable for TonemapReinhardParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        use std::sync::LazyLock;
        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/tonemap_reinhard_f32.wgsl"
            ))
        });

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: SHADER_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::gpu::BufferFormat::F32Vec4,
        }])
    }
}

#[rasmcore_macros::register_filter(
    name = "tonemap_reinhard",
    category = "tonemapping",
    group = "tonemap",
    variant = "reinhard",
    reference = "Reinhard et al. 2002 photographic tone reproduction"
)]
pub fn tonemap_reinhard_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    crate::domain::color_grading::tonemap_reinhard(pixels, info)
}
