//! Filter: emboss (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for emboss (empty — fixed kernel, no user controls).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct EmbossParams {}

impl rasmcore_pipeline::GpuCapable for EmbossParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        use std::sync::LazyLock;
        static EMBOSS_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/emboss_f32.wgsl"
            ))
        });

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: EMBOSS_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::gpu::BufferFormat::F32Vec4,
        }])
    }
}

#[rasmcore_macros::register_filter(
    name = "emboss",
    category = "effect",
    reference = "3D relief embossing via directional kernel"
)]
pub fn emboss(
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
    emboss_impl(&pixels, info)
}
