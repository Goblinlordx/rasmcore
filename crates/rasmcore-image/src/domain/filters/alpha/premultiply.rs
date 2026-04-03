//! Filter: premultiply (category: alpha)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for premultiply (empty — no user controls).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct PremultiplyParams {}

impl rasmcore_pipeline::GpuCapable for PremultiplyParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        use std::sync::LazyLock;
        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/premultiply_f32.wgsl"
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

/// Convert straight alpha to premultiplied alpha (RGBA8 only).
#[rasmcore_macros::register_filter(
    name = "premultiply",
    category = "alpha",
    group = "alpha",
    variant = "premultiply",
    reference = "premultiplied alpha conversion"
)]
pub fn premultiply(
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
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "premultiply requires RGBA8".into(),
        ));
    }
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let a = chunk[3] as u16;
        chunk[0] = ((chunk[0] as u16 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u16 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u16 * a + 127) / 255) as u8;
    }
    Ok(result)
}
