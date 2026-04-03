//! Filter: hue_rotate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Rotate hue by `degrees` (0-360). Works on RGB8 and RGBA8 images.

/// Parameters for hue_rotate.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct HueRotateParams {
    /// Hue rotation in degrees (0-360)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub degrees: f32,
}
impl ColorLutOp for HueRotateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::HueRotate(self.degrees).to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "hue_rotate",
    category = "color",
    reference = "HSV hue rotation",
    color_op = "true"
)]
pub fn hue_rotate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &HueRotateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let degrees = config.degrees;

    apply_color_op(pixels, info, &ColorOp::HueRotate(degrees))
}

impl crate::domain::filter_traits::GpuFilter for HueRotateParams {
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
            include_str!("../../../shaders/hue_rotate_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.degrees.to_le_bytes());
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
