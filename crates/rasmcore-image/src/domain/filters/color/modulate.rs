//! Filter: modulate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Combined brightness/saturation/hue adjustment in HSB color space.
///
/// IM equivalent: -modulate brightness,saturation,hue
/// Uses HSB (same as HSV where B=V=max(R,G,B)), not HSL.
/// Identity at (100, 100, 0).

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// HSB modulate — combined brightness, saturation, hue adjustment.
pub struct ModulateParams {
    /// Brightness percentage (100 = unchanged, 0 = black, 200 = 2x bright)
    #[param(min = 0.0, max = 200.0, step = 1.0, default = 100.0)]
    pub brightness: f32,
    /// Saturation percentage (100 = unchanged, 0 = grayscale, 200 = 2x saturated)
    #[param(min = 0.0, max = 200.0, step = 1.0, default = 100.0)]
    pub saturation: f32,
    /// Hue rotation in degrees (0 = unchanged)
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0)]
    pub hue: f32,
}
impl ColorLutOp for ModulateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Modulate {
            brightness: self.brightness,
            saturation: self.saturation,
            hue: self.hue,
        }
        .to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "modulate",
    category = "color",
    reference = "luma-preserving HSL modulation",
    color_op = "true"
)]
pub fn modulate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ModulateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let brightness = config.brightness;
    let saturation = config.saturation;
    let hue = config.hue;

    apply_color_op(
        pixels,
        info,
        &ColorOp::Modulate {
            brightness: brightness / 100.0,
            saturation: saturation / 100.0,
            hue,
        },
    )
}

impl crate::domain::filter_traits::GpuFilter for ModulateParams {
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
        // Reuse the hue_rotate shader's HSL helpers but with modulate entry point
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/modulate_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&(self.brightness / 100.0).to_le_bytes());
        params.extend_from_slice(&(self.saturation / 100.0).to_le_bytes());
        params.extend_from_slice(&self.hue.to_le_bytes());
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
