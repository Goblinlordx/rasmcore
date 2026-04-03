//! Filter: colorize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Colorize with selectable method.
///
/// - `"w3c"` (default): Photoshop/W3C Color blend mode — SetLum/ClipColor
///   with BT.601 luma. Industry standard. Collapses to neutral at L=0/1.
/// - `"lab"`: CIELAB perceptual — replaces a*b* chrominance with parabolic
///   weighting by L*. Preserves subtle tint at highlights/shadows.
///   Based on the libvips/sharp tint() approach.

#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ColorizeParams {
    /// Target color to blend toward
    pub target: crate::domain::param_types::ColorRgb,
    /// Blend amount (0=none, 1=full tint)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
    /// Colorize method
    #[param(
        default = "w3c",
        hint = "rc.enum",
        options = "w3c:Photoshop/W3C standard — SetLum/ClipColor with BT.601 luma|lab:CIELAB perceptual — parabolic weighting, natural tint at extremes"
    )]
    pub method: String,
}
impl ColorLutOp for ColorizeParams {
    fn build_clut(&self) -> ColorLut3D {
        let target_norm = [
            self.target.r as f32 / 255.0,
            self.target.g as f32 / 255.0,
            self.target.b as f32 / 255.0,
        ];
        let amt = self.amount;
        // Empty method defaults to "w3c" (ConfigParams Default gives "" for String)
        let op = if self.method == "lab" {
            ColorOp::ColorizeLab(target_norm, amt)
        } else {
            ColorOp::Colorize(target_norm, amt)
        };
        op.to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "colorize",
    category = "color",
    reference = "W3C Compositing Level 1 / Photoshop Color blend mode",
    color_op = "true"
)]
pub fn colorize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ColorizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();

    let target_norm = [
        config.target.r as f32 / 255.0,
        config.target.g as f32 / 255.0,
        config.target.b as f32 / 255.0,
    ];
    let amount = config.amount.clamp(0.0, 1.0);

    // Empty method defaults to "w3c" (ConfigParams Default gives "" for String)
    let op = if config.method == "lab" {
        ColorOp::ColorizeLab(target_norm, amount)
    } else {
        ColorOp::Colorize(target_norm, amount)
    };
    apply_color_op(pixels, info, &op)
}

impl crate::domain::filter_traits::GpuFilter for ColorizeParams {
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
        // Only W3C mode has a GPU shader — LAB falls back to CPU
        if self.method == "lab" {
            return None;
        }
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/colorize_f32.wgsl").to_string()
        });
        let target_r = self.target.r as f32 / 255.0;
        let target_g = self.target.g as f32 / 255.0;
        let target_b = self.target.b as f32 / 255.0;
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&target_r.to_le_bytes());
        params.extend_from_slice(&target_g.to_le_bytes());
        params.extend_from_slice(&target_b.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
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
