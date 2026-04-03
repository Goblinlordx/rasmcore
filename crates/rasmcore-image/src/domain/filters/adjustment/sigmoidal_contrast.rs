//! Filter: sigmoidal_contrast (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Sigmoidal contrast: S-curve contrast adjustment.
/// Matches ImageMagick `-sigmoidal-contrast strengthxmidpoint%`.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Sigmoidal contrast — S-curve contrast adjustment
pub struct SigmoidalContrastParams {
    /// Contrast strength (0-20, 0 = identity)
    #[param(min = 0.0, max = 20.0, step = 0.1, default = 3.0)]
    pub strength: f32,
    /// Midpoint percentage (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 50.0)]
    pub midpoint: f32,
    /// true = increase contrast (sharpen), false = decrease contrast (soften)
    #[param(default = true)]
    pub sharpen: bool,
}
impl LutPointOp for SigmoidalContrastParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::SigmoidalContrast {
            strength: self.strength,
            midpoint: self.midpoint / 100.0,
            sharpen: self.sharpen,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "sigmoidal_contrast",
    category = "adjustment",
    reference = "sigmoidal transfer function contrast",
    point_op = "true"
)]
pub fn sigmoidal_contrast(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SigmoidalContrastParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let strength = config.strength;
    let midpoint = config.midpoint;
    let sharpen = config.sharpen;

    crate::domain::point_ops::sigmoidal_contrast(pixels, info, strength, midpoint / 100.0, sharpen)
}

impl crate::domain::filter_traits::GpuFilter for SigmoidalContrastParams {
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
            include_str!("../../../shaders/sigmoidal_contrast_f32.wgsl").to_string()
        });
        let strength = self.strength;
        let midpoint = self.midpoint / 100.0;
        let sig = |v: f32| -> f32 { 1.0 / (1.0 + (-strength * (v - midpoint)).exp()) };
        let sig_0 = sig(0.0);
        let sig_1 = sig(1.0);
        let sig_range = sig_1 - sig_0;
        let sig_range_inv = if sig_range.abs() > 1e-10 { 1.0 / sig_range } else { 1.0 };
        let is_sharpen: f32 = if self.sharpen { 1.0 } else { 0.0 };

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&strength.to_le_bytes());
        params.extend_from_slice(&midpoint.to_le_bytes());
        params.extend_from_slice(&sig_0.to_le_bytes());
        params.extend_from_slice(&sig_range_inv.to_le_bytes());
        params.extend_from_slice(&sig_range.to_le_bytes());
        params.extend_from_slice(&is_sharpen.to_le_bytes());
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
