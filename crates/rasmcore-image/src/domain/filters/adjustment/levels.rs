//! Filter: levels (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Levels adjustment: remap [black, white] input range with gamma curve.
/// Matches ImageMagick `-level black%,white%,gamma`.

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Levels adjustment — remap input black/white points with gamma
pub struct LevelsParams {
    /// Input black point (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 0.0)]
    pub black_point: f32,
    /// Input white point (0-100%)
    #[param(min = 0.0, max = 100.0, step = 0.1, default = 100.0)]
    pub white_point: f32,
    /// Gamma correction (0.1-10.0, 1.0 = linear)
    #[param(min = 0.1, max = 10.0, step = 0.01, default = 1.0)]
    pub gamma: f32,
}
impl LutPointOp for LevelsParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Levels {
            black: self.black_point / 100.0,
            white: self.white_point / 100.0,
            gamma: self.gamma,
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "levels",
    category = "adjustment",
    reference = "input/output level remapping",
    point_op = "true"
)]
pub fn levels(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LevelsParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let black_point = config.black_point;
    let white_point = config.white_point;
    let gamma = config.gamma;

    // Convert percentage to fraction
    crate::domain::point_ops::levels(
        pixels,
        info,
        black_point / 100.0,
        white_point / 100.0,
        gamma,
    )
}

impl crate::domain::filter_traits::GpuFilter for LevelsParams {
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
            include_str!("../../../shaders/levels_f32.wgsl").to_string()
        });
        let black = self.black_point / 100.0;
        let white = self.white_point / 100.0;
        let range = (white - black).max(1e-6);
        let range_inv = 1.0 / range;
        let inv_gamma = 1.0 / self.gamma.max(0.001);
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&black.to_le_bytes());
        params.extend_from_slice(&range_inv.to_le_bytes());
        params.extend_from_slice(&inv_gamma.to_le_bytes());
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
