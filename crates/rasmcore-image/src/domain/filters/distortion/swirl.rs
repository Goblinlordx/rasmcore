//! Filter: swirl (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


/// Swirl: rotate pixels around center with angle decreasing by distance.
/// Matches ImageMagick `-swirl {degrees}`:
/// - Default radius = max(width/2, height/2)
/// - Factor = 1 - sqrt(distance²) / radius, then angle = degrees * factor²
/// - Aspect ratio scaling for non-square images
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "swirl", gpu = "true", category = "distortion", reference = "vortex rotation distortion")]
pub struct SwirlParams {
    /// Rotation angle in degrees
    #[param(min = -720.0, max = 720.0, step = 5.0, default = 90.0, hint = "rc.signed_slider")]
    pub angle: f32,
    /// Radius of effect (0 = auto from image size)
    #[param(
        min = 0.0,
        max = 2000.0,
        step = 10.0,
        default = 0.0,
        hint = "rc.log_slider"
    )]
    pub radius: f32,
}

impl CpuFilter for SwirlParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;

        let w = info.width as f32;
        let h = info.height as f32;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let rad = if self.radius <= 0.0 { cx.max(cy) } else { self.radius };
        let angle_rad = self.angle.to_radians();
        let (scale_x, scale_y) = if info.width > info.height {
            (1.0f32, w / h)
        } else if info.height > info.width {
            (h / w, 1.0f32)
        } else {
            (1.0, 1.0)
        };

        // Sampling: Bilinear — IM implements -swirl in effect.c with bilinear,
        // not in distort.c with EWA. Bilinear gives exact match (MAE 0.00 vs IM).
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::FullImage,
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let dx = scale_x * (xf - cx);
                let dy = scale_y * (yf - cy);
                let dist = (dx * dx + dy * dy).sqrt();
                let t = (1.0 - dist / rad).max(0.0);
                let rot = angle_rad * t * t;
                let (cos_r, sin_r) = (rot.cos(), rot.sin());
                ((cos_r * dx - sin_r * dy) / scale_x + cx,
                 (sin_r * dx + cos_r * dy) / scale_y + cy)
            },
            &|xf, yf| {
                crate::domain::ewa::jacobian_swirl(xf, yf, cx, cy, angle_rad, rad, scale_x, scale_y)
            },
        )
    }
}

impl GpuFilter for SwirlParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static SWIRL: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/swirl.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.angle.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: SWIRL.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}
