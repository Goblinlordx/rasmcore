//! Filter: spherize (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


/// Spherize: apply spherical projection for bulge/pinch effect.
/// `amount > 0` = bulge (fisheye), `amount < 0` = pinch.
/// `amount = 0` is identity.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "spherize", category = "distortion", reference = "spherical bulge distortion")]
pub struct SpherizeParams {
    /// Bulge/pinch amount (-1 to 1, positive = bulge)
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.5, hint = "rc.signed_slider")]
    pub amount: f32,
}

impl CpuFilter for SpherizeParams {
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
        let radius = cx.min(cy);
        let amt = self.amount.clamp(-1.0, 1.0);

        // Sampling: Ewa — no direct IM equivalent. EWA suits the nonlinear radial
        // mapping (powf-based bulge/pinch has anisotropic stretching near the edge).
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::FullImage,
            DistortionSampling::Ewa,
            &|xf, yf| {
                let dx = (xf - cx) / radius;
                let dy = (yf - cy) / radius;
                let r = (dx * dx + dy * dy).sqrt();
                if r >= 1.0 || r == 0.0 {
                    // Outside effect radius or at center: identity mapping
                    (xf, yf)
                } else {
                    let new_r = if amt >= 0.0 {
                        r.powf(1.0 / (1.0 + amt))
                    } else {
                        r.powf(1.0 + amt.abs())
                    };
                    let scale = new_r / r;
                    (dx * scale * radius + cx, dy * scale * radius + cy)
                }
            },
            &|xf, yf| {
                crate::domain::ewa::jacobian_spherize(xf, yf, cx, cy, amt, radius)
            },
        )
    }
}

impl GpuFilter for SpherizeParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::F32Vec4)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SPHERIZE_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling_f32(include_str!("../../../shaders/spherize_f32.wgsl")));
        use rasmcore_gpu_shaders as shaders;

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: SPHERIZE_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}

