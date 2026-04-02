//! Filter: spherize (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


/// Spherize: apply spherical projection for bulge/pinch effect.
/// `amount > 0` = bulge (fisheye), `amount < 0` = pinch.
/// `amount = 0` is identity.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "spherize", gpu = "true", category = "distortion", reference = "spherical bulge distortion")]
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

        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
            let cfg = self.clone();
            return process_via_8bit(&pixels, info16, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                let mut u = |_: Rect| Ok(px.to_vec());
                cfg.compute(r, &mut u, i8)
            });
        }

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
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static SPHERIZE: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/spherize.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: SPHERIZE.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}
