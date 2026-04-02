//! Filter: polar (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Polar: convert Cartesian image to polar-coordinate projection.
///
/// Maps the rectangular image into a polar representation where:
/// - Output x-axis represents angle (0 to 2π across width)
/// - Output y-axis represents radius (0 to max_radius across height)
///
/// Uses IM pixel-center convention (+0.5) and f64 precision.
/// Equivalent to ImageMagick `-distort DePolar "max_radius"`.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "polar", gpu = "true", category = "distortion", group = "distort_polar", variant = "to_polar", reference = "Cartesian to polar coordinate transform")]
pub struct PolarParams {}

impl CpuFilter for PolarParams {
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
            return process_via_8bit(&pixels, info16, |p8, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                let mut u = |_: Rect| Ok(p8.to_vec());
                Self{}.compute(r, &mut u, i8)
            });
        }

        // Use f64 throughout to match IM's double-precision pipeline
        let wf = info.width as f64;
        let hf = info.height as f64;
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let max_radius = cx.min(cy);
        let two_pi = std::f64::consts::TAU;

        // Sampling: Bilinear — matches IM DePolar empirically (MAE 1.95).
        // EWA gives worse parity (MAE 5.18) for the uniform polar-to-Cartesian mapping.
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::FullImage,
            DistortionSampling::Bilinear,
            &|xf, yf| {
                // IM pixel-center convention: d.x = i + 0.5
                let dx = xf as f64 + 0.5;
                let dy = yf as f64 + 0.5;
                // Invert depolar's angle mapping: angle = (dx - w/2) / w * 2π
                let angle = (dx - cx) / wf * two_pi;
                let radius = dy / hf * max_radius;
                // Invert depolar's atan2(ii, jj): ii = r*sin(a), jj = r*cos(a)
                let sx = (cx + radius * angle.sin() - 0.5) as f32;
                let sy = (cy + radius * angle.cos() - 0.5) as f32;
                (sx, sy)
            },
            &|xf, yf| {
                crate::domain::ewa::jacobian_polar(xf, yf,
                    info.width as f32, info.height as f32,
                    max_radius as f32)
            },
        )
    }
}

impl GpuFilter for PolarParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static POLAR: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/polar.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: POLAR.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: Default::default(),
        }])
    }
}
