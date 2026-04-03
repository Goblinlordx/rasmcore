//! Filter: depolar (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// DePolar: convert polar-coordinate image back to Cartesian projection.
///
/// Inverse of `polar`: maps a polar representation back to rectangular.
/// For each output pixel, compute radius and angle from center,
/// then look up in the polar-space input image.
///
/// Equivalent to ImageMagick `-distort DePolar "max_radius"`.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "depolar", category = "distortion", group = "distort_polar", variant = "from_polar", reference = "polar to Cartesian coordinate transform")]
pub struct DepolarParams {}

impl CpuFilter for DepolarParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;

        // Use f64 throughout to match IM's double-precision pipeline
        let wf = info.width as f64;
        let hf = info.height as f64;
        // IM uses pixel-center convention: d.x = i + 0.5, center = w/2
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let max_radius = (wf * 0.5).min(hf * 0.5);
        let two_pi = std::f64::consts::TAU;
        let c6 = wf / two_pi;
        let c7 = hf / max_radius;
        let half_w = wf * 0.5;

        // Sampling: Ewa — matches IM Polar (distort.c EWA engine, MAE 2.55).
        // The Cartesian-to-polar mapping benefits from EWA's anisotropic filtering
        // near the center where radial lines converge.
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::FullImage,
            DistortionSampling::Ewa,
            &|xf, yf| {
                // IM pixel-center: d.x = i + 0.5
                let xf64 = xf as f64 + 0.5;
                let yf64 = yf as f64 + 0.5;
                let ii = xf64 - cx;
                let jj = yf64 - cy;
                let radius = (ii * ii + jj * jj).sqrt();
                let angle = ii.atan2(jj);
                let mut xx = angle / two_pi;
                xx -= xx.round();
                let sx = (xx * two_pi * c6 + half_w - 0.5) as f32;
                let sy = (radius * c7 - 0.5) as f32;
                (sx, sy)
            },
            &|xf, yf| {
                let xf64 = xf as f64 + 0.5;
                let yf64 = yf as f64 + 0.5;
                let ii = xf64 - cx;
                let jj = yf64 - cy;
                let r2 = ii * ii + jj * jj;
                if r2 < 1e-10 {
                    crate::domain::ewa::JACOBIAN_IDENTITY
                } else {
                    let r = r2.sqrt();
                    [
                        [(jj / r2 * c6) as f32, (-ii / r2 * c6) as f32],
                        [(ii / r * c7) as f32, (jj / r * c7) as f32],
                    ]
                }
            },
        )
    }
}

impl GpuFilter for DepolarParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static DEPOLAR: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/depolar.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: DEPOLAR.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: buffer_format,
        }])
    }
}
