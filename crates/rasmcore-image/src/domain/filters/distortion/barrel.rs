//! Filter: barrel (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


/// Barrel distortion: apply radial polynomial distortion.
/// `r_distorted = r * (1 + k1*r² + k2*r⁴)`.
/// `k1 > 0` = barrel, `k1 < 0` = pincushion.
/// This is the inverse of the `undistort` correction filter.
/// Matches ImageMagick `-distort Barrel` normalization: `rscale = 2/min(w,h)`.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "barrel", category = "distortion", reference = "Brown-Conrady radial distortion model")]
pub struct BarrelParams {
    /// Radial distortion coefficient (positive = barrel, negative = pincushion)
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.3, hint = "rc.signed_slider")]
    pub k1: f32,
    /// Higher-order radial coefficient
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.0, hint = "rc.signed_slider")]
    pub k2: f32,
}

impl CpuFilter for BarrelParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;

        let k1 = self.k1;
        let k2 = self.k2;
        // IM: center = w/2, rscale = 2/min(w,h), pixel-center convention (i+0.5)
        let wf = info.width as f64;
        let hf = info.height as f64;
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let rscale = 2.0 / wf.min(hf);
        // IM denormalizes coefficients: A *= rscale³, B *= rscale²
        let a_coeff = k1 as f64 * rscale * rscale * rscale;
        let b_coeff = k2 as f64 * rscale * rscale;

        let distort_f64 = move |ox: f64, oy: f64| -> (f64, f64) {
            let di = ox - cx;
            let dj = oy - cy;
            let dr = (di * di + dj * dj).sqrt();
            let df = a_coeff * dr * dr * dr + b_coeff * dr * dr + 1.0;
            (di * df + cx, dj * df + cy)
        };

        // Sampling: EwaClamp — IM barrel uses EWA with -virtual-pixel Edge (edge-clamp).
        // EwaClamp matches this border behavior. MAE ~8.24 vs IM (residual from
        // Robidoux vs Laguerre kernel and k1/k2 coefficient mapping differences).
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::FullImage,
            DistortionSampling::EwaClamp,
            &|xf, yf| {
                // IM pixel-center: d.x = i + 0.5
                let xf64 = xf as f64 + 0.5;
                let yf64 = yf as f64 + 0.5;
                let (sx, sy) = distort_f64(xf64, yf64);
                (sx as f32, sy as f32)
            },
            &|xf, yf| {
                let xf64 = xf as f64 + 0.5;
                let yf64 = yf as f64 + 0.5;
                let h_step = 0.5;
                let (sx_px, sy_px) = distort_f64(xf64 + h_step, yf64);
                let (sx_mx, sy_mx) = distort_f64(xf64 - h_step, yf64);
                let (sx_py, sy_py) = distort_f64(xf64, yf64 + h_step);
                let (sx_my, sy_my) = distort_f64(xf64, yf64 - h_step);
                let inv_2h = 1.0 / (2.0 * h_step);
                [
                    [
                        ((sx_px - sx_mx) * inv_2h) as f32,
                        ((sx_py - sx_my) * inv_2h) as f32,
                    ],
                    [
                        ((sy_px - sy_mx) * inv_2h) as f32,
                        ((sy_py - sy_my) * inv_2h) as f32,
                    ],
                ]
            },
        )
    }
}

impl GpuFilter for BarrelParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static BARREL_U32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/barrel.wgsl")));
        static BARREL_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling_f32(include_str!("../../../shaders/barrel_f32.wgsl")));

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.k1.to_le_bytes());
        params.extend_from_slice(&self.k2.to_le_bytes());

        let (shader, fmt) = match buffer_format {
            BufferFormat::F32Vec4 => (BARREL_F32.clone(), BufferFormat::F32Vec4),
            _ => (BARREL_U32.clone(), BufferFormat::U32Packed),
        };

        Some(vec![GpuOp::Compute {
            shader,
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: fmt,
        }])
    }
}

