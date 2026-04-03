//! Filter: wave (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Wave: sinusoidal displacement along one axis.
///
/// Displaces pixels sinusoidally: horizontal wave shifts rows up/down,
/// vertical wave shifts columns left/right.
///
/// Equivalent to ImageMagick `-wave {amplitude}x{wavelength}`.

#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "wave", category = "distortion", reference = "sinusoidal wave displacement")]
pub struct WaveParams {
    /// Displacement amplitude in pixels
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 10.0)]
    pub amplitude: f32,
    /// Wavelength in pixels
    #[param(min = 1.0, max = 500.0, step = 5.0, default = 50.0)]
    pub wavelength: f32,
    /// Vertical wave (1.0) vs horizontal (0.0)
    #[param(min = 0.0, max = 1.0, step = 1.0, default = 0.0)]
    pub vertical: f32,
}

impl CpuFilter for WaveParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;

        let amplitude = self.amplitude;
        let wl = if self.wavelength.abs() < 1e-6 { 1.0 } else { self.wavelength };
        let is_vert = self.vertical >= 0.5;
        let overlap = amplitude.ceil() as u32 + 1;
        let dummy_j = crate::domain::ewa::JACOBIAN_IDENTITY;

        // Sampling: Bilinear — IM implements -wave in effect.c with bilinear,
        // not in distort.c with EWA. Bilinear gives exact match (MAE 0.00 vs IM).
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(overlap),
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let two_pi = std::f32::consts::TAU;
                if is_vert {
                    (xf - amplitude * (two_pi * yf / wl).sin(), yf)
                } else {
                    (xf, yf - amplitude * (two_pi * xf / wl).sin())
                }
            },
            &|_xf, _yf| dummy_j,
        )
    }
}

impl GpuFilter for WaveParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::F32Vec4)
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
        static WAVE_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling_f32(include_str!("../../../shaders/wave_f32.wgsl")));

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amplitude.to_le_bytes());
        params.extend_from_slice(&self.wavelength.to_le_bytes());
        params.extend_from_slice(&self.vertical.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let shader = WAVE_F32.clone();

        Some(vec![GpuOp::Compute {
            shader,
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}

