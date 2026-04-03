//! Filter: spin_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Rotational (spin) blur around image center.
///
/// Matches ImageMagick's `-rotational-blur` algorithm exactly.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "spin_blur", category = "spatial")]
pub struct SpinBlurParams {
    /// Center X as fraction of width (0.0 = left, 0.5 = center, 1.0 = right)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Center Y as fraction of height (0.0 = top, 0.5 = center, 1.0 = bottom)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Rotation angle in degrees (max blur at edges)
    #[param(min = 0.0, max = 360.0, step = 0.5, default = 10.0)]
    pub angle: f32,
}

impl CpuFilter for SpinBlurParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;
        let info = &ImageInfo { width: request.width, height: request.height, ..*info };
        let pixels = pixels.as_slice();
        validate_format(info.format)?;

        if self.angle == 0.0 {
            return Ok(pixels.to_vec());
        }

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;
        let angle_rad = (self.angle as f64).to_radians();

        // Convert to f32 samples [0,1] for format-agnostic processing
        let samples = pixels_to_f32_samples(pixels, info.format);

        let half_diag = ((w as f64 / 2.0).powi(2) + (h as f64 / 2.0).powi(2)).sqrt();
        let mut n = (2.0 * (angle_rad * half_diag).ceil() + 2.0) as usize;
        if n.is_multiple_of(2) { n += 1; }
        n = n.max(3);

        let half_n = n / 2;
        let mut cos_table = vec![0.0f64; n];
        let mut sin_table = vec![0.0f64; n];
        for i in 0..n {
            let offset = angle_rad * (i as f64 - half_n as f64) / n as f64;
            cos_table[i] = offset.cos();
            sin_table[i] = offset.sin();
        }

        let inv_n = 1.0 / n as f64;
        let mut out = vec![0.0f32; w * h * ch];

        for py in 0..h {
            for px in 0..w {
                let dx = px as f64 - cx as f64;
                let dy = py as f64 - cy as f64;
                let mut accum = vec![0.0f64; ch];

                for j in 0..n {
                    let sx = cx as f64 + dx * cos_table[j] - dy * sin_table[j];
                    let sy = cy as f64 + dx * sin_table[j] + dy * cos_table[j];

                    let fx = sx.floor();
                    let fy = sy.floor();
                    let frac_x = sx - fx;
                    let frac_y = sy - fy;
                    let x0 = (fx as isize).clamp(0, w as isize - 1) as usize;
                    let y0 = (fy as isize).clamp(0, h as isize - 1) as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let y1 = (y0 + 1).min(h - 1);

                    let w00 = (1.0 - frac_x) * (1.0 - frac_y);
                    let w10 = frac_x * (1.0 - frac_y);
                    let w01 = (1.0 - frac_x) * frac_y;
                    let w11 = frac_x * frac_y;

                    let i00 = (y0 * w + x0) * ch;
                    let i10 = (y0 * w + x1) * ch;
                    let i01 = (y1 * w + x0) * ch;
                    let i11 = (y1 * w + x1) * ch;

                    for c in 0..ch {
                        accum[c] += samples[i00 + c] as f64 * w00
                            + samples[i10 + c] as f64 * w10
                            + samples[i01 + c] as f64 * w01
                            + samples[i11 + c] as f64 * w11;
                    }
                }

                let dst = (py * w + px) * ch;
                for c in 0..ch {
                    out[dst + c] = (accum[c] * inv_n) as f32;
                }
            }
        }

        Ok(f32_samples_to_pixels(&out, info.format))
    }
}

impl GpuFilter for SpinBlurParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::F32Vec4)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;
        static SPIN_BLUR_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling_f32(include_str!("../../../shaders/spin_blur_f32.wgsl")));

        let samples = ((self.angle.abs() * 180.0 / std::f32::consts::PI).ceil() as u32).clamp(8, 128);

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.center_x.to_le_bytes());
        params.extend_from_slice(&self.center_y.to_le_bytes());
        params.extend_from_slice(&self.angle.to_le_bytes());
        params.extend_from_slice(&samples.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let shader = SPIN_BLUR_F32.clone();

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

