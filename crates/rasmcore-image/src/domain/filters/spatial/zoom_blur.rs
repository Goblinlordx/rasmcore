//! Filter: zoom_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "zoom_blur", category = "spatial",
    group = "blur",
    variant = "zoom",
    reference = "radial kernel simulating lens zoom"
)]
pub struct ZoomBlurParams {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)]
    pub factor: f32,
}

impl CpuFilter for ZoomBlurParams {
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

        if self.factor == 0.0 {
            return Ok(pixels.to_vec());
        }

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;

        // Convert to f32 samples [0,1] for format-agnostic processing
        let samples = pixels_to_f32_samples(pixels, info.format);
        let mut out = vec![0.0f32; w * h * ch];

        for py in 0..h {
            for px in 0..w {
                let x_start = px as f32;
                let y_start = py as f32;
                let x_end = x_start + (cx - x_start) * self.factor;
                let y_end = y_start + (cy - y_start) * self.factor;

                let dist = ((x_end - x_start).powi(2) + (y_end - y_start).powi(2)).sqrt();
                let mut xy_len = (dist.ceil() as usize) + 1;
                xy_len = xy_len.max(3);
                if xy_len > 100 {
                    xy_len = (100 + ((xy_len - 100) as f32).sqrt() as usize).min(200);
                }

                let inv_len = 1.0 / xy_len as f32;
                let dxx = (x_end - x_start) * inv_len;
                let dyy = (y_end - y_start) * inv_len;

                let mut ix = x_start;
                let mut iy = y_start;
                let mut accum = vec![0.0f32; ch];

                for _ in 0..xy_len {
                    let fx = ix.floor();
                    let fy = iy.floor();
                    let dx = ix - fx;
                    let dy = iy - fy;
                    let x0 = (fx as i32).clamp(0, w as i32 - 1) as usize;
                    let y0 = (fy as i32).clamp(0, h as i32 - 1) as usize;
                    let x1 = ((fx as i32) + 1).clamp(0, w as i32 - 1) as usize;
                    let y1 = ((fy as i32) + 1).clamp(0, h as i32 - 1) as usize;

                    for c in 0..ch {
                        let p00 = samples[(y0 * w + x0) * ch + c];
                        let p10 = samples[(y0 * w + x1) * ch + c];
                        let p01 = samples[(y1 * w + x0) * ch + c];
                        let p11 = samples[(y1 * w + x1) * ch + c];
                        let mix0 = dy * (p01 - p00) + p00;
                        let mix1 = dy * (p11 - p10) + p10;
                        accum[c] += dx * (mix1 - mix0) + mix0;
                    }

                    ix += dxx;
                    iy += dyy;
                }

                let dst = (py * w + px) * ch;
                for c in 0..ch {
                    out[dst + c] = accum[c] * inv_len;
                }
            }
        }

        Ok(f32_samples_to_pixels(&out, info.format))
    }
}

impl GpuFilter for ZoomBlurParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::F32Vec4)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, _buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;
        static ZOOM_BLUR_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling_f32(include_str!("../../../shaders/zoom_blur_f32.wgsl")));

        if self.factor == 0.0 {
            return None;
        }
        let samples = ((self.factor.abs() * 64.0).ceil() as u32).clamp(8, 128);

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.center_x.to_le_bytes());
        params.extend_from_slice(&self.center_y.to_le_bytes());
        params.extend_from_slice(&self.factor.to_le_bytes());
        params.extend_from_slice(&samples.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let shader = ZOOM_BLUR_F32.clone();

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
