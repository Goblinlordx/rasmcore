//! Filter: median (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Median filter with given radius. Window is (2*radius+1)^2.
///
/// Uses histogram sliding-window (Huang algorithm) for radius > 2 giving
/// O(1) amortized per pixel. Falls back to sorting for radius <= 2 where
/// the small window makes sorting faster than histogram maintenance.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "median", category = "spatial",
    group = "denoise",
    variant = "median",
    reference = "median rank filter"
)]
pub struct MedianParams {
    /// Filter radius in pixels
    #[param(min = 1, max = 20, step = 1, default = 3, hint = "rc.log_slider")]
    pub radius: u32,
}

impl CpuFilter for MedianParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = self.radius;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let info = &ImageInfo {
            width: expanded.width,
            height: expanded.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let radius = self.radius;

        if radius == 0 {
            return Ok(pixels.to_vec());
        }
        validate_format(info.format)?;

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);

        if !is_16bit(info.format) && !is_float(info.format) {
            // Fast u8 path using histogram/sort
            let result = if radius <= 2 {
                median_sort(pixels, w, h, ch, radius)?
            } else {
                median_histogram(pixels, w, h, ch, radius)?
            };
            return Ok(crop_to_request(&result, expanded, request, info.format));
        }

        // f32-native path for u16/f16/f32 formats
        let samples = pixels_to_f32_samples(pixels, info.format);
        let r = radius as i32;
        let window_size = ((2 * r + 1) * (2 * r + 1)) as usize;
        let median_pos = window_size / 2;
        let mut out_samples = vec![0.0f32; samples.len()];
        let mut window = Vec::with_capacity(window_size);

        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    window.clear();
                    for ky in -r..=r {
                        let sy = reflect(y as i32 + ky, h);
                        for kx in -r..=r {
                            let sx = reflect(x as i32 + kx, w);
                            window.push(samples[(sy * w + sx) * ch + c]);
                        }
                    }
                    window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    out_samples[(y * w + x) * ch + c] = window[median_pos];
                }
            }
        }

        let result = f32_samples_to_pixels(&out_samples, info.format);
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for MedianParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static MEDIAN_U32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/median.wgsl")));
        static MEDIAN_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops_f32(include_str!("../../../shaders/median_f32.wgsl")));

        if self.radius > 3 {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let (shader, fmt) = match buffer_format {
            BufferFormat::F32Vec4 => (MEDIAN_F32.clone(), BufferFormat::F32Vec4),
            _ => (MEDIAN_U32.clone(), BufferFormat::U32Packed),
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
