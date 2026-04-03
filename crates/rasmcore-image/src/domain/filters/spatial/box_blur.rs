//! Filter: box_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;
use crate::domain::filter_traits::GpuFilter;

/// Box blur — separable uniform-weight kernel with O(1) running-sum.
///
/// Each output pixel is the mean of all pixels in a (2r+1)x(2r+1) window.
/// Implemented as two separable passes (horizontal then vertical), each
/// using a running sum: add the entering pixel, subtract the leaving pixel.
/// Cost is O(1) per pixel regardless of radius.
///
/// Reference: matches Photoshop's Box Blur and OpenCV's cv2.blur().

/// Parameters for box blur (uniform-weight kernel).
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "box_blur",
    category = "spatial",
    group = "blur",
    variant = "box",
    reference = "Photoshop Box Blur / OpenCV cv2.blur"
)]
pub struct BoxBlurParams {
    /// Blur radius in pixels (kernel width = 2*radius + 1)
    #[param(min = 1, max = 100, step = 1, default = 3)]
    pub radius: u32,
}

impl CpuFilter for BoxBlurParams {
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
        validate_format(info.format)?;

        if radius == 0 {
            return Ok(pixels.to_vec());
        }

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);
        let r = radius as usize;
        let inv_diam = 1.0 / (2 * r + 1) as f32;

        // Convert to f32 samples [0,1] for format-agnostic processing
        let samples = pixels_to_f32_samples(pixels, info.format);
        let color_ch = if ch == 4 { 3 } else { ch };

        // Helper: load pixel channels into [f32; 4] (zero-padded for ch < 4)
        #[inline(always)]
        fn load_pixel(src: &[f32], offset: usize, ch: usize) -> [f32; 4] {
            let mut p = [0.0f32; 4];
            for c in 0..ch.min(4) {
                p[c] = src[offset + c];
            }
            p
        }

        // Helper: store [f32; 4] as pixel channels, preserving alpha if RGBA
        #[inline(always)]
        fn store_pixel(dst: &mut [f32], offset: usize, vals: [f32; 4], ch: usize, alpha_src: &[f32]) {
            let color_ch = if ch == 4 { 3 } else { ch };
            for c in 0..color_ch {
                dst[offset + c] = vals[c];
            }
            if ch == 4 {
                dst[offset + 3] = alpha_src[offset + 3]; // preserve alpha
            }
        }

        // Horizontal pass — running sum across all channels simultaneously
        let mut hpass = vec![0.0f32; samples.len()];

        for y in 0..h {
            let row = y * w * ch;
            let mut sums = [0.0f32; 4];

            // Initialize sums for first pixel (clamped boundary)
            for k in 0..=r {
                let sx = k.min(w - 1);
                let p = load_pixel(&samples, row + sx * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] += p[c];
                }
            }
            for _ in 0..r {
                let p = load_pixel(&samples, row, color_ch);
                for c in 0..color_ch {
                    sums[c] += p[c];
                }
            }
            // Store first pixel
            let mut mean = [0.0f32; 4];
            for c in 0..color_ch {
                mean[c] = sums[c] * inv_diam;
            }
            store_pixel(&mut hpass, row, mean, ch, &samples);

            // Slide across row
            for x in 1..w {
                let add_x = (x + r).min(w - 1);
                let add = load_pixel(&samples, row + add_x * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] += add[c];
                }

                if x <= r {
                    let sub = load_pixel(&samples, row, color_ch);
                    for c in 0..color_ch {
                        sums[c] -= sub[c];
                    }
                } else {
                    let sub_x = x - r - 1;
                    let sub = load_pixel(&samples, row + sub_x * ch, color_ch);
                    for c in 0..color_ch {
                        sums[c] -= sub[c];
                    }
                }

                for c in 0..color_ch {
                    mean[c] = sums[c] * inv_diam;
                }
                store_pixel(&mut hpass, row + x * ch, mean, ch, &samples);
            }
        }

        // Vertical pass — running sum down columns, all channels simultaneously
        let mut out = vec![0.0f32; samples.len()];

        for x in 0..w {
            let mut sums = [0.0f32; 4];

            for k in 0..=r {
                let sy = k.min(h - 1);
                let p = load_pixel(&hpass, (sy * w + x) * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] += p[c];
                }
            }
            for _ in 0..r {
                let p = load_pixel(&hpass, x * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] += p[c];
                }
            }
            let mut mean = [0.0f32; 4];
            for c in 0..color_ch {
                mean[c] = sums[c] * inv_diam;
            }
            store_pixel(&mut out, x * ch, mean, ch, &hpass);

            for y in 1..h {
                let add_y = (y + r).min(h - 1);
                let add = load_pixel(&hpass, (add_y * w + x) * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] += add[c];
                }

                if y <= r {
                    let sub = load_pixel(&hpass, x * ch, color_ch);
                    for c in 0..color_ch {
                        sums[c] -= sub[c];
                    }
                } else {
                    let sub_y = y - r - 1;
                    let sub = load_pixel(&hpass, (sub_y * w + x) * ch, color_ch);
                    for c in 0..color_ch {
                        sums[c] -= sub[c];
                    }
                }

                for c in 0..color_ch {
                    mean[c] = sums[c] * inv_diam;
                }
                store_pixel(&mut out, (y * w + x) * ch, mean, ch, &hpass);
            }
        }

        let result = f32_samples_to_pixels(&out, info.format);
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for BoxBlurParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
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

        static BOX_BLUR_U32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/box_blur.wgsl")));
        static BOX_BLUR_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops_f32(include_str!("../../../shaders/box_blur_f32.wgsl")));

        if self.radius == 0 {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let shader = match buffer_format {
            BufferFormat::F32Vec4 => BOX_BLUR_F32.clone(),
            BufferFormat::U32Packed => BOX_BLUR_U32.clone(),
        };

        Some(vec![
            GpuOp::Compute {
                shader: shader.clone(),
                entry_point: "blur_h",
                workgroup_size: [256, 1, 1],
                params: params.clone(),
                extra_buffers: vec![],
                buffer_format,
            },
            GpuOp::Compute {
                shader,
                entry_point: "blur_v",
                workgroup_size: [1, 256, 1],
                params,
                extra_buffers: vec![],
                buffer_format,
            },
        ])
    }
}
