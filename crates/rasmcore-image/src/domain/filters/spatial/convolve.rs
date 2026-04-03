//! Filter: convolve (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::GpuFilter;


#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ConvolveParams {
    pub kw: u32,
    pub kh: u32,
    pub divisor: f32,
}

impl ConvolveParams {
    /// Build GPU operations for convolution with the given kernel weights.
    ///
    /// The kernel slice must have exactly `kw * kh` elements.
    /// Weights are pre-divided by `divisor` before uploading to the GPU.
    pub fn gpu_ops_with_kernel(
        &self,
        width: u32,
        height: u32,
        kernel: &[f32],
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static CONVOLVE_U32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/convolve.wgsl")));
        static CONVOLVE_F32: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops_f32(include_str!("../../../shaders/convolve_f32.wgsl")));

        let kw = self.kw;
        let kh = self.kh;
        if kw == 0 || kh == 0 || kernel.len() != (kw * kh) as usize {
            return None;
        }

        // Pre-divide kernel weights by divisor
        let inv_div = if self.divisor.abs() > f32::EPSILON { 1.0 / self.divisor } else { 1.0 };
        let mut kernel_buf = Vec::with_capacity(kernel.len() * 4);
        for &w in kernel {
            kernel_buf.extend_from_slice(&(w * inv_div).to_le_bytes());
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&kw.to_le_bytes());
        params.extend_from_slice(&kh.to_le_bytes());

        let shader = match buffer_format {
            BufferFormat::F32Vec4 => CONVOLVE_F32.clone(),
            BufferFormat::U32Packed => CONVOLVE_U32.clone(),
        };

        Some(vec![GpuOp::Compute {
            shader,
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![kernel_buf],
            buffer_format,
        }])
    }
}

impl GpuFilter for ConvolveParams {
    fn gpu_ops(
        &self,
        _width: u32,
        _height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        // Kernel weights are external to ConvolveParams; callers with kernel
        // data should use `gpu_ops_with_kernel()` directly.
        None
    }

    fn gpu_ops_with_format(
        &self,
        _width: u32,
        _height: u32,
        _buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        None
    }
}

impl InputRectProvider for ConvolveParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.kw.max(self.kh) / 2;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

#[rasmcore_macros::register_filter(
    name = "convolve",
    category = "spatial",
    reference = "custom NxN kernel convolution"
)]
pub fn convolve(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    kernel: &[f32],
    config: &ConvolveParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.kw.max(config.kh) / 2;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let kw = config.kw;
    let kh = config.kh;
    let divisor = config.divisor;

    let kw = kw as usize;
    let kh = kh as usize;
    if kw.is_multiple_of(2) || kh.is_multiple_of(2) || kw * kh != kernel.len() {
        return Err(ImageError::InvalidParameters(
            "kernel dimensions must be odd and match kernel length".into(),
        ));
    }
    validate_format(info.format)?;

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);

    // For u8 formats, use the optimized u8 path (with WASM SIMD)
    if !is_16bit(info.format) && !is_float(info.format) {
        if let Some((row_k, col_k)) = is_separable(kernel, kw, kh) {
            let result = convolve_separable(pixels, w, h, ch, &row_k, &col_k, divisor)?;
            return Ok(crop_to_request(&result, expanded, request, info.format));
        }

        let rw = kw / 2;
        let rh = kh / 2;
        let inv_div = 1.0 / divisor;
        let padded = pad_reflect(pixels, w, h, ch, rw.max(rh));
        let pw = w + 2 * rw.max(rh);
        let mut out = vec![0u8; pixels.len()];
        let pad = rw.max(rh);

        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let mut sum = 0.0f32;
                    for ky in 0..kh {
                        let row_off = (y + pad - rh + ky) * pw * ch;
                        for kx in 0..kw {
                            let px_off = row_off + (x + pad - rw + kx) * ch + c;
                            sum += kernel[ky * kw + kx] * padded[px_off] as f32;
                        }
                    }
                    out[(y * w + x) * ch + c] = (sum * inv_div).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        return Ok(crop_to_request(&out, expanded, request, info.format));
    }

    // Non-u8 formats: convert to f32 samples, process in f32, convert back
    let samples = pixels_to_f32_samples(pixels, info.format);

    if let Some((row_k, col_k)) = is_separable(kernel, kw, kh) {
        let result_f32 = convolve_separable_f32(&samples, w, h, ch, &row_k, &col_k, divisor)?;
        let result = f32_samples_to_pixels(&result_f32, info.format);
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    // General 2D convolution on f32 samples
    let rw = kw / 2;
    let rh = kh / 2;
    let pad = rw.max(rh);
    let inv_div = 1.0 / divisor;

    // Pad f32 samples with reflect border
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    let mut padded = vec![0.0f32; pw * ph * ch];
    for py in 0..ph {
        let sy = reflect(py as i32 - pad as i32, h);
        for px in 0..pw {
            let sx = reflect(px as i32 - pad as i32, w);
            let src = (sy * w + sx) * ch;
            let dst = (py * pw + px) * ch;
            padded[dst..dst + ch].copy_from_slice(&samples[src..src + ch]);
        }
    }

    let mut out_f32 = vec![0.0f32; w * h * ch];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let row_off = (y + pad - rh + ky) * pw * ch;
                    for kx in 0..kw {
                        let px_off = row_off + (x + pad - rw + kx) * ch + c;
                        sum += kernel[ky * kw + kx] * padded[px_off];
                    }
                }
                out_f32[(y * w + x) * ch + c] = sum * inv_div;
            }
        }
    }
    let result = f32_samples_to_pixels(&out_f32, info.format);
    Ok(crop_to_request(&result, expanded, request, info.format))
}
