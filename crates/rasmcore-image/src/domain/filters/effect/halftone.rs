//! Filter: halftone (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "halftone", category = "effect", reference = "CMYK-style halftone dot pattern")]
pub struct HalftoneParams {
    pub dot_size: f32,
    pub angle_offset: f32,
}

impl CpuFilter for HalftoneParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let dot_size = self.dot_size;
    let angle_offset = self.angle_offset;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let ds = dot_size.max(1.0);

    // Standard CMYK screen angles (degrees)
    let angles_deg = [
        15.0 + angle_offset, // Cyan
        75.0 + angle_offset, // Magenta
        0.0 + angle_offset,  // Yellow
        45.0 + angle_offset, // Key (Black)
    ];
    let angles_rad: Vec<f32> = angles_deg.iter().map(|a| a.to_radians()).collect();

    // Frequency in pixels (dots per pixel = 1/dot_size)
    let freq = std::f32::consts::PI / ds;

    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * ch;

            // Get RGB
            let r = pixels[off] as f32 / 255.0;
            let g = if ch >= 3 {
                pixels[off + 1] as f32 / 255.0
            } else {
                r
            };
            let b = if ch >= 3 {
                pixels[off + 2] as f32 / 255.0
            } else {
                r
            };

            // RGB → CMYK
            let k = 1.0 - r.max(g).max(b);
            let (c_val, m_val, y_val) = if k >= 1.0 {
                (0.0, 0.0, 0.0)
            } else {
                let inv_k = 1.0 / (1.0 - k);
                (
                    (1.0 - r - k) * inv_k,
                    (1.0 - g - k) * inv_k,
                    (1.0 - b - k) * inv_k,
                )
            };
            let cmyk = [c_val, m_val, y_val, k];

            // Apply halftone screen per CMYK channel
            let mut screened = [0.0f32; 4];
            let xf = x as f32;
            let yf = y as f32;
            for i in 0..4 {
                let cos_a = angles_rad[i].cos();
                let sin_a = angles_rad[i].sin();
                // Rotated coordinates
                let rx = xf * cos_a + yf * sin_a;
                let ry = -xf * sin_a + yf * cos_a;
                // Sine-wave screen threshold
                let screen = ((rx * freq).sin() * (ry * freq).sin() + 1.0) * 0.5;
                screened[i] = if cmyk[i] > screen { 1.0 } else { 0.0 };
            }

            // CMYK → RGB: R = (1-C)(1-K), G = (1-M)(1-K), B = (1-Y)(1-K)
            let ro = ((1.0 - screened[0]) * (1.0 - screened[3]) * 255.0).round() as u8;
            let go = ((1.0 - screened[1]) * (1.0 - screened[3]) * 255.0).round() as u8;
            let bo = ((1.0 - screened[2]) * (1.0 - screened[3]) * 255.0).round() as u8;

            if ch == 1 {
                // Grayscale: use luminance
                out[off] = ((ro as u16 * 77 + go as u16 * 150 + bo as u16 * 29 + 128) >> 8) as u8;
            } else {
                out[off] = ro;
                out[off + 1] = go;
                out[off + 2] = bo;
                if ch == 4 {
                    out[off + 3] = pixels[off + 3]; // preserve alpha
                }
            }
        }
    }

    Ok(out)
}
}

impl GpuFilter for HalftoneParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        None
    }

    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        if buffer_format != rasmcore_pipeline::gpu::BufferFormat::F32Vec4 {
            return None;
        }
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/halftone_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.dot_size.max(1.0).to_le_bytes());
        params.extend_from_slice(&self.angle_offset.to_le_bytes());
        Some(vec![GpuOp::Compute {
            shader: SHADER.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::gpu::BufferFormat::F32Vec4,
        }])
    }
}

