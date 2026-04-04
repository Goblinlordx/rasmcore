//! Filter: vignette_powerlaw (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Power-law vignette — simple radial falloff.
///
/// Multiplies each pixel by `1.0 - strength * (dist / max_dist)^falloff`.
/// This is a computationally cheap alternative to the Gaussian vignette
/// with a different aesthetic (smooth polynomial falloff vs. Gaussian).
#[allow(clippy::too_many_arguments)]
/// Parameters for the power-law vignette mode.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "vignette_powerlaw", category = "enhancement", group = "vignette", variant = "powerlaw", reference = "power-law radial falloff")]
pub struct VignettePowerlawParams {
    /// Darkening strength (0=none, 1=fully black at corners)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
    /// Radial falloff exponent (higher = sharper transition)
    #[param(min = 0.5, max = 5.0, step = 0.1, default = 2.0)]
    pub falloff: f32,
    /// Full canvas width
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_width: u32,
    /// Full canvas height
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub full_height: u32,
    /// X offset
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub offset_x: u32,
    /// Y offset
    #[param(min = 0, max = 65535, step = 1, default = 0, hint = "rc.pixels")]
    pub offset_y: u32,
}

impl CpuFilter for VignettePowerlawParams {
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
    let strength = self.strength;
    let falloff = self.falloff;
    let full_width = self.full_width;
    let full_height = self.full_height;
    let offset_x = self.offset_x;
    let offset_y = self.offset_y;

    validate_format(info.format)?;

    let ch = channels(info.format);
    let color_ch = if ch == 4 { 3 } else { ch };
    let w = info.width as usize;
    let h = info.height as usize;
    let fw = full_width as f64;
    let fh = full_height as f64;
    let cx = fw / 2.0;
    let cy = fh / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    let strength_d = strength as f64;
    let falloff_d = falloff as f64;

    // Compute factor for each pixel position (shared between u8 and f32 paths)
    let compute_factor = |row: usize, col: usize| -> f64 {
        let abs_y = (offset_y as usize + row) as f64 + 0.5;
        let abs_x = (offset_x as usize + col) as f64 + 0.5;
        let dy = abs_y - cy;
        let dx = abs_x - cx;
        let dist = (dx * dx + dy * dy).sqrt();
        let t = (dist / max_dist).powf(falloff_d);
        1.0 - strength_d * t
    };

    if is_f32(info.format) {
        let mut samples: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for row in 0..h {
            for col in 0..w {
                let factor = compute_factor(row, col) as f32;
                let idx = (row * w + col) * ch;
                for c in 0..color_ch {
                    samples[idx + c] *= factor;
                }
            }
        }
        return Ok(samples.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let mut result = pixels.to_vec();
    for row in 0..h {
        for col in 0..w {
            let factor = compute_factor(row, col);
            let idx = (row * w + col) * ch;
            for c in 0..color_ch {
                let v = result[idx + c] as f64 * factor;
                result[idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}
}

impl GpuFilter for VignettePowerlawParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/vignette_powerlaw_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.strength.to_le_bytes());
        params.extend_from_slice(&self.falloff.to_le_bytes());
        params.extend_from_slice(&self.full_width.to_le_bytes());
        params.extend_from_slice(&self.full_height.to_le_bytes());
        params.extend_from_slice(&self.offset_x.to_le_bytes());
        params.extend_from_slice(&self.offset_y.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: SHADER_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}

