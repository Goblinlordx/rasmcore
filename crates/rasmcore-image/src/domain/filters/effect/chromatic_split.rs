//! Filter: chromatic_split (category: effect)
//!
//! Separate RGB channels and apply independent spatial offsets, then recombine.
//! Creates a "prism" or "RGB split" look popular in social/consumer apps.
//! Reference: PicsArt/Pixlr RGB split effect.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Parameters for the chromatic split effect.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "chromatic_split", category = "effect", reference = "PicsArt/Pixlr RGB channel split")]
pub struct ChromaticSplitParams {
    /// Red channel X offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 5.0, hint = "rc.signed_slider")]
    pub red_dx: f32,
    /// Red channel Y offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub red_dy: f32,
    /// Green channel X offset in pixels (usually 0)
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub green_dx: f32,
    /// Green channel Y offset in pixels (usually 0)
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub green_dy: f32,
    /// Blue channel X offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = -5.0, hint = "rc.signed_slider")]
    pub blue_dx: f32,
    /// Blue channel Y offset in pixels
    #[param(min = -50.0, max = 50.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub blue_dy: f32,
}

const CHROMATIC_SPLIT_WGSL: &str = include_str!("../../../shaders/chromatic_split.wgsl");

impl rasmcore_pipeline::GpuCapable for ChromaticSplitParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.red_dx.to_le_bytes());
        params.extend_from_slice(&self.red_dy.to_le_bytes());
        params.extend_from_slice(&self.green_dx.to_le_bytes());
        params.extend_from_slice(&self.green_dy.to_le_bytes());
        params.extend_from_slice(&self.blue_dx.to_le_bytes());
        params.extend_from_slice(&self.blue_dy.to_le_bytes());

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(&[rasmcore_gpu_shaders::PIXEL_OPS], CHROMATIC_SPLIT_WGSL),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}

impl CpuFilter for ChromaticSplitParams {
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
    validate_format(info.format)?;

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "chromatic_split requires RGB".into(),
        ));
    }

    let offsets = [
        (self.red_dx, self.red_dy),
        (self.green_dx, self.green_dy),
        (self.blue_dx, self.blue_dy),
    ];

    if is_f32(info.format) {
        let samples: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = vec![0.0f32; samples.len()];
        for y in 0..h {
            for x in 0..w {
                let dst = (y * w + x) * ch;
                for c in 0..3 {
                    let (dx, dy) = offsets[c];
                    let sx = (x as f32 + dx).round() as isize;
                    let sy = (y as f32 + dy).round() as isize;
                    let sx = sx.clamp(0, w as isize - 1) as usize;
                    let sy = sy.clamp(0, h as isize - 1) as usize;
                    out[dst + c] = samples[(sy * w + sx) * ch + c];
                }
                if ch == 4 { out[dst + 3] = samples[dst + 3]; }
            }
        }
        return Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let mut out = vec![0u8; pixels.len()];
    for y in 0..h {
        for x in 0..w {
            let dst_idx = (y * w + x) * ch;
            for c in 0..3 {
                let (dx, dy) = offsets[c];
                let sx = (x as f32 + dx).round() as isize;
                let sy = (y as f32 + dy).round() as isize;
                let sx = sx.clamp(0, w as isize - 1) as usize;
                let sy = sy.clamp(0, h as isize - 1) as usize;
                out[dst_idx + c] = pixels[(sy * w + sx) * ch + c];
            }
            if ch == 4 { out[dst_idx + 3] = pixels[dst_idx + 3]; }
        }
    }
    Ok(out)
}
}

