//! Filter: chromatic_aberration (category: effect)
//!
//! Simulates lateral chromatic aberration (transverse CA) by radially displacing
//! R and B channels away from the image center. Amount increases with distance
//! from center, mimicking real lens CA behavior. Green channel stays in place.
//! This is a creative simulation, not a correction filter (cf. OpenCV undistort).

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Parameters for the chromatic aberration simulation.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "chromatic_aberration", category = "effect", reference = "lateral chromatic aberration simulation")]
pub struct ChromaticAberrationParams {
    /// CA strength — radial pixel shift at image corners (0 = none)
    #[param(min = 0.0, max = 20.0, step = 0.5, default = 3.0)]
    pub strength: f32,
}

const CA_WGSL: &str = include_str!("../../../shaders/chromatic_aberration.wgsl");

impl rasmcore_pipeline::GpuCapable for ChromaticAberrationParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        if self.strength.abs() < 1e-6 {
            return None; // identity — no GPU needed
        }
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.strength.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(&[rasmcore_gpu_shaders::PIXEL_OPS], CA_WGSL),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}

impl CpuFilter for ChromaticAberrationParams {
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
            "chromatic_aberration requires RGB".into(),
        ));
    }

    let strength = self.strength;
    if strength.abs() < 1e-6 {
        return Ok(pixels.to_vec());
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    let r_scale = strength / max_dist;
    let b_scale = -strength / max_dist;

    // Shared per-pixel logic: compute source coords for R and B channels
    #[inline]
    fn ca_src(x: usize, y: usize, cx: f32, cy: f32, scale: f32, w: usize, h: usize) -> (usize, usize) {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        let shift = dist * scale;
        let inv = 1.0 / dist.max(1e-6);
        let sx = (x as f32 + dx * inv * shift).round() as isize;
        let sy = (y as f32 + dy * inv * shift).round() as isize;
        (sx.clamp(0, w as isize - 1) as usize, sy.clamp(0, h as isize - 1) as usize)
    }

    if is_f32(info.format) {
        let samples: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = vec![0.0f32; samples.len()];
        for y in 0..h {
            for x in 0..w {
                let dst = (y * w + x) * ch;
                let (rsx, rsy) = ca_src(x, y, cx, cy, r_scale, w, h);
                out[dst] = samples[(rsy * w + rsx) * ch];
                out[dst + 1] = samples[dst + 1]; // green unchanged
                let (bsx, bsy) = ca_src(x, y, cx, cy, b_scale, w, h);
                out[dst + 2] = samples[(bsy * w + bsx) * ch + 2];
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
            let dst = (y * w + x) * ch;
            let (rsx, rsy) = ca_src(x, y, cx, cy, r_scale, w, h);
            out[dst] = pixels[(rsy * w + rsx) * ch];
            out[dst + 1] = pixels[dst + 1];
            let (bsx, bsy) = ca_src(x, y, cx, cy, b_scale, w, h);
            out[dst + 2] = pixels[(bsy * w + bsx) * ch + 2];
            if ch == 4 { out[dst + 3] = pixels[dst + 3]; }
        }
    }
    Ok(out)
}
}

