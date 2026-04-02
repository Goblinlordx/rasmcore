//! Filter: ca_remove (category: correction)
//!
//! Lateral chromatic aberration removal: radially shift R and B channels
//! toward the G channel to correct color fringing from lens CA.
//! Inverse of the chromatic_aberration effect — here we *correct* the shift.
//! Reference: Lightroom/Affinity Photo CA removal — radial R/B realignment.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for chromatic aberration removal.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct CaRemoveParams {
    /// Red channel radial shift correction (-0.01 to 0.01, negative = inward)
    #[param(min = -0.01, max = 0.01, step = 0.0005, default = 0.0, hint = "rc.signed_slider")]
    pub red_shift: f32,
    /// Blue channel radial shift correction (-0.01 to 0.01, negative = inward)
    #[param(min = -0.01, max = 0.01, step = 0.0005, default = 0.0, hint = "rc.signed_slider")]
    pub blue_shift: f32,
}

const CA_REMOVE_WGSL: &str = include_str!("../../../shaders/ca_remove.wgsl");

impl rasmcore_pipeline::GpuCapable for CaRemoveParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        if self.red_shift.abs() < 1e-7 && self.blue_shift.abs() < 1e-7 {
            return None;
        }
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.red_shift.to_le_bytes());
        params.extend_from_slice(&self.blue_shift.to_le_bytes());

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(
                &[rasmcore_gpu_shaders::PIXEL_OPS, rasmcore_gpu_shaders::SAMPLE_BILINEAR],
                CA_REMOVE_WGSL,
            ),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: Default::default(),
        }])
    }
}

#[rasmcore_macros::register_filter(
    name = "ca_remove",
    category = "correction",
    reference = "Lightroom lateral CA removal — radial R/B channel realignment"
)]
pub fn ca_remove(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &CaRemoveParams,
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
        return Err(ImageError::UnsupportedFormat("ca_remove requires RGB".into()));
    }

    let r_shift = config.red_shift;
    let b_shift = config.blue_shift;
    if r_shift.abs() < 1e-7 && b_shift.abs() < 1e-7 {
        return Ok(pixels.to_vec());
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    // Shared radial shift logic
    #[inline]
    fn shifted_coord(x: usize, y: usize, cx: f32, cy: f32, shift: f32, w: usize, h: usize) -> (f32, f32) {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let r2 = dx * dx + dy * dy;
        // Shift proportional to r²: corrected = original * (1 + shift * r²/max_r²)
        let max_r2 = cx * cx + cy * cy;
        let scale = 1.0 + shift * r2 / max_r2;
        (cx + dx * scale, cy + dy * scale)
    }

    #[inline]
    fn bilinear_sample_channel(pixels: &[u8], w: usize, h: usize, ch: usize, fx: f32, fy: f32, c: usize) -> u8 {
        let x0 = (fx.floor() as isize).clamp(0, w as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, h as isize - 1) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let xf = fx - fx.floor();
        let yf = fy - fy.floor();

        let p00 = pixels[(y0 * w + x0) * ch + c] as f32;
        let p10 = pixels[(y0 * w + x1) * ch + c] as f32;
        let p01 = pixels[(y1 * w + x0) * ch + c] as f32;
        let p11 = pixels[(y1 * w + x1) * ch + c] as f32;

        let top = p00 * (1.0 - xf) + p10 * xf;
        let bot = p01 * (1.0 - xf) + p11 * xf;
        (top * (1.0 - yf) + bot * yf + 0.5).clamp(0.0, 255.0) as u8
    }

    if is_f32(info.format) {
        let samples: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        #[inline]
        fn bilinear_f32(samples: &[f32], w: usize, h: usize, ch: usize, fx: f32, fy: f32, c: usize) -> f32 {
            let x0 = (fx.floor() as isize).clamp(0, w as isize - 1) as usize;
            let y0 = (fy.floor() as isize).clamp(0, h as isize - 1) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let xf = fx - fx.floor();
            let yf = fy - fy.floor();
            let p00 = samples[(y0 * w + x0) * ch + c];
            let p10 = samples[(y0 * w + x1) * ch + c];
            let p01 = samples[(y1 * w + x0) * ch + c];
            let p11 = samples[(y1 * w + x1) * ch + c];
            let top = p00 * (1.0 - xf) + p10 * xf;
            let bot = p01 * (1.0 - xf) + p11 * xf;
            top * (1.0 - yf) + bot * yf
        }

        let mut out = samples.clone();
        for y in 0..h {
            for x in 0..w {
                let dst = (y * w + x) * ch;
                let (rx, ry) = shifted_coord(x, y, cx, cy, r_shift, w, h);
                out[dst] = bilinear_f32(&samples, w, h, ch, rx, ry, 0);
                // Green unchanged
                let (bx, by) = shifted_coord(x, y, cx, cy, b_shift, w, h);
                out[dst + 2] = bilinear_f32(&samples, w, h, ch, bx, by, 2);
            }
        }
        return Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            ca_remove(r, &mut u, i8, config)
        });
    }

    let mut out = pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let dst = (y * w + x) * ch;
            // Red: sample from shifted position (bilinear)
            let (rx, ry) = shifted_coord(x, y, cx, cy, r_shift, w, h);
            out[dst] = bilinear_sample_channel(pixels, w, h, ch, rx, ry, 0);
            // Green: unchanged (out[dst+1] already correct from copy)
            // Blue: sample from shifted position
            let (bx, by) = shifted_coord(x, y, cx, cy, b_shift, w, h);
            out[dst + 2] = bilinear_sample_channel(pixels, w, h, ch, bx, by, 2);
            // Alpha: unchanged
        }
    }
    Ok(out)
}
