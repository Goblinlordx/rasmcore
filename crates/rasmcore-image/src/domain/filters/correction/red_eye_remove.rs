//! Filter: red_eye_remove (category: correction)
//!
//! Red-eye removal: within a specified circular region, detect pixels with
//! red hue and high saturation, then desaturate and darken them to natural eye color.
//! Soft edge at region boundary via distance falloff.
//! Reference: Affinity Photo / Pixlr / Photopea red-eye removal tool.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for red-eye removal.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct RedEyeRemoveParams {
    /// Center X of the eye region (pixels)
    #[param(min = 0, max = 8000, step = 1, default = 0, hint = "rc.pixels")]
    pub center_x: u32,
    /// Center Y of the eye region (pixels)
    #[param(min = 0, max = 8000, step = 1, default = 0, hint = "rc.pixels")]
    pub center_y: u32,
    /// Radius of the eye region (pixels)
    #[param(min = 1, max = 200, step = 1, default = 20, hint = "rc.pixels")]
    pub radius: u32,
    /// Darkening factor (0 = no darkening, 1 = full black)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub darken: f32,
    /// Saturation detection threshold (0-1, pixels above this in red hue range are corrected)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.3)]
    pub threshold: f32,
}

const RED_EYE_WGSL: &str = include_str!("../../../shaders/red_eye_remove.wgsl");

impl rasmcore_pipeline::GpuCapable for RedEyeRemoveParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.center_x.to_le_bytes());
        params.extend_from_slice(&self.center_y.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&self.darken.to_le_bytes());
        params.extend_from_slice(&self.threshold.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(&[rasmcore_gpu_shaders::PIXEL_OPS], RED_EYE_WGSL),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}

#[rasmcore_macros::register_filter(
    name = "red_eye_remove",
    category = "correction",
    reference = "Affinity Photo red-eye removal — localized red desaturation"
)]
pub fn red_eye_remove(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &RedEyeRemoveParams,
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
        return Err(ImageError::UnsupportedFormat("red_eye_remove requires RGB".into()));
    }

    let cx = config.center_x as f32;
    let cy = config.center_y as f32;
    let radius = config.radius as f32;
    let darken = config.darken;
    let threshold = config.threshold;

    if is_f32(info.format) {
        let mut out: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > radius { continue; }

                let idx = (y * w + x) * ch;
                let (r, g, b) = (out[idx], out[idx + 1], out[idx + 2]);
                let (hue, sat, lum) = crate::domain::color_grading::rgb_to_hsl(r, g, b);

                if !is_red_hue(hue) || sat < threshold { continue; }

                let edge_factor = 1.0 - (dist / radius).powi(2);
                let (nr, ng, nb) = crate::domain::color_grading::hsl_to_rgb(hue, 0.0, lum * (1.0 - darken * edge_factor));
                out[idx] = nr;
                out[idx + 1] = ng;
                out[idx + 2] = nb;
            }
        }
        return Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            red_eye_remove(r, &mut u, i8, config)
        });
    }

    let mut out = pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > radius { continue; }

            let idx = (y * w + x) * ch;
            let r = pixels[idx] as f32 / 255.0;
            let g = pixels[idx + 1] as f32 / 255.0;
            let b = pixels[idx + 2] as f32 / 255.0;

            let (hue, sat, lum) = crate::domain::color_grading::rgb_to_hsl(r, g, b);

            // Only correct red-hue, high-saturation pixels
            if !is_red_hue(hue) || sat < threshold { continue; }

            // Soft edge: full correction at center, fading at boundary
            let edge_factor = 1.0 - (dist / radius).powi(2);
            // Desaturate to 0, darken by factor
            let (nr, ng, nb) = crate::domain::color_grading::hsl_to_rgb(
                hue,
                0.0,
                lum * (1.0 - darken * edge_factor),
            );
            out[idx] = (nr * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 1] = (ng * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 2] = (nb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Check if a hue angle is in the "red" range: [340, 360) or [0, 20].
#[inline]
fn is_red_hue(hue: f32) -> bool {
    hue >= 340.0 || hue <= 20.0
}
