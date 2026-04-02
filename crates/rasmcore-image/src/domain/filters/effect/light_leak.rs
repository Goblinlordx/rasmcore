//! Filter: light_leak (category: effect)
//!
//! Procedural light leak overlay: generates a warm-toned radial gradient
//! and composites it onto the image using screen blend mode.
//! Reference: Instagram/PicsArt light leak filters.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for the light leak effect.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct LightLeakParams {
    /// Leak intensity (0 = none, 1 = full)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub intensity: f32,
    /// Horizontal position of leak center (0 = left, 1 = right)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.2)]
    pub position_x: f32,
    /// Vertical position of leak center (0 = top, 1 = bottom)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.3)]
    pub position_y: f32,
    /// Leak radius as fraction of image diagonal (0.1-1.0)
    #[param(min = 0.1, max = 1.0, step = 0.05, default = 0.5)]
    pub radius: f32,
    /// Leak color warmth — hue angle: 0=red, 30=orange, 60=yellow
    #[param(min = 0.0, max = 60.0, step = 5.0, default = 25.0)]
    pub warmth: f32,
}

/// Screen blend: 1 - (1-a)(1-b). Lightens the image.
#[inline]
fn screen_blend(base: f32, overlay: f32) -> f32 {
    1.0 - (1.0 - base) * (1.0 - overlay)
}

const LIGHT_LEAK_WGSL: &str = include_str!("../../../shaders/light_leak.wgsl");

impl rasmcore_pipeline::GpuCapable for LightLeakParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        if self.intensity < 1e-6 {
            return None;
        }
        let hue = self.warmth.clamp(0.0, 60.0);
        let (lr, lg, lb) = crate::domain::color_grading::hsl_to_rgb(hue, 1.0, 0.5);

        let mut params = Vec::with_capacity(48);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.intensity.to_le_bytes());
        params.extend_from_slice(&self.position_x.to_le_bytes());
        params.extend_from_slice(&self.position_y.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&lr.to_le_bytes());
        params.extend_from_slice(&lg.to_le_bytes());
        params.extend_from_slice(&lb.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(&[rasmcore_gpu_shaders::PIXEL_OPS], LIGHT_LEAK_WGSL),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}

#[rasmcore_macros::register_filter(
    name = "light_leak",
    category = "effect",
    reference = "procedural warm light leak overlay"
)]
pub fn light_leak(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LightLeakParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            light_leak(r, &mut u, i8, config)
        });
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);

    let intensity = config.intensity;
    if intensity < 1e-6 {
        return Ok(pixels.to_vec());
    }

    // Convert warmth hue to RGB tint color (HSL with S=1, L=0.5)
    let hue = config.warmth.clamp(0.0, 60.0);
    let (lr, lg, lb) = crate::domain::color_grading::hsl_to_rgb(hue, 1.0, 0.5);

    // Leak center and radius in pixel coordinates
    let cx = config.position_x * w as f32;
    let cy = config.position_y * h as f32;
    let diag = ((w * w + h * h) as f32).sqrt();
    let leak_radius = config.radius * diag;
    let inv_radius = 1.0 / leak_radius.max(1.0);

    if is_f32(info.format) {
        let mut out: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for y in 0..h {
            let dy = y as f32 - cy;
            for x in 0..w {
                let dx = x as f32 - cx;
                let dist = (dx * dx + dy * dy).sqrt();
                let t = (1.0 - dist * inv_radius).clamp(0.0, 1.0);
                let leak_strength = t * t * intensity;
                if leak_strength < 1e-6 { continue; }

                let idx = (y * w + x) * ch;
                out[idx] = screen_blend(out[idx], lr * leak_strength);
                out[idx + 1] = screen_blend(out[idx + 1], lg * leak_strength);
                out[idx + 2] = screen_blend(out[idx + 2], lb * leak_strength);
            }
        }

        return Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    let mut out = pixels.to_vec();

    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;
            let dist = (dx * dx + dy * dy).sqrt();

            // Radial falloff: 1.0 at center, 0.0 at radius edge
            let t = (1.0 - dist * inv_radius).clamp(0.0, 1.0);
            // Smooth falloff (quadratic ease)
            let leak_strength = t * t * intensity;

            if leak_strength < 1e-6 {
                continue;
            }

            let idx = (y * w + x) * ch;
            let base_r = pixels[idx] as f32 / 255.0;
            let base_g = pixels[idx + 1] as f32 / 255.0;
            let base_b = pixels[idx + 2] as f32 / 255.0;

            // Screen blend with the leak color, modulated by strength
            let overlay_r = lr * leak_strength;
            let overlay_g = lg * leak_strength;
            let overlay_b = lb * leak_strength;

            out[idx] = (screen_blend(base_r, overlay_r) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 1] =
                (screen_blend(base_g, overlay_g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 2] =
                (screen_blend(base_b, overlay_b) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }

    Ok(out)
}
