//! Filter: chromatic_aberration (category: effect)
//!
//! Simulates lateral chromatic aberration (transverse CA) by radially displacing
//! R and B channels away from the image center. Amount increases with distance
//! from center, mimicking real lens CA behavior. Green channel stays in place.
//! Reference: typical lens CA simulation found in Pixlr/Snapseed.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for the chromatic aberration simulation.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ChromaticAberrationParams {
    /// CA strength — radial pixel shift at image corners (0 = none)
    #[param(min = 0.0, max = 20.0, step = 0.5, default = 3.0)]
    pub strength: f32,
}

#[rasmcore_macros::register_filter(
    name = "chromatic_aberration",
    category = "effect",
    reference = "lateral chromatic aberration simulation"
)]
pub fn chromatic_aberration(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ChromaticAberrationParams,
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
            chromatic_aberration(r, &mut u, i8, config)
        });
    }
    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            chromatic_aberration(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "chromatic_aberration requires RGB".into(),
        ));
    }

    let strength = config.strength;
    if strength.abs() < 1e-6 {
        return Ok(pixels.to_vec());
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();

    // Red shifts outward (+), Blue shifts inward (-) relative to center
    // This mimics real lateral CA where short wavelengths (blue) refract more
    let r_scale = strength / max_dist;
    let b_scale = -strength / max_dist;

    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;
            let dist = (dx * dx + dy * dy).sqrt();
            let dst_idx = (y * w + x) * ch;

            // Red channel: sample from radially shifted position
            let r_shift = dist * r_scale;
            let r_sx = (x as f32 + dx / dist.max(1e-6) * r_shift).round() as isize;
            let r_sy = (y as f32 + dy / dist.max(1e-6) * r_shift).round() as isize;
            let r_sx = r_sx.clamp(0, w as isize - 1) as usize;
            let r_sy = r_sy.clamp(0, h as isize - 1) as usize;
            out[dst_idx] = pixels[(r_sy * w + r_sx) * ch];

            // Green channel: no shift
            out[dst_idx + 1] = pixels[dst_idx + 1];

            // Blue channel: sample from radially shifted position (opposite direction)
            let b_shift = dist * b_scale;
            let b_sx = (x as f32 + dx / dist.max(1e-6) * b_shift).round() as isize;
            let b_sy = (y as f32 + dy / dist.max(1e-6) * b_shift).round() as isize;
            let b_sx = b_sx.clamp(0, w as isize - 1) as usize;
            let b_sy = b_sy.clamp(0, h as isize - 1) as usize;
            out[dst_idx + 2] = pixels[(b_sy * w + b_sx) * ch + 2];

            // Copy alpha if present
            if ch == 4 {
                out[dst_idx + 3] = pixels[dst_idx + 3];
            }
        }
    }

    Ok(out)
}
