//! Tool: sponge (category: tool)
//!
//! Localized saturation adjustment: increase or decrease saturation within
//! the masked region, scaled by mask intensity.
//! Reference: Photoshop Sponge tool (saturate/desaturate modes).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for sponge tool.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SpongeParams {
    /// Mode: 0 = saturate, 1 = desaturate
    #[param(min = 0, max = 1, step = 1, default = 0)]
    pub mode: u32,
    /// Adjustment strength (0 = no effect, 1 = full)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
}

#[rasmcore_macros::register_compositor(
    name = "sponge",
    category = "tool",
    group = "brush",
    variant = "sponge",
    reference = "Photoshop Sponge tool — localized saturation adjustment"
)]
pub fn sponge(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    mask_pixels: &[u8],
    mask_info: &ImageInfo,
    mode: u32,
    strength: f32,
) -> Result<Vec<u8>, ImageError> {
    let w = fg_info.width as usize;
    let h = fg_info.height as usize;
    let ch = match fg_info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "sponge requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mw = mask_info.width as usize;
    let mh = mask_info.height as usize;
    let mask_bpp = mask_pixels.len() / (mw * mh).max(1);

    let is_saturate = mode == 0;
    let mut out = fg_pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            let mx = (x * mw / w).min(mw - 1);
            let my = (y * mh / h).min(mh - 1);
            let mask_val = sample_mask_gray(mask_pixels, mw, mx, my, mask_bpp);
            if mask_val == 0 { continue; }

            let blend = strength * mask_val as f32 / 255.0;
            if blend < 1e-6 { continue; }

            let idx = (y * w + x) * ch;
            let r = fg_pixels[idx] as f32 / 255.0;
            let g = fg_pixels[idx + 1] as f32 / 255.0;
            let b = fg_pixels[idx + 2] as f32 / 255.0;

            // Convert to HSL
            let (hue, sat, lum) = crate::domain::color_grading::rgb_to_hsl(r, g, b);

            // Adjust saturation
            let new_sat = if is_saturate {
                (sat + blend * (1.0 - sat)).clamp(0.0, 1.0)
            } else {
                (sat * (1.0 - blend)).clamp(0.0, 1.0)
            };

            let (nr, ng, nb) = crate::domain::color_grading::hsl_to_rgb(hue, new_sat, lum);
            out[idx] = (nr * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 1] = (ng * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[idx + 2] = (nb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            // Alpha preserved from original (if ch == 4)
        }
    }

    Ok(out)
}

fn sample_mask_gray(mask: &[u8], mw: usize, mx: usize, my: usize, bpp: usize) -> u8 {
    match bpp {
        1 => mask[my * mw + mx],
        3 => {
            let b = (my * mw + mx) * 3;
            let r = mask[b] as u32;
            let g = mask[b + 1] as u32;
            let bl = mask[b + 2] as u32;
            ((r * 2126 + g * 7152 + bl * 722 + 5000) / 10000) as u8
        }
        4 => {
            let b = (my * mw + mx) * 4;
            let r = mask[b] as u32;
            let g = mask[b + 1] as u32;
            let bl = mask[b + 2] as u32;
            ((r * 2126 + g * 7152 + bl * 722 + 5000) / 10000) as u8
        }
        _ => 0,
    }
}
