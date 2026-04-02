//! Tool: smudge (category: tool)
//!
//! Directional pixel smearing: for each masked pixel, blend it toward its
//! neighbor in the stroke direction. Strength controls blend ratio.
//! Reference: Photoshop Smudge / Krita smudge brush (simplified — no history).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for smudge tool.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SmudgeParams {
    /// Stroke direction X component (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub direction_x: f32,
    /// Stroke direction Y component (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub direction_y: f32,
    /// Smudge strength (0 = no effect, 1 = full smear)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
}

#[rasmcore_macros::register_compositor(
    name = "smudge",
    category = "tool",
    group = "brush",
    variant = "smudge",
    reference = "Photoshop Smudge tool — directional pixel averaging"
)]
pub fn smudge(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    mask_pixels: &[u8],
    mask_info: &ImageInfo,
    direction_x: f32,
    direction_y: f32,
    strength: f32,
) -> Result<Vec<u8>, ImageError> {
    let w = fg_info.width as usize;
    let h = fg_info.height as usize;
    let ch = match fg_info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "smudge requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mw = mask_info.width as usize;
    let mh = mask_info.height as usize;
    let mask_bpp = mask_pixels.len() / (mw * mh).max(1);

    // Normalize direction vector
    let len = (direction_x * direction_x + direction_y * direction_y).sqrt().max(1e-6);
    let dx = direction_x / len;
    let dy = direction_y / len;

    let mut out = fg_pixels.to_vec();
    let color_ch = if ch == 4 { 3 } else { ch };

    for y in 0..h {
        for x in 0..w {
            // Sample mask
            let mx = (x * mw / w).min(mw - 1);
            let my = (y * mh / h).min(mh - 1);
            let mask_val = sample_mask_gray(mask_pixels, mw, mx, my, mask_bpp);
            if mask_val == 0 { continue; }

            let blend = strength * mask_val as f32 / 255.0;
            if blend < 1e-6 { continue; }

            // Sample neighbor in stroke direction
            let nx = (x as f32 - dx).round() as isize;
            let ny = (y as f32 - dy).round() as isize;
            if nx < 0 || nx >= w as isize || ny < 0 || ny >= h as isize { continue; }

            let src_idx = (ny as usize * w + nx as usize) * ch;
            let dst_idx = (y * w + x) * ch;

            for c in 0..color_ch {
                let orig = fg_pixels[dst_idx + c] as f32;
                let neighbor = fg_pixels[src_idx + c] as f32;
                out[dst_idx + c] = (orig * (1.0 - blend) + neighbor * blend + 0.5) as u8;
            }
        }
    }

    Ok(out)
}

/// Extract a single grayscale value from a mask pixel at (mx, my).
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
