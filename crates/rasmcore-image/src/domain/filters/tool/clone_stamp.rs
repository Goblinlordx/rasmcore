//! Tool: clone_stamp (category: tool)
//!
//! Copy pixels from a source offset position, blended into the image via a
//! grayscale mask. White mask = full clone, black = original untouched.
//! The mask is the "bg" input (brush stroke), the image is the "fg" input.
//! Reference: Photoshop Clone Stamp / GIMP Clone tool.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Clone stamp: copy from source offset, blend via mask.
#[rasmcore_macros::register_compositor(
    name = "clone_stamp",
    category = "tool",
    group = "brush",
    variant = "clone_stamp",
    reference = "Photoshop Clone Stamp — pixel copy from source offset"
)]
pub fn clone_stamp(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    mask_pixels: &[u8],
    mask_info: &ImageInfo,
    offset_x: i32,
    offset_y: i32,
) -> Result<Vec<u8>, ImageError> {
    let w = fg_info.width as usize;
    let h = fg_info.height as usize;
    let ch = match fg_info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "clone_stamp requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mw = mask_info.width as usize;
    let mh = mask_info.height as usize;
    let mask_bpp = mask_pixels.len() / (mw * mh).max(1);

    let mut out = fg_pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            // Sample mask (nearest-neighbor if sizes differ)
            let mx = (x * mw / w).min(mw - 1);
            let my = (y * mh / h).min(mh - 1);
            let mask_val = match mask_bpp {
                1 => mask_pixels[my * mw + mx],
                3 => {
                    let base = (my * mw + mx) * 3;
                    let r = mask_pixels[base] as u32;
                    let g = mask_pixels[base + 1] as u32;
                    let b = mask_pixels[base + 2] as u32;
                    ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
                }
                4 => {
                    let base = (my * mw + mx) * 4;
                    let r = mask_pixels[base] as u32;
                    let g = mask_pixels[base + 1] as u32;
                    let b = mask_pixels[base + 2] as u32;
                    ((r * 2126 + g * 7152 + b * 722 + 5000) / 10000) as u8
                }
                _ => 0,
            };

            if mask_val == 0 {
                continue;
            }

            // Source pixel at offset position
            let sx = x as i32 + offset_x;
            let sy = y as i32 + offset_y;
            if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 {
                continue;
            }
            let src_idx = (sy as usize * w + sx as usize) * ch;
            let dst_idx = (y * w + x) * ch;

            if mask_val == 255 {
                // Full clone
                out[dst_idx..dst_idx + ch].copy_from_slice(&fg_pixels[src_idx..src_idx + ch]);
            } else {
                // Partial blend: lerp(original, source, mask/255)
                let alpha = mask_val as f32 / 255.0;
                let inv = 1.0 - alpha;
                let color_ch = if ch == 4 { 3 } else { ch };
                for c in 0..color_ch {
                    let orig = fg_pixels[dst_idx + c] as f32;
                    let src = fg_pixels[src_idx + c] as f32;
                    out[dst_idx + c] = (orig * inv + src * alpha + 0.5) as u8;
                }
                // Alpha channel preserved from original
            }
        }
    }

    Ok(out)
}
