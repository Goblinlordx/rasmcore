//! Tool: healing_brush (category: tool)
//!
//! Seamless clone that matches color/luminance at the boundary.
//! Approach: clone from source offset, then gradient-domain blend at the
//! mask boundary so the cloned region smoothly matches surrounding context.
//! This is a simplified Poisson-like blend (GIMP healing approach).
//! Reference: GIMP Heal tool / OpenCV cv::seamlessClone (simplified).

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Healing brush: clone + boundary color matching.
#[rasmcore_macros::register_compositor(
    name = "healing_brush",
    category = "tool",
    group = "brush",
    variant = "healing",
    reference = "GIMP heal / OpenCV seamlessClone (boundary-matching clone)"
)]
pub fn healing_brush(
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
                "healing_brush requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mw = mask_info.width as usize;
    let mh = mask_info.height as usize;
    let mask_bpp = mask_pixels.len() / (mw * mh).max(1);

    // Build mask buffer (normalized to 0-255, resampled to image size)
    let mask = build_mask_buffer(mask_pixels, mw, mh, mask_bpp, w, h);

    // Step 1: clone from source offset
    let mut cloned = fg_pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let m = mask[y * w + x];
            if m == 0 { continue; }

            let sx = x as i32 + offset_x;
            let sy = y as i32 + offset_y;
            if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 { continue; }

            let src_idx = (sy as usize * w + sx as usize) * ch;
            let dst_idx = (y * w + x) * ch;
            let color_ch = if ch == 4 { 3 } else { ch };
            for c in 0..color_ch {
                cloned[dst_idx + c] = fg_pixels[src_idx + c];
            }
        }
    }

    // Step 2: compute boundary color correction
    // For each masked pixel, compute the difference between original and cloned
    // at the boundary, then interpolate that correction inward using distance
    // from the boundary (simplified Poisson).
    let boundary = compute_boundary_mask(&mask, w, h);
    let correction = compute_boundary_correction(fg_pixels, &cloned, &boundary, &mask, w, h, ch);

    // Step 3: apply correction inside the mask
    let mut out = fg_pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            let m = mask[y * w + x];
            if m == 0 { continue; }

            let idx = (y * w + x) * ch;
            let alpha = m as f32 / 255.0;
            let color_ch = if ch == 4 { 3 } else { ch };
            for c in 0..color_ch {
                // Cloned value + color correction, blended by mask strength
                let corrected = (cloned[idx + c] as f32 + correction[(y * w + x) * 3 + c]).clamp(0.0, 255.0);
                out[idx + c] = (fg_pixels[idx + c] as f32 * (1.0 - alpha) + corrected * alpha + 0.5) as u8;
            }
        }
    }

    Ok(out)
}

/// Extract grayscale mask, resample to target size via nearest-neighbor.
fn build_mask_buffer(
    mask_pixels: &[u8], mw: usize, mh: usize, mask_bpp: usize,
    w: usize, h: usize,
) -> Vec<u8> {
    let mut mask = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mx = (x * mw / w).min(mw - 1);
            let my = (y * mh / h).min(mh - 1);
            mask[y * w + x] = match mask_bpp {
                1 => mask_pixels[my * mw + mx],
                3 => {
                    let b = (my * mw + mx) * 3;
                    let r = mask_pixels[b] as u32;
                    let g = mask_pixels[b + 1] as u32;
                    let bl = mask_pixels[b + 2] as u32;
                    ((r * 2126 + g * 7152 + bl * 722 + 5000) / 10000) as u8
                }
                4 => {
                    let b = (my * mw + mx) * 4;
                    let r = mask_pixels[b] as u32;
                    let g = mask_pixels[b + 1] as u32;
                    let bl = mask_pixels[b + 2] as u32;
                    ((r * 2126 + g * 7152 + bl * 722 + 5000) / 10000) as u8
                }
                _ => 0,
            };
        }
    }
    mask
}

/// Identify boundary pixels (mask > 0 with at least one mask=0 4-neighbor).
fn compute_boundary_mask(mask: &[u8], w: usize, h: usize) -> Vec<bool> {
    let mut boundary = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 { continue; }
            let is_boundary = (x == 0 || mask[y * w + x - 1] == 0)
                || (x + 1 >= w || mask[y * w + x + 1] == 0)
                || (y == 0 || mask[(y - 1) * w + x] == 0)
                || (y + 1 >= h || mask[(y + 1) * w + x] == 0);
            boundary[y * w + x] = is_boundary;
        }
    }
    boundary
}

/// Compute per-pixel color correction by averaging the boundary original-vs-cloned
/// difference and interpolating inward by distance from boundary.
fn compute_boundary_correction(
    original: &[u8], cloned: &[u8], boundary: &[bool], mask: &[u8],
    w: usize, h: usize, ch: usize,
) -> Vec<f32> {
    let color_ch = if ch == 4 { 3 } else { ch };

    // Compute average color difference at boundary pixels
    let mut sum_diff = [0.0f64; 3];
    let mut count = 0u64;
    for y in 0..h {
        for x in 0..w {
            if !boundary[y * w + x] { continue; }
            let idx = (y * w + x) * ch;
            for c in 0..color_ch {
                sum_diff[c] += original[idx + c] as f64 - cloned[idx + c] as f64;
            }
            count += 1;
        }
    }

    let avg_diff: [f32; 3] = if count > 0 {
        [
            (sum_diff[0] / count as f64) as f32,
            (sum_diff[1] / count as f64) as f32,
            (sum_diff[2] / count as f64) as f32,
        ]
    } else {
        [0.0; 3]
    };

    // For simplicity, apply uniform correction (full Poisson would solve a
    // Laplacian but that's much more complex). The uniform correction matches
    // the GIMP simple heal approach.
    let mut correction = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 { continue; }
            let idx = (y * w + x) * 3;
            correction[idx] = avg_diff[0];
            correction[idx + 1] = avg_diff[1];
            correction[idx + 2] = avg_diff[2];
        }
    }

    correction
}
