//! Content-aware image operations.
//!
//! - Seam carving: content-aware resize via energy-based seam removal
//! - Selective color: per-hue-range channel adjustment

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

// ─── Seam Carving ────────────────────────────────────────────────────────

/// Compute energy map using dual-gradient (Sobel magnitude).
/// Returns energy per pixel as f64 (higher = more important to preserve).
fn compute_energy(pixels: &[u8], w: usize, h: usize, channels: usize) -> Vec<f64> {
    let mut energy = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = if x >= w - 1 { w - 1 } else { x + 1 };
            let y0 = if y == 0 { 0 } else { y - 1 };
            let y1 = if y >= h - 1 { h - 1 } else { y + 1 };

            let mut dx2 = 0.0f64;
            let mut dy2 = 0.0f64;
            for c in 0..channels.min(3) {
                let left = pixels[(y * w + x0) * channels + c] as f64;
                let right = pixels[(y * w + x1) * channels + c] as f64;
                let up = pixels[(y0 * w + x) * channels + c] as f64;
                let down = pixels[(y1 * w + x) * channels + c] as f64;
                dx2 += (right - left) * (right - left);
                dy2 += (down - up) * (down - up);
            }
            energy[y * w + x] = (dx2 + dy2).sqrt();
        }
    }
    energy
}

/// Find minimum-energy vertical seam via dynamic programming.
/// Returns seam as Vec of x-coordinates, one per row (top to bottom).
fn find_vertical_seam(energy: &[f64], w: usize, h: usize) -> Vec<usize> {
    // Build cumulative energy table
    let mut dp = vec![0.0f64; w * h];
    // First row = energy
    dp[..w].copy_from_slice(&energy[..w]);

    // Fill DP table
    for y in 1..h {
        for x in 0..w {
            let above = dp[(y - 1) * w + x];
            let above_left = if x > 0 {
                dp[(y - 1) * w + x - 1]
            } else {
                f64::MAX
            };
            let above_right = if x < w - 1 {
                dp[(y - 1) * w + x + 1]
            } else {
                f64::MAX
            };
            dp[y * w + x] = energy[y * w + x] + above.min(above_left).min(above_right);
        }
    }

    // Backtrack from bottom row
    let mut seam = vec![0usize; h];
    // Find minimum in last row
    let last_row = &dp[(h - 1) * w..h * w];
    seam[h - 1] = last_row
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    for y in (0..h - 1).rev() {
        let x = seam[y + 1];
        let mut best_x = x;
        let mut best_e = dp[y * w + x];
        if x > 0 && dp[y * w + x - 1] < best_e {
            best_e = dp[y * w + x - 1];
            best_x = x - 1;
        }
        if x < w - 1 && dp[y * w + x + 1] < best_e {
            best_x = x + 1;
        }
        seam[y] = best_x;
    }

    seam
}

/// Remove a vertical seam from the image, reducing width by 1.
fn remove_vertical_seam(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    seam: &[usize],
) -> Vec<u8> {
    let new_w = w - 1;
    let mut out = Vec::with_capacity(new_w * h * channels);
    for y in 0..h {
        let skip_x = seam[y];
        for x in 0..w {
            if x != skip_x {
                for c in 0..channels {
                    out.push(pixels[(y * w + x) * channels + c]);
                }
            }
        }
    }
    out
}

/// Content-aware resize (seam carving) — reduce width by removing low-energy seams.
///
/// Uses the Avidan & Shamir (2007) algorithm:
/// 1. Compute energy map (gradient magnitude)
/// 2. Find minimum-energy vertical seam via DP
/// 3. Remove seam
/// 4. Repeat until target width reached
///
/// `target_width` must be less than current width.
pub fn seam_carve_width(
    pixels: &[u8],
    info: &ImageInfo,
    target_width: u32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if target_width >= info.width {
        return Err(ImageError::InvalidParameters(
            "target_width must be less than current width".into(),
        ));
    }
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "seam carving on {other:?}"
            )));
        }
    };

    let mut data = pixels.to_vec();
    let mut w = info.width as usize;
    let h = info.height as usize;

    let seams_to_remove = w - target_width as usize;
    for _ in 0..seams_to_remove {
        let energy = compute_energy(&data, w, h, channels);
        let seam = find_vertical_seam(&energy, w, h);
        data = remove_vertical_seam(&data, w, h, channels, &seam);
        w -= 1;
    }

    let new_info = ImageInfo {
        width: w as u32,
        height: info.height,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((data, new_info))
}

/// Content-aware resize — reduce height by removing horizontal seams.
///
/// Transposes the image, applies vertical seam carving, transposes back.
pub fn seam_carve_height(
    pixels: &[u8],
    info: &ImageInfo,
    target_height: u32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if target_height >= info.height {
        return Err(ImageError::InvalidParameters(
            "target_height must be less than current height".into(),
        ));
    }
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "seam carving on {other:?}"
            )));
        }
    };

    let (w, h) = (info.width as usize, info.height as usize);

    // Transpose
    let mut transposed = vec![0u8; w * h * channels];
    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                transposed[(x * h + y) * channels + c] = pixels[(y * w + x) * channels + c];
            }
        }
    }

    let trans_info = ImageInfo {
        width: info.height,
        height: info.width,
        format: info.format,
        color_space: info.color_space,
    };
    let (carved, carved_info) = seam_carve_width(&transposed, &trans_info, target_height)?;

    // Transpose back
    let new_h = carved_info.width as usize;
    let new_w = carved_info.height as usize;
    let mut result = vec![0u8; new_w * new_h * channels];
    for y in 0..new_h {
        for x in 0..new_w {
            for c in 0..channels {
                result[(y * new_w + x) * channels + c] = carved[(x * new_h + y) * channels + c];
            }
        }
    }

    let new_info = ImageInfo {
        width: new_w as u32,
        height: new_h as u32,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((result, new_info))
}

// ─── Selective Color ─────────────────────────────────────────────────────

/// Hue range for selective color adjustment.
#[derive(Debug, Clone, Copy)]
pub struct HueRange {
    /// Center hue in degrees (0-360).
    pub center: f32,
    /// Width of the hue range in degrees (total spread, not half).
    pub width: f32,
}

/// Selective color adjustment parameters.
#[derive(Debug, Clone)]
pub struct SelectiveColorParams {
    /// Target hue range to adjust.
    pub hue_range: HueRange,
    /// Hue shift in degrees (-180 to 180).
    pub hue_shift: f32,
    /// Saturation multiplier (0 = desaturate, 1 = unchanged, 2 = double).
    pub saturation: f32,
    /// Lightness offset (-1.0 to 1.0).
    pub lightness: f32,
}

/// Apply selective color adjustment — modify only pixels within a specific hue range.
///
/// Converts each pixel to HSL, checks if its hue falls within the target range,
/// and if so, applies hue shift, saturation scale, and lightness offset.
/// Pixels outside the range are unchanged.
pub fn selective_color(
    pixels: &[u8],
    info: &ImageInfo,
    params: &SelectiveColorParams,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "selective color requires Rgb8".into(),
        ));
    }

    let n = (info.width * info.height) as usize;
    let mut out = pixels.to_vec();

    let half_width = params.hue_range.width / 2.0;
    let center = params.hue_range.center;

    for i in 0..n {
        let r = pixels[i * 3] as f32 / 255.0;
        let g = pixels[i * 3 + 1] as f32 / 255.0;
        let b = pixels[i * 3 + 2] as f32 / 255.0;

        let (h, s, l) = rgb_to_hsl(r, g, b);

        // Check if pixel's hue is within target range (wrapping around 360°)
        let hue_diff = ((h - center + 180.0).rem_euclid(360.0)) - 180.0;
        if hue_diff.abs() > half_width {
            continue; // Outside range — skip
        }

        // Smooth falloff at edges of range (cosine taper)
        let blend = if half_width > 0.0 {
            let t = hue_diff.abs() / half_width;
            0.5 * (1.0 + (t * std::f32::consts::PI).cos()) // 1.0 at center, 0.0 at edge
        } else {
            1.0
        };

        let new_h = (h + params.hue_shift * blend).rem_euclid(360.0);
        let new_s = (s * (1.0 + (params.saturation - 1.0) * blend)).clamp(0.0, 1.0);
        let new_l = (l + params.lightness * blend).clamp(0.0, 1.0);

        let (nr, ng, nb) = hsl_to_rgb(new_h, new_s, new_l);
        out[i * 3] = (nr * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (ng * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (nb * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(out)
}

// ─── HSL Helpers ─────────────────────────────────────────────────────────

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < 1e-6 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < 1e-6 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (h * 60.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    #[test]
    fn seam_carve_reduces_width() {
        let info = ImageInfo {
            width: 16,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Gradient image
        let mut pixels = vec![0u8; 16 * 8];
        for y in 0..8 {
            for x in 0..16 {
                pixels[y * 16 + x] = (x * 16) as u8;
            }
        }
        let (result, new_info) = seam_carve_width(&pixels, &info, 12).unwrap();
        assert_eq!(new_info.width, 12);
        assert_eq!(new_info.height, 8);
        assert_eq!(result.len(), 12 * 8);
    }

    #[test]
    fn seam_carve_preserves_high_energy() {
        let info = ImageInfo {
            width: 8,
            height: 4,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Strong vertical edge at x=4
        let mut pixels = vec![0u8; 8 * 4];
        for y in 0..4 {
            for x in 0..8 {
                pixels[y * 8 + x] = if x >= 4 { 255 } else { 0 };
            }
        }
        let (result, new_info) = seam_carve_width(&pixels, &info, 6).unwrap();
        assert_eq!(new_info.width, 6);
        // The edge should be preserved — low-energy columns (uniform 0 or 255) should be removed
        // Check that the result still has a transition from dark to bright
        let has_dark = result.iter().any(|&v| v < 50);
        let has_bright = result.iter().any(|&v| v > 200);
        assert!(
            has_dark && has_bright,
            "seam carving should preserve the edge"
        );
    }

    #[test]
    fn seam_carve_height_reduces() {
        let info = ImageInfo {
            width: 8,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..8 * 16).map(|i| (i % 256) as u8).collect();
        let (result, new_info) = seam_carve_height(&pixels, &info, 12).unwrap();
        assert_eq!(new_info.width, 8);
        assert_eq!(new_info.height, 12);
        assert_eq!(result.len(), 8 * 12);
    }

    #[test]
    fn selective_color_shifts_target_hue() {
        let info = ImageInfo {
            width: 3,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Red, Green, Blue pixels
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let params = SelectiveColorParams {
            hue_range: HueRange {
                center: 0.0,
                width: 60.0,
            }, // Target reds
            hue_shift: 120.0, // Shift red → green
            saturation: 1.0,
            lightness: 0.0,
        };
        let result = selective_color(&pixels, &info, &params).unwrap();
        // Red pixel should have shifted toward green
        assert!(
            result[1] > result[0],
            "shifted red should have more green than red"
        );
        // Green and blue pixels should be unchanged
        assert_eq!(result[3], 0);
        assert_eq!(result[4], 255);
        assert_eq!(result[5], 0);
        assert_eq!(result[6], 0);
        assert_eq!(result[7], 0);
        assert_eq!(result[8], 255);
    }

    #[test]
    fn selective_color_identity_outside_range() {
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![0, 0, 255]; // Pure blue (hue ≈ 240°)
        let params = SelectiveColorParams {
            hue_range: HueRange {
                center: 0.0,
                width: 30.0,
            }, // Target reds only
            hue_shift: 90.0,
            saturation: 2.0,
            lightness: 0.5,
        };
        let result = selective_color(&pixels, &info, &params).unwrap();
        // Blue is outside the red hue range — should be unchanged
        assert_eq!(result, pixels);
    }
}
