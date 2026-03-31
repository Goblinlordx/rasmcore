//! Image quality comparison metrics — PSNR, SSIM, RMSE, MAE, Delta E.
//!
//! All functions take two pixel buffers with matching `ImageInfo` and return
//! a scalar quality metric. Alpha channels in RGBA images are ignored — only
//! color channels are compared.

use super::color_spaces::rgb_to_lab;
use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Extract color-only samples from a pixel buffer, ignoring alpha.
/// Returns (samples, channels_per_pixel).
fn color_samples(pixels: &[u8], format: PixelFormat) -> Result<(Vec<u8>, usize), ImageError> {
    match format {
        PixelFormat::Gray8 => Ok((pixels.to_vec(), 1)),
        PixelFormat::Rgb8 => Ok((pixels.to_vec(), 3)),
        PixelFormat::Rgba8 => {
            let rgb: Vec<u8> = pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            Ok((rgb, 3))
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "metrics require Gray8, RGB8, or RGBA8 — got {other:?}"
        ))),
    }
}

fn validate_pair(
    a: &[u8],
    info_a: &ImageInfo,
    b: &[u8],
    info_b: &ImageInfo,
) -> Result<(Vec<u8>, Vec<u8>, usize), ImageError> {
    if info_a.width != info_b.width || info_a.height != info_b.height {
        return Err(ImageError::InvalidParameters(format!(
            "dimension mismatch: {}x{} vs {}x{}",
            info_a.width, info_a.height, info_b.width, info_b.height
        )));
    }
    let (sa, ca) = color_samples(a, info_a.format)?;
    let (sb, cb) = color_samples(b, info_b.format)?;
    if ca != cb {
        return Err(ImageError::InvalidParameters(format!(
            "channel mismatch: {ca} vs {cb}"
        )));
    }
    if sa.len() != sb.len() {
        return Err(ImageError::InvalidParameters(
            "pixel buffer size mismatch".into(),
        ));
    }
    Ok((sa, sb, ca))
}

/// Mean Absolute Error — average per-sample absolute difference.
pub fn mae(a: &[u8], info_a: &ImageInfo, b: &[u8], info_b: &ImageInfo) -> Result<f64, ImageError> {
    let (sa, sb, _) = validate_pair(a, info_a, b, info_b)?;
    if sa.is_empty() {
        return Ok(0.0);
    }
    let sum: f64 = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    Ok(sum / sa.len() as f64)
}

/// Root Mean Squared Error.
pub fn rmse(a: &[u8], info_a: &ImageInfo, b: &[u8], info_b: &ImageInfo) -> Result<f64, ImageError> {
    let (sa, sb, _) = validate_pair(a, info_a, b, info_b)?;
    if sa.is_empty() {
        return Ok(0.0);
    }
    let mse: f64 = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / sa.len() as f64;
    Ok(mse.sqrt())
}

/// Peak Signal-to-Noise Ratio in dB (8-bit, MAX=255).
///
/// Returns `f64::INFINITY` for identical images.
pub fn psnr(a: &[u8], info_a: &ImageInfo, b: &[u8], info_b: &ImageInfo) -> Result<f64, ImageError> {
    let (sa, sb, _) = validate_pair(a, info_a, b, info_b)?;
    if sa.is_empty() {
        return Ok(f64::INFINITY);
    }
    let mse: f64 = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / sa.len() as f64;
    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (255.0_f64 * 255.0 / mse).log10())
}

/// Structural Similarity Index (Wang et al. 2004).
///
/// Uses 11x11 Gaussian-weighted windows. Returns mean SSIM across all windows.
/// For multi-channel images, SSIM is computed per-channel and averaged.
///
/// Returns a value in \[−1, 1\]; typically in \[0, 1\] for natural images.
/// Identical images return exactly 1.0.
pub fn ssim(a: &[u8], info_a: &ImageInfo, b: &[u8], info_b: &ImageInfo) -> Result<f64, ImageError> {
    let (sa, sb, channels) = validate_pair(a, info_a, b, info_b)?;
    let w = info_a.width as usize;
    let h = info_a.height as usize;

    if w < SSIM_WINDOW || h < SSIM_WINDOW {
        return Err(ImageError::InvalidParameters(format!(
            "image too small for SSIM ({w}x{h}, need at least {SSIM_WINDOW}x{SSIM_WINDOW})"
        )));
    }

    let mut total_ssim = 0.0;
    for ch in 0..channels {
        let a_ch: Vec<f64> = sa
            .iter()
            .skip(ch)
            .step_by(channels)
            .map(|&v| v as f64)
            .collect();
        let b_ch: Vec<f64> = sb
            .iter()
            .skip(ch)
            .step_by(channels)
            .map(|&v| v as f64)
            .collect();
        total_ssim += ssim_channel(&a_ch, &b_ch, w, h);
    }
    Ok(total_ssim / channels as f64)
}

const SSIM_WINDOW: usize = 11;
const SSIM_SIGMA: f64 = 1.5;
const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0); // 6.5025
const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0); // 58.5225

/// Precomputed 11x11 Gaussian kernel (sigma=1.5), normalized to sum=1.
fn gaussian_kernel() -> Vec<f64> {
    let half = SSIM_WINDOW as i32 / 2;
    let mut kernel = vec![0.0f64; SSIM_WINDOW * SSIM_WINDOW];
    let mut sum = 0.0;
    for y in 0..SSIM_WINDOW {
        for x in 0..SSIM_WINDOW {
            let dx = x as i32 - half;
            let dy = y as i32 - half;
            let g = (-(dx * dx + dy * dy) as f64 / (2.0 * SSIM_SIGMA * SSIM_SIGMA)).exp();
            kernel[y * SSIM_WINDOW + x] = g;
            sum += g;
        }
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// SSIM for a single channel (f64 samples, row-major w×h).
fn ssim_channel(a: &[f64], b: &[f64], w: usize, h: usize) -> f64 {
    let kernel = gaussian_kernel();
    let mut ssim_sum = 0.0;
    let mut count = 0u64;

    for wy in 0..=(h - SSIM_WINDOW) {
        for wx in 0..=(w - SSIM_WINDOW) {
            let mut mu_a = 0.0;
            let mut mu_b = 0.0;
            let mut sig_a2 = 0.0;
            let mut sig_b2 = 0.0;
            let mut sig_ab = 0.0;

            for ky in 0..SSIM_WINDOW {
                for kx in 0..SSIM_WINDOW {
                    let g = kernel[ky * SSIM_WINDOW + kx];
                    let va = a[(wy + ky) * w + (wx + kx)];
                    let vb = b[(wy + ky) * w + (wx + kx)];
                    mu_a += g * va;
                    mu_b += g * vb;
                }
            }

            for ky in 0..SSIM_WINDOW {
                for kx in 0..SSIM_WINDOW {
                    let g = kernel[ky * SSIM_WINDOW + kx];
                    let va = a[(wy + ky) * w + (wx + kx)];
                    let vb = b[(wy + ky) * w + (wx + kx)];
                    let da = va - mu_a;
                    let db = vb - mu_b;
                    sig_a2 += g * da * da;
                    sig_b2 += g * db * db;
                    sig_ab += g * da * db;
                }
            }

            let num = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
            let den = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_a2 + sig_b2 + C2);
            ssim_sum += num / den;
            count += 1;
        }
    }

    if count == 0 {
        1.0
    } else {
        ssim_sum / count as f64
    }
}

/// Mean CIE76 Delta E between two images.
///
/// Converts both images to CIE Lab via sRGB→XYZ→Lab, then computes per-pixel
/// Euclidean distance in Lab space. Returns the mean across all pixels.
///
/// Both images must be RGB8 or RGBA8 (alpha ignored).
pub fn delta_e_cie76(
    a: &[u8],
    info_a: &ImageInfo,
    b: &[u8],
    info_b: &ImageInfo,
) -> Result<f64, ImageError> {
    let (sa, sb, channels) = validate_pair(a, info_a, b, info_b)?;
    if channels != 3 {
        return Err(ImageError::InvalidParameters(
            "Delta E requires RGB or RGBA images".into(),
        ));
    }
    let n = sa.len() / 3;
    if n == 0 {
        return Ok(0.0);
    }
    let mut sum = 0.0;
    for i in 0..n {
        let (l1, a1, b1) = rgb_to_lab(
            sa[i * 3] as f64 / 255.0,
            sa[i * 3 + 1] as f64 / 255.0,
            sa[i * 3 + 2] as f64 / 255.0,
        );
        let (l2, a2, b2) = rgb_to_lab(
            sb[i * 3] as f64 / 255.0,
            sb[i * 3 + 1] as f64 / 255.0,
            sb[i * 3 + 2] as f64 / 255.0,
        );
        let dl = l1 - l2;
        let da = a1 - a2;
        let db = b1 - b2;
        sum += (dl * dl + da * da + db * db).sqrt();
    }
    Ok(sum / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn info_rgb8(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn info_rgba8(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn info_gray8(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    // ─── Identical images ──────────────────────────────────────────────

    #[test]
    fn identical_rgb_psnr_infinity() {
        let px = vec![128u8; 48 * 48 * 3];
        let info = info_rgb8(48, 48);
        assert_eq!(psnr(&px, &info, &px, &info).unwrap(), f64::INFINITY);
    }

    #[test]
    fn identical_rgb_ssim_one() {
        let px: Vec<u8> = (0..48 * 48 * 3).map(|i| (i % 256) as u8).collect();
        let info = info_rgb8(48, 48);
        let s = ssim(&px, &info, &px, &info).unwrap();
        assert!((s - 1.0).abs() < 1e-10, "SSIM of identical images: {s}");
    }

    #[test]
    fn identical_rgb_rmse_zero() {
        let px = vec![100u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        assert_eq!(rmse(&px, &info, &px, &info).unwrap(), 0.0);
    }

    #[test]
    fn identical_rgb_mae_zero() {
        let px = vec![100u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        assert_eq!(mae(&px, &info, &px, &info).unwrap(), 0.0);
    }

    #[test]
    fn identical_rgb_delta_e_zero() {
        let px = vec![100u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        assert_eq!(delta_e_cie76(&px, &info, &px, &info).unwrap(), 0.0);
    }

    // ─── Known difference ──────────────────────────────────────────────

    #[test]
    fn mae_known_difference() {
        // All pixels differ by exactly 10
        let a = vec![100u8; 16 * 16 * 3];
        let b = vec![110u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        let m = mae(&a, &info, &b, &info).unwrap();
        assert!((m - 10.0).abs() < 1e-10, "MAE should be 10.0, got {m}");
    }

    #[test]
    fn rmse_known_difference() {
        let a = vec![100u8; 16 * 16 * 3];
        let b = vec![110u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        let r = rmse(&a, &info, &b, &info).unwrap();
        assert!((r - 10.0).abs() < 1e-10, "RMSE should be 10.0, got {r}");
    }

    #[test]
    fn psnr_known_difference() {
        // MSE = 100, PSNR = 10*log10(255^2/100) = 10*log10(650.25) ≈ 28.13
        let a = vec![100u8; 16 * 16 * 3];
        let b = vec![110u8; 16 * 16 * 3];
        let info = info_rgb8(16, 16);
        let p = psnr(&a, &info, &b, &info).unwrap();
        let expected = 10.0 * (255.0_f64 * 255.0 / 100.0).log10();
        assert!(
            (p - expected).abs() < 0.01,
            "PSNR should be {expected:.2}, got {p:.2}"
        );
    }

    // ─── SSIM properties ───────────────────────────────────────────────

    #[test]
    fn ssim_symmetry() {
        let a: Vec<u8> = (0..48 * 48 * 3).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..48 * 48 * 3).map(|i| ((i + 50) % 256) as u8).collect();
        let info = info_rgb8(48, 48);
        let s1 = ssim(&a, &info, &b, &info).unwrap();
        let s2 = ssim(&b, &info, &a, &info).unwrap();
        assert!(
            (s1 - s2).abs() < 1e-10,
            "SSIM should be symmetric: {s1} vs {s2}"
        );
    }

    #[test]
    fn ssim_bounds() {
        let a: Vec<u8> = (0..48 * 48 * 3).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..48 * 48 * 3)
            .map(|i| ((i * 7 + 100) % 256) as u8)
            .collect();
        let info = info_rgb8(48, 48);
        let s = ssim(&a, &info, &b, &info).unwrap();
        assert!(s >= -1.0 && s <= 1.0, "SSIM out of bounds: {s}");
    }

    // ─── RGBA (alpha ignored) ──────────────────────────────────────────

    #[test]
    fn rgba_alpha_ignored() {
        // Same color channels, different alpha — metrics should treat as identical
        let mut a = vec![0u8; 16 * 16 * 4];
        let mut b = vec![0u8; 16 * 16 * 4];
        for i in 0..16 * 16 {
            a[i * 4] = 100;
            a[i * 4 + 1] = 150;
            a[i * 4 + 2] = 200;
            a[i * 4 + 3] = 255; // opaque
            b[i * 4] = 100;
            b[i * 4 + 1] = 150;
            b[i * 4 + 2] = 200;
            b[i * 4 + 3] = 0; // transparent
        }
        let info = info_rgba8(16, 16);
        assert_eq!(mae(&a, &info, &b, &info).unwrap(), 0.0);
        assert_eq!(psnr(&a, &info, &b, &info).unwrap(), f64::INFINITY);
    }

    // ─── Gray8 support ─────────────────────────────────────────────────

    #[test]
    fn gray8_mae() {
        let a = vec![100u8; 16 * 16];
        let b = vec![110u8; 16 * 16];
        let info = info_gray8(16, 16);
        let m = mae(&a, &info, &b, &info).unwrap();
        assert!((m - 10.0).abs() < 1e-10);
    }

    // ─── Dimension mismatch ────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_errors() {
        let a = vec![0u8; 8 * 8 * 3];
        let b = vec![0u8; 16 * 16 * 3];
        let info_a = info_rgb8(8, 8);
        let info_b = info_rgb8(16, 16);
        assert!(mae(&a, &info_a, &b, &info_b).is_err());
    }

    // ─── Delta E sanity ────────────────────────────────────────────────

    #[test]
    fn delta_e_nonzero_for_different_colors() {
        let a = vec![255u8, 0, 0]; // red
        let b = vec![0u8, 255, 0]; // green
        let info = info_rgb8(1, 1);
        let de = delta_e_cie76(&a, &info, &b, &info).unwrap();
        assert!(
            de > 50.0,
            "red vs green should have large Delta E, got {de}"
        );
    }

    #[test]
    fn delta_e_small_for_similar_colors() {
        let a = vec![100u8, 100, 100];
        let b = vec![101u8, 100, 100];
        let info = info_rgb8(1, 1);
        let de = delta_e_cie76(&a, &info, &b, &info).unwrap();
        assert!(
            de < 2.0,
            "near-identical colors should have small Delta E, got {de}"
        );
    }
}
