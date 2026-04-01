//! Blur helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Young/van Vliet IIR gaussian blur on a single-channel f32 buffer.
///
/// Exact port of GEGL's `gegl:gaussian-blur` IIR implementation from gblur-1d.c.
/// Uses the recursive (IIR) algorithm from:
///   I.T. Young, L.J. van Vliet, "Recursive implementation of the Gaussian
///   filter", Signal Processing 44 (1995) 139-151.
///
/// Properties:
/// - O(1) per pixel regardless of sigma (vs O(sigma) for FIR)
/// - Infinite support (exact gaussian frequency response)
/// - Separable: applied as H then V, each forward+backward
/// - Right boundary correction via 3x3 matrix (matches GEGL exactly)
///
/// Used by shadow_highlight, retinex, clarity, frequency separation.
pub fn blur_1ch_f32(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return data.to_vec();
    }

    // For very small sigma (< 0.5), IIR is unstable — fall back to FIR
    if sigma < 0.5 {
        return blur_1ch_f32_fir(data, w, h, sigma);
    }

    let (b, m) = yvv_find_constants(sigma);

    // Horizontal pass
    let mut out = data.to_vec();
    for y in 0..h {
        let off = y * w;
        yvv_blur_1d(&mut out[off..off + w], &b, &m);
    }

    // Vertical pass (extract column, blur, write back)
    let mut col = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = out[y * w + x];
        }
        yvv_blur_1d(&mut col, &b, &m);
        for y in 0..h {
            out[y * w + x] = col[y];
        }
    }

    out
}

/// FIR fallback for very small sigma where IIR is unstable.
pub fn blur_1ch_f32_fir(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let ksize = ((sigma * 3.0).ceil() as usize) * 2 + 1;
    let ksize = ksize.max(3);
    let kernel = gaussian_kernel_1d(ksize, sigma);
    let half = ksize / 2;

    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sx =
                    (x as isize + k as isize - half as isize).clamp(0, w as isize - 1) as usize;
                sum += data[y * w + sx] * kval;
            }
            tmp[y * w + x] = sum;
        }
    }

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sy =
                    (y as isize + k as isize - half as isize).clamp(0, h as isize - 1) as usize;
                sum += tmp[sy * w + x] * kval;
            }
            out[y * w + x] = sum;
        }
    }
    out
}

pub fn blur_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &BlurParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;

    if radius < 0.0 {
        return Err(ImageError::InvalidParameters(
            "blur radius must be >= 0".into(),
        ));
    }
    validate_format(info.format)?;

    if radius == 0.0 {
        return Ok(pixels.to_vec());
    }

    // 16-bit: delegate to 8-bit path via process_via_8bit (convolve only supports u8)
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| blur_impl(p8, i8, config));
    }

    // In the old libblur API, radius was effectively sigma
    let sigma = radius;

    // Large sigma: use box blur approximation (O(1) per pixel, 3 passes)
    if sigma >= 20.0 {
        return gaussian_blur_box_approx(pixels, info, sigma);
    }

    // Build separable Gaussian kernel
    let ksize = {
        let k = (sigma * 6.0 + 1.0).round() as usize;
        if k % 2 == 0 { k + 1 } else { k }
    };
    let ksize = ksize.max(3);
    let k1d = gaussian_kernel_1d(ksize, sigma);

    // Build 2D kernel as outer product (convolve auto-detects separable)
    let mut kernel_2d = vec![0.0f32; ksize * ksize];
    for y in 0..ksize {
        for x in 0..ksize {
            kernel_2d[y * ksize + x] = k1d[y] * k1d[x];
        }
    }

    // Use our own convolve with auto-separable detection + WASM SIMD
    let full_rect = Rect::new(0, 0, info.width, info.height);
    let mut u = |_: Rect| Ok(pixels.to_vec());
    convolve(
        full_rect,
        &mut u,
        info,
        &kernel_2d,
        &ConvolveParams {
            kw: ksize as u32,
            kh: ksize as u32,
            divisor: 1.0,
        },
    )
}

/// Single-pass box blur on an f32 buffer (single channel, row-major).
///
/// Uses a sliding sum for O(1) per pixel regardless of radius.
/// Border handling: extend edge pixels (clamp).
pub fn box_blur_pass_f32(data: &mut [f32], w: usize, h: usize, radius: usize) {
    if radius == 0 {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];

    // Horizontal pass
    let diameter = 2 * radius + 1;
    let inv_d = 1.0 / diameter as f32;
    for y in 0..h {
        let row = y * w;
        // Initialize running sum for first output pixel
        let mut sum = 0.0f32;
        for kx in 0..diameter {
            let sx = (kx as isize - radius as isize).clamp(0, (w - 1) as isize) as usize;
            sum += data[row + sx];
        }
        tmp[row] = sum * inv_d;

        for x in 1..w {
            // Add new right pixel, subtract old left pixel
            let add_x = (x + radius).min(w - 1);
            let sub_x = (x as isize - radius as isize - 1).max(0) as usize;
            sum += data[row + add_x] - data[row + sub_x];
            tmp[row + x] = sum * inv_d;
        }
    }

    // Vertical pass
    for x in 0..w {
        let mut sum = 0.0f32;
        for ky in 0..diameter {
            let sy = (ky as isize - radius as isize).clamp(0, (h - 1) as isize) as usize;
            sum += tmp[sy * w + x];
        }
        data[x] = sum * inv_d;

        for y in 1..h {
            let add_y = (y + radius).min(h - 1);
            let sub_y = (y as isize - radius as isize - 1).max(0) as usize;
            sum += tmp[add_y * w + x] - tmp[sub_y * w + x];
            data[y * w + x] = sum * inv_d;
        }
    }
}

/// Compute box blur radii for a 3-pass stackable approximation of Gaussian blur.
///
/// Three sequential box blur passes approximate a Gaussian via the central limit
/// theorem. Returns three radii. Based on the algorithm from:
/// "Fast Almost-Gaussian Filtering" (Kovesi, 2010).
pub fn box_blur_radii_for_gaussian(sigma: f32) -> [usize; 3] {
    // ideal box filter width: w = sqrt(12*sigma^2/n + 1), n = 3 passes
    let w_ideal = ((12.0 * sigma * sigma / 3.0) + 1.0).sqrt();
    let wl = (w_ideal.floor() as usize) | 1; // round down to odd
    let wu = wl + 2; // next odd

    // how many passes use wl vs wu to best approximate the target variance
    let m = ((12.0 * sigma * sigma - (3 * wl * wl + 12 * wl + 9) as f32) / (4 * (wl + 2)) as f32)
        .round() as usize;

    let mut radii = [0usize; 3];
    for (i, r) in radii.iter_mut().enumerate() {
        *r = if i < m { wu / 2 } else { wl / 2 };
    }
    radii
}

/// 3-pass stackable box blur approximating a Gaussian with the given sigma.
///
/// Operates on an f32 buffer (single channel). O(1) per pixel regardless of sigma.
/// 1D IIR gaussian blur (forward + backward with edge-replicated padding).
///
/// Uses f64 intermediates for precision (matching GEGL's gdouble).
/// Pads the buffer with replicated edge values to handle boundaries
/// correctly without the complex matrix correction.
/// Compute Young/van Vliet IIR coefficients and boundary correction matrix.
/// Exact port of GEGL's `iir_young_find_constants`.
///
/// Returns (b[4], m[3][3]) where b[0] is scale, b[1-3] are recursive coefficients,
/// and m is the right-boundary correction matrix.
/// Gaussian blur approximation for u8 images using 3-pass stackable box blur.
///
/// For large sigma (>= 20), this is dramatically faster than the exact separable
/// Gaussian: O(6*N) vs O(2*K*N) where K can be 481 for sigma=80.
/// Quality: PSNR >= 35dB compared to true Gaussian for sigma >= 20.
pub fn gaussian_blur_box_approx(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    // Process each channel independently in f32 domain
    let mut result = pixels.to_vec();
    let mut channel_buf = vec![0.0f32; w * h];

    for c in 0..channels {
        // Extract channel to f32 buffer
        for i in 0..(w * h) {
            channel_buf[i] = pixels[i * channels + c] as f32;
        }

        // Apply 3-pass box blur
        stackable_box_blur_f32(&mut channel_buf, w, h, sigma);

        // Write back to result
        for i in 0..(w * h) {
            result[i * channels + c] = channel_buf[i].round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// Separable Gaussian blur operating entirely in f64 precision.
/// Matches IM's KernelRank=3 Blur kernel construction (Photoshop-derived
/// 3x oversampled Gaussian for better normalization) and edge-clamp border.
pub fn gaussian_blur_f32(
    pixels: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    krad: usize,
    sigma: f64,
) -> Vec<f32> {
    // Build 1D Gaussian kernel using IM's KernelRank=3 technique:
    // Generate a Gaussian 3x wider, accumulate 3 samples per output bin.
    const KERNEL_RANK: usize = 3;
    let ksize = 2 * krad + 1;
    let mut kernel = vec![0.0f64; ksize];
    let sigma_scaled = sigma * KERNEL_RANK as f64;
    let alpha = 1.0 / (2.0 * sigma_scaled * sigma_scaled);
    let beta = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma_scaled);
    let v = (ksize * KERNEL_RANK - 1) / 2;
    for u_i in 0..=(2 * v) {
        let u = u_i as i64 - v as i64;
        let idx = u_i / KERNEL_RANK;
        kernel[idx] += (-(u * u) as f64 * alpha).exp() * beta;
    }
    let ksum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= ksum;
    }

    let n = w * h * ch;
    // Convert input to Q16-HDRI scale (0-65535) matching IM's ScaleCharToQuantum
    // IM stores pixels as float (Quantum) in Q16-HDRI mode: value * 257.0
    const Q16_SCALE: f32 = 257.0;
    let input: Vec<f32> = pixels.iter().map(|&v| v as f32 * Q16_SCALE).collect();

    // Horizontal pass (edge-clamp border)
    let mut tmp = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut sum = 0.0f64;
                for (ki, &kval) in kernel.iter().enumerate().take(ksize) {
                    let sx = (x as i32 + ki as i32 - krad as i32).clamp(0, w as i32 - 1) as usize;
                    sum += kval * input[(y * w + sx) * ch + c] as f64;
                }
                tmp[(y * w + x) * ch + c] = sum as f32;
            }
        }
    }

    // Vertical pass (edge-clamp border)
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut sum = 0.0f64;
                for (ki, &kval) in kernel.iter().enumerate().take(ksize) {
                    let sy = (y as i32 + ki as i32 - krad as i32).clamp(0, h as i32 - 1) as usize;
                    sum += kval * tmp[(sy * w + x) * ch + c] as f64;
                }
                out[(y * w + x) * ch + c] = sum as f32;
            }
        }
    }
    out
}

/// Separable Gaussian blur (f64 precision) for adaptive threshold Gaussian mode.
#[allow(clippy::needless_range_loop)]
pub fn gaussian_blur_f64(pixels: &[u8], w: usize, h: usize, ksize: usize, sigma: f64) -> Vec<f64> {
    let r = ksize / 2;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f64; ksize];
    let mut sum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - r as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in &mut kernel {
        *k /= sum;
    }

    let hs = h as isize;
    let ws = w as isize;

    // Horizontal pass
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f64;
            for k in 0..ksize {
                let sx = x as isize + k as isize - r as isize;
                let sx = reflect101(sx, ws) as usize;
                val += pixels[y * w + sx] as f64 * kernel[k];
            }
            tmp[y * w + x] = val;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f64;
            for k in 0..ksize {
                let sy = y as isize + k as isize - r as isize;
                let sy = reflect101(sy, hs) as usize;
                val += tmp[sy * w + x] * kernel[k];
            }
            result[y * w + x] = val;
        }
    }

    result
}

#[allow(clippy::needless_range_loop)]
/// Separable 1D Gaussian blur on a f64 single-channel buffer.
///
/// Uses zero-padding outside image bounds (matching ImageMagick's vignette
/// canvas behaviour). Kernel radius computed via IM's `GetOptimalKernelWidth2D`
/// algorithm for exact Q16-compatible truncation.
pub fn gaussian_blur_mask(data: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = im_gaussian_kernel_radius(sigma);
    if radius == 0 {
        return data.to_vec();
    }

    // Build 1D Gaussian kernel with IM-matched radius
    let ksize = 2 * radius + 1;
    let mut kernel = vec![0.0f64; ksize];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x * inv_2s2).exp();
        sum += kernel[i];
    }
    for v in &mut kernel {
        *v /= sum;
    }

    // Horizontal pass
    let mut tmp = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for ki in 0..ksize {
                let src_col = col as isize + ki as isize - radius as isize;
                if src_col >= 0 && (src_col as usize) < w {
                    acc += data[row * w + src_col as usize] * kernel[ki];
                }
            }
            tmp[row * w + col] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for ki in 0..ksize {
                let src_row = row as isize + ki as isize - radius as isize;
                if src_row >= 0 && (src_row as usize) < h {
                    acc += tmp[src_row as usize * w + col] * kernel[ki];
                }
            }
            out[row * w + col] = acc;
        }
    }

    out
}

