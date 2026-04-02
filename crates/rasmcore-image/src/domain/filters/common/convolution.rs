//! Convolution helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Two-pass separable convolution: horizontal then vertical.
pub fn convolve_separable(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    row_k: &[f32],
    col_k: &[f32],
    divisor: f32,
) -> Result<Vec<u8>, ImageError> {
    let rw = row_k.len() / 2;
    let rh = col_k.len() / 2;
    let pad = rw.max(rh);
    let inv_div = 1.0 / divisor;

    // Pad input
    let padded = pad_reflect(pixels, w, h, channels, pad);
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;

    // Pass 1: horizontal convolution → intermediate f32 buffer
    let mut tmp = vec![0.0f32; ph * w * channels];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        // Process 4 output values at a time with f32x4
        let total_out = ph * w * channels;
        let simd_chunks = total_out / 4;

        for chunk in 0..simd_chunks {
            let out_base = chunk * 4;
            // Determine y, x, c for each of the 4 outputs
            // Since channels is typically 1, 3, or 4, batch across x*channels
            let mut accum = f32x4_splat(0.0);

            for kx in 0..row_k.len() {
                let k_val = f32x4_splat(row_k[kx]);
                // Compute source indices for each of the 4 outputs
                let mut vals = [0.0f32; 4];
                for lane in 0..4 {
                    let idx = out_base + lane;
                    let y = idx / (w * channels);
                    let rem = idx % (w * channels);
                    let x = rem / channels;
                    let c = rem % channels;
                    let src_idx = (y * pw + x + pad - rw + kx) * channels + c;
                    vals[lane] = padded[src_idx] as f32;
                }
                let src_vec = f32x4(vals[0], vals[1], vals[2], vals[3]);
                accum = f32x4_add(accum, f32x4_mul(k_val, src_vec));
            }

            tmp[out_base] = f32x4_extract_lane::<0>(accum);
            tmp[out_base + 1] = f32x4_extract_lane::<1>(accum);
            tmp[out_base + 2] = f32x4_extract_lane::<2>(accum);
            tmp[out_base + 3] = f32x4_extract_lane::<3>(accum);
        }
        // Remainder
        for idx in simd_chunks * 4..total_out {
            let y = idx / (w * channels);
            let rem = idx % (w * channels);
            let x = rem / channels;
            let c = rem % channels;
            let mut sum = 0.0f32;
            for kx in 0..row_k.len() {
                sum += row_k[kx] * padded[(y * pw + x + pad - rw + kx) * channels + c] as f32;
            }
            tmp[idx] = sum;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for y in 0..ph {
            for x in 0..w {
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for kx in 0..row_k.len() {
                        sum +=
                            row_k[kx] * padded[(y * pw + x + pad - rw + kx) * channels + c] as f32;
                    }
                    tmp[(y * w + x) * channels + c] = sum;
                }
            }
        }
    }

    // Pass 2: vertical convolution on intermediate buffer
    let mut out = vec![0u8; w * h * channels];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let inv_div_vec = f32x4_splat(inv_div);
        let zero = f32x4_splat(0.0);
        let max_val = f32x4_splat(255.0);
        let total_out = w * h * channels;
        let simd_chunks = total_out / 4;

        for chunk in 0..simd_chunks {
            let out_base = chunk * 4;
            let mut accum = f32x4_splat(0.0);

            for ky in 0..col_k.len() {
                let k_val = f32x4_splat(col_k[ky]);
                let mut vals = [0.0f32; 4];
                for lane in 0..4 {
                    let idx = out_base + lane;
                    let y = idx / (w * channels);
                    let rem = idx % (w * channels);
                    let x = rem / channels;
                    let c = rem % channels;
                    let src_idx = ((y + pad - rh + ky) * w + x) * channels + c;
                    vals[lane] = tmp[src_idx];
                }
                let src_vec = f32x4(vals[0], vals[1], vals[2], vals[3]);
                accum = f32x4_add(accum, f32x4_mul(k_val, src_vec));
            }

            let scaled = f32x4_mul(accum, inv_div_vec);
            let clamped = f32x4_max(zero, f32x4_min(max_val, scaled));
            out[out_base] = f32x4_extract_lane::<0>(clamped) as u8;
            out[out_base + 1] = f32x4_extract_lane::<1>(clamped) as u8;
            out[out_base + 2] = f32x4_extract_lane::<2>(clamped) as u8;
            out[out_base + 3] = f32x4_extract_lane::<3>(clamped) as u8;
        }
        // Remainder
        for idx in simd_chunks * 4..total_out {
            let y = idx / (w * channels);
            let rem = idx % (w * channels);
            let x = rem / channels;
            let c = rem % channels;
            let mut sum = 0.0f32;
            for ky in 0..col_k.len() {
                sum += col_k[ky] * tmp[((y + pad - rh + ky) * w + x) * channels + c];
            }
            out[idx] = (sum * inv_div).round().clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for ky in 0..col_k.len() {
                        sum += col_k[ky] * tmp[((y + pad - rh + ky) * w + x) * channels + c];
                    }
                    out[(y * w + x) * channels + c] =
                        (sum * inv_div).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Two-pass separable convolution on f32 channel data.
///
/// Same algorithm as `convolve_separable` but operates on `&[f32]` channel
/// values and returns `Vec<f32>`. No u8 conversion — full f32 precision
/// throughout. Used by the f32 pipeline path.
pub fn convolve_separable_f32(
    samples: &[f32],
    w: usize,
    h: usize,
    channels: usize,
    row_k: &[f32],
    col_k: &[f32],
    divisor: f32,
) -> Result<Vec<f32>, ImageError> {
    let rw = row_k.len() / 2;
    let rh = col_k.len() / 2;
    let pad = rw.max(rh);
    let inv_div = 1.0 / divisor;

    // Pad input (reflect border)
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    let mut padded = vec![0.0f32; pw * ph * channels];
    for py in 0..ph {
        let sy = reflect(py as i32 - pad as i32, h);
        for px in 0..pw {
            let sx = reflect(px as i32 - pad as i32, w);
            let src = (sy * w + sx) * channels;
            let dst = (py * pw + px) * channels;
            padded[dst..dst + channels].copy_from_slice(&samples[src..src + channels]);
        }
    }

    // Pass 1: horizontal convolution → intermediate buffer
    let mut tmp = vec![0.0f32; ph * w * channels];
    for y in 0..ph {
        for x in 0..w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for kx in 0..row_k.len() {
                    sum += row_k[kx] * padded[(y * pw + x + pad - rw + kx) * channels + c];
                }
                tmp[(y * w + x) * channels + c] = sum;
            }
        }
    }

    // Pass 2: vertical convolution → output
    let mut out = vec![0.0f32; w * h * channels];
    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for ky in 0..col_k.len() {
                    sum += col_k[ky] * tmp[((y + pad - rh + ky) * w + x) * channels + c];
                }
                out[(y * w + x) * channels + c] = sum * inv_div;
            }
        }
    }

    Ok(out)
}

/// Generate a 1D Gaussian kernel matching OpenCV's `getGaussianKernel`.
///
/// `k[i] = exp(-0.5 * ((i - center) / sigma)^2)`, normalized to sum=1.
pub fn gaussian_kernel_1d(ksize: usize, sigma: f32) -> Vec<f32> {
    let center = (ksize / 2) as f32;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    let inv_sum = 1.0 / sum;
    for v in &mut kernel {
        *v *= inv_sum;
    }
    kernel
}

/// Compute the optimal Gaussian kernel width for a given sigma.
///
/// Reimplements ImageMagick 7's `GetOptimalKernelWidth2D` from `gem.c`:
/// starts at width 5 and grows by 2 until the normalized edge value of
/// the 2D Gaussian drops below `1.0 / quantum_range`.
///
/// For Q16 (quantum_range = 65535) this produces kernel radii that exactly
/// match ImageMagick 7.1.x's `-vignette` and `-gaussian-blur` operators.
pub fn im_gaussian_kernel_radius(sigma: f64) -> usize {
    const QUANTUM_SCALE: f64 = 1.0 / 65535.0; // Q16
    let gamma = sigma.abs();
    if gamma < 1.0e-12 {
        return 1;
    }
    let alpha = 1.0 / (2.0 * gamma * gamma);
    let beta = 1.0 / (2.0 * std::f64::consts::PI * gamma * gamma);

    let mut width: usize = 5;
    loop {
        let j = (width - 1) / 2;
        let ji = j as isize;
        let mut normalize = 0.0f64;
        for v in -ji..=ji {
            for u in -ji..=ji {
                normalize += (-(((u * u + v * v) as f64) * alpha)).exp() * beta;
            }
        }
        let value = (-((j * j) as f64 * alpha)).exp() * beta / normalize;
        if value < QUANTUM_SCALE || value < 1.0e-12 {
            break;
        }
        width += 2;
    }
    (width - 2 - 1) / 2 // convert width to radius
}

/// Detect if a 2D kernel is separable (rank-1: K = col * row^T).
///
/// Returns `Some((row_kernel, col_kernel))` if separable, `None` otherwise.
pub fn is_separable(kernel: &[f32], kw: usize, kh: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    // Find the first non-zero row to use as reference
    let mut ref_row = None;
    for r in 0..kh {
        let row_sum: f32 = (0..kw).map(|c| kernel[r * kw + c].abs()).sum();
        if row_sum > 1e-10 {
            ref_row = Some(r);
            break;
        }
    }
    let ref_row = ref_row?;

    // Extract row kernel from the reference row
    let row_k: Vec<f32> = (0..kw).map(|c| kernel[ref_row * kw + c]).collect();

    // Find the first non-zero element in the reference row for column scale
    let ref_col = (0..kw).find(|&c| row_k[c].abs() > 1e-10)?;
    let scale = row_k[ref_col];

    // Extract column kernel: col[r] = kernel[r][ref_col] / scale
    let col_k: Vec<f32> = (0..kh).map(|r| kernel[r * kw + ref_col] / scale).collect();

    // Verify: kernel[r][c] ≈ col[r] * row[c] for all r, c
    for r in 0..kh {
        for c in 0..kw {
            let expected = col_k[r] * row_k[c];
            if (kernel[r * kw + c] - expected).abs() > 1e-4 {
                return None;
            }
        }
    }

    Some((row_k, col_k))
}

pub fn sharpen_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &SharpenParams,
) -> Result<Vec<u8>, ImageError> {
    let amount = config.amount;

    validate_format(info.format)?;

    // 16-bit: work in f32 for full precision
    if is_16bit(info.format) {
        let orig_f32 = u16_pixels_to_f32(pixels);
        let _info_8 = ImageInfo {
            format: match info.format {
                PixelFormat::Rgb16 => PixelFormat::Rgb8,
                PixelFormat::Rgba16 => PixelFormat::Rgba8,
                PixelFormat::Gray16 => PixelFormat::Gray8,
                other => other,
            },
            ..*info
        };
        let blurred = blur_impl(pixels, info, &BlurParams { radius: 1.0 })?;
        let blur_f32 = u16_pixels_to_f32(&blurred);
        let result_f32: Vec<f32> = orig_f32
            .iter()
            .zip(blur_f32.iter())
            .map(|(&o, &b)| (o + amount * (o - b)).clamp(0.0, 1.0))
            .collect();
        return Ok(f32_to_u16_pixels(&result_f32));
    }

    // Blur with a small radius for the unsharp mask
    let blurred = blur_impl(pixels, info, &BlurParams { radius: 1.0 })?;

    let mut result = vec![0u8; pixels.len()];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let amount_vec = f32x4_splat(amount);
        let zero = f32x4_splat(0.0);
        let max_val = f32x4_splat(255.0);
        let len = pixels.len();
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            // Load 4 bytes, widen to f32x4
            let orig = f32x4(
                pixels[base] as f32,
                pixels[base + 1] as f32,
                pixels[base + 2] as f32,
                pixels[base + 3] as f32,
            );
            let blur_v = f32x4(
                blurred[base] as f32,
                blurred[base + 1] as f32,
                blurred[base + 2] as f32,
                blurred[base + 3] as f32,
            );
            let diff = f32x4_sub(orig, blur_v);
            let scaled = f32x4_mul(diff, amount_vec);
            let sharp = f32x4_add(orig, scaled);
            let clamped = f32x4_max(zero, f32x4_min(max_val, sharp));

            result[base] = f32x4_extract_lane::<0>(clamped) as u8;
            result[base + 1] = f32x4_extract_lane::<1>(clamped) as u8;
            result[base + 2] = f32x4_extract_lane::<2>(clamped) as u8;
            result[base + 3] = f32x4_extract_lane::<3>(clamped) as u8;
        }
        // Remainder
        for i in chunks * 4..len {
            let orig = pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            let sharp = orig + amount * (orig - blur_val);
            result[i] = sharp.clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..pixels.len() {
            let orig = pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            let sharp = orig + amount * (orig - blur_val);
            result[i] = sharp.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// Box mean via integral image — O(1) per pixel regardless of radius.
/// Box mean matching OpenCV's boxFilter with BORDER_REFLECT.
/// Pads data with reflect border, computes f32 SAT, queries fixed-size window.
pub fn box_mean(data: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let n = w * h;
    let r = radius;
    let ksize = 2 * r + 1;

    // Pad with BORDER_REFLECT (edge pixel duplicated)
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0.0f32; pw * ph];
    for py in 0..ph {
        // BORDER_REFLECT: idx < 0 → |idx+1|, idx >= size → 2*size - idx - 1
        let sy = if py < r {
            r - 1 - py // reflect with duplication
        } else if py >= h + r {
            2 * h - (py - r) - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                r - 1 - px
            } else if px >= w + r {
                2 * w - (px - r) - 1
            } else {
                px - r
            };
            padded[py * pw + px] = data[sy * w + sx];
        }
    }

    // Build SAT in f32
    let mut sat = vec![0.0f32; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] =
                padded[y * pw + x] + sat[y * (pw + 1) + (x + 1)] + sat[(y + 1) * (pw + 1) + x]
                    - sat[y * (pw + 1) + x];
        }
    }

    // Query: fixed ksize window centered on each original pixel
    let count = (ksize * ksize) as f32;
    let mut result = vec![0.0f32; n];
    for y in 0..h {
        let y0 = y; // in padded coords, original pixel y is at py = y + r, window starts at y
        let y1 = y + ksize;
        for x in 0..w {
            let x0 = x;
            let x1 = x + ksize;
            let sum = sat[y1 * (pw + 1) + x1] - sat[y0 * (pw + 1) + x1] - sat[y1 * (pw + 1) + x0]
                + sat[y0 * (pw + 1) + x0];
            result[y * w + x] = sum / count;
        }
    }

    result
}

/// Box mean via integral image (f64 precision).
#[allow(dead_code)] // reserved for future adaptive-threshold modes
pub fn box_mean_f64(data: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let n = w * h;
    let r = radius;

    // Build integral image with BORDER_REFLECT_101 padding
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0.0f64; pw * ph];
    for py in 0..ph {
        let sy = if py < r {
            r - 1 - py
        } else if py >= h + r {
            2 * h - (py - r) - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                r - 1 - px
            } else if px >= w + r {
                2 * w - (px - r) - 1
            } else {
                px - r
            };
            padded[py * pw + px] = data[sy.min(h - 1) * w + sx.min(w - 1)];
        }
    }

    // Build SAT
    let mut sat = vec![0.0f64; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] =
                padded[y * pw + x] + sat[y * (pw + 1) + (x + 1)] + sat[(y + 1) * (pw + 1) + x]
                    - sat[y * (pw + 1) + x];
        }
    }

    // Query box means
    let ksize = 2 * r + 1;
    let area = (ksize * ksize) as f64;
    let mut result = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let y1 = y;
            let x1 = x;
            let y2 = y + ksize;
            let x2 = x + ksize;
            let sum = sat[y2 * (pw + 1) + x2] - sat[y1 * (pw + 1) + x2] - sat[y2 * (pw + 1) + x1]
                + sat[y1 * (pw + 1) + x1];
            result[y * w + x] = sum / area;
        }
    }

    result
}

/// Integer box mean via integral image with BORDER_REPLICATE padding.
/// Matches OpenCV's boxFilter(src, CV_8U, ksize, BORDER_REPLICATE) exactly.
pub fn box_mean_u8_replicate(pixels: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let r = radius;
    let ksize = 2 * r + 1;

    // Pad with BORDER_REPLICATE: clamp to edge
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0u32; pw * ph];
    for py in 0..ph {
        let sy = if py < r {
            0
        } else if py >= h + r {
            h - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                0
            } else if px >= w + r {
                w - 1
            } else {
                px - r
            };
            padded[py * pw + px] = pixels[sy * w + sx] as u32;
        }
    }

    // Build integral image (i64 for safe subtraction)
    let mut sat = vec![0i64; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] = padded[y * pw + x] as i64
                + sat[y * (pw + 1) + (x + 1)]
                + sat[(y + 1) * (pw + 1) + x]
                - sat[y * (pw + 1) + x];
        }
    }

    // Query box means with rounded integer division (matches OpenCV boxFilter CV_8U)
    let area = (ksize * ksize) as i64;
    let half_area = area / 2;
    let n = w * h;
    let mut result = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let y1 = y;
            let x1 = x;
            let y2 = y + ksize;
            let x2 = x + ksize;
            let sum = sat[y2 * (pw + 1) + x2] - sat[y1 * (pw + 1) + x2] - sat[y2 * (pw + 1) + x1]
                + sat[y1 * (pw + 1) + x1];
            result[y * w + x] = ((sum + half_area) / area) as u8;
        }
    }

    result
}

