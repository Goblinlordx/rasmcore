//! Denoise helpers for filters.

#[allow(unused_imports)]
use super::*;


pub fn guided_filter_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &GuidedFilterParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;
    let epsilon = config.epsilon;

    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "guided filter requires Gray8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let r = radius as usize;
    let eps = epsilon;

    // Convert to f32
    let input: Vec<f32> = pixels.iter().map(|&v| v as f32 / 255.0).collect();

    // For self-guided: guide = input
    let guide = &input;

    // Box mean via integral image (O(1) per pixel)
    let mean_i = box_mean(&input, w, h, r);
    let mean_p = box_mean(&input, w, h, r); // p = input for self-guided

    // mean(I*p)
    let ip: Vec<f32> = input
        .iter()
        .zip(guide.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let mean_ip = box_mean(&ip, w, h, r);

    // mean(I*I)
    let ii: Vec<f32> = guide.iter().map(|&v| v * v).collect();
    let mean_ii = box_mean(&ii, w, h, r);

    // Compute a and b for each window
    let n = w * h;
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    for i in 0..n {
        let cov_ip = mean_ip[i] - mean_i[i] * mean_p[i];
        let var_i = mean_ii[i] - mean_i[i] * mean_i[i];
        a[i] = cov_ip / (var_i + eps);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }

    // Average a and b over window
    let mean_a = box_mean(&a, w, h, r);
    let mean_b = box_mean(&b, w, h, r);

    // Output: mean_a * I + mean_b
    let mut result = vec![0u8; n];
    for i in 0..n {
        let v = (mean_a[i] * guide[i] + mean_b[i]) * 255.0;
        result[i] = v.round().clamp(0.0, 255.0) as u8;
    }

    Ok(result)
}

/// Hat-shaped weighting function for Debevec method.
/// w(z) = z + 1 for z <= 127, 256 - z for z >= 128.
/// Gives highest weight to mid-tone pixels.
#[inline]
pub fn hat_weight(z: usize) -> f64 {
    if z <= 127 {
        (z + 1) as f64
    } else {
        (256 - z) as f64
    }
}

/// Non-local means denoising for grayscale images.
///
/// With `NlmAlgorithm::OpenCv` (default): replicates OpenCV's
/// `fastNlMeansDenoising` exactly — integer SSD with bit-shift division
/// to approximate average, precomputed weight LUT indexed by integer
/// almost-average-distance, fixed-point integer accumulation.
///
/// With `NlmAlgorithm::Classic`: standard Buades et al. 2005 with float math.
pub fn nlm_denoise(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "NLM denoising currently supports Gray8 only".into(),
        ));
    }
    match params.algorithm {
        NlmAlgorithm::OpenCv => nlm_denoise_opencv(pixels, info, params),
        NlmAlgorithm::Classic => nlm_denoise_classic(pixels, info, params),
    }
}

/// Classic NLM (Buades 2005) with float math.
pub fn nlm_denoise_classic(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let ps = params.patch_size as usize;
    let ss = params.search_size as usize;
    let pr = ps / 2;
    let sr = ss / 2;
    let h2 = params.h * params.h;
    let patch_area = (ps * ps) as f32;

    let mut out = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut weight_sum: f32 = 0.0;
            let mut pixel_sum: f32 = 0.0;

            let sy_start = (y as i32 - sr as i32).max(0) as usize;
            let sy_end = (y + sr + 1).min(h);
            let sx_start = (x as i32 - sr as i32).max(0) as usize;
            let sx_end = (x + sr + 1).min(w);

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    let mut ssd: f32 = 0.0;
                    for py in 0..ps {
                        for ppx in 0..ps {
                            let y1 = reflect101(y as isize + py as isize - pr as isize, h as isize)
                                as usize;
                            let x1 = reflect101(x as isize + ppx as isize - pr as isize, w as isize)
                                as usize;
                            let y2 = reflect101(sy as isize + py as isize - pr as isize, h as isize)
                                as usize;
                            let x2 =
                                reflect101(sx as isize + ppx as isize - pr as isize, w as isize)
                                    as usize;
                            let d = pixels[y1 * w + x1] as f32 - pixels[y2 * w + x2] as f32;
                            ssd += d * d;
                        }
                    }
                    let weight = (-ssd / (patch_area * h2)).exp();
                    weight_sum += weight;
                    pixel_sum += weight * pixels[sy * w + sx] as f32;
                }
            }

            out[y * w + x] = if weight_sum > 0.0 {
                (pixel_sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                pixels[y * w + x]
            };
        }
    }

    Ok(out)
}

/// OpenCV-exact NLM implementation.
///
/// Replicates `FastNlMeansDenoisingInvoker` from OpenCV 4.x:
/// - `copyMakeBorder(BORDER_DEFAULT)` → reflect101 padding
/// - Integer SSD between patches
/// - `almostAvgDist = ssd >> bin_shift` (bit-shift approximation of SSD/N)
/// - Precomputed `almost_dist2weight[almostAvgDist]` LUT
/// - Fixed-point integer accumulation with `fixed_point_mult`
/// - `divByWeightsSum` with rounding
pub fn nlm_denoise_opencv(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let tw = params.patch_size as usize; // template window size
    let sw = params.search_size as usize; // search window size
    let thr = tw / 2; // template half radius
    let shr = sw / 2; // search half radius
    let border = shr + thr;

    // Create border-extended image (BORDER_REFLECT_101)
    let ew = w + 2 * border;
    let eh = h + 2 * border;
    let mut ext = vec![0u8; ew * eh];
    for ey in 0..eh {
        for ex in 0..ew {
            let sy = reflect101(ey as isize - border as isize, h as isize) as usize;
            let sx = reflect101(ex as isize - border as isize, w as isize) as usize;
            ext[ey * ew + ex] = pixels[sy * w + sx];
        }
    }

    // Precompute weight LUT (matches OpenCV's constructor)
    let tw_sq = tw * tw;
    let bin_shift = {
        let mut p = 0u32;
        while (1u32 << p) < tw_sq as u32 {
            p += 1;
        }
        p
    };
    let almost_dist2actual: f64 = (1u64 << bin_shift) as f64 / tw_sq as f64;
    // DistSquared::maxDist<uchar>() = sampleMax * sampleMax * channels = 255*255*1
    let max_dist: i32 = 255 * 255;
    let almost_max_dist = (max_dist as f64 / almost_dist2actual + 1.0) as usize;

    // fixed_point_mult: max value that won't overflow i32 accumulation
    let max_estimate_sum = sw as i64 * sw as i64 * 255i64;
    let fixed_point_mult = (i32::MAX as i64 / max_estimate_sum).min(255) as i32;

    let weight_threshold = (0.001 * fixed_point_mult as f64) as i32;

    let mut lut = vec![0i32; almost_max_dist];
    for (ad, lut_entry) in lut.iter_mut().enumerate().take(almost_max_dist) {
        let dist = ad as f64 * almost_dist2actual;
        // OpenCV DistSquared::calcWeight: exp(-dist / (h*h * channels))
        // Note: -dist (NOT -dist*dist) because dist is already squared per-pixel distance.
        // For grayscale (channels=1): exp(-dist / (h*h))
        let wf = (-dist / (params.h as f64 * params.h as f64)).exp();
        let wi = (fixed_point_mult as f64 * wf + 0.5) as i32;
        *lut_entry = if wi < weight_threshold { 0 } else { wi };
    }

    let mut out = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut estimation: i64 = 0;
            let mut weights_sum: i64 = 0;

            // For each search window position
            for sy in 0..sw {
                for sx in 0..sw {
                    // Compute SSD between patches (integer)
                    let mut ssd: i32 = 0;
                    for ty in 0..tw {
                        for tx in 0..tw {
                            let a_y = border + y - thr + ty;
                            let a_x = border + x - thr + tx;
                            let b_y = border + y - shr + sy - thr + ty;
                            let b_x = border + x - shr + sx - thr + tx;
                            let a = ext[a_y * ew + a_x] as i32;
                            let b = ext[b_y * ew + b_x] as i32;
                            ssd += (a - b) * (a - b);
                        }
                    }

                    let almost_avg_dist = (ssd >> bin_shift) as usize;
                    let weight = lut[almost_avg_dist.min(lut.len() - 1)] as i64;

                    let p = ext[(border + y - shr + sy) * ew + (border + x - shr + sx)] as i64;
                    estimation += weight * p;
                    weights_sum += weight;
                }
            }

            // OpenCV divByWeightsSum: (unsigned(estimation) + weights_sum/2) / weights_sum
            out[y * w + x] = if weights_sum > 0 {
                ((estimation as u64 + weights_sum as u64 / 2) / weights_sum as u64).min(255) as u8
            } else {
                pixels[y * w + x]
            };
        }
    }

    Ok(out)
}

/// Multi-Scale Retinex (MSR).
///
/// Averages SSR at multiple Gaussian scales for balanced enhancement across
/// fine and coarse detail. Default scales: sigma = [15, 80, 250].
///
/// `MSR(x,y) = (1/N) * sum_i [log(I(x,y)) - log(G(x,y,sigma_i) * I(x,y))]`
///
/// - `sigmas`: Gaussian scales (typical: &[15.0, 80.0, 250.0])
///
/// Reference: Jobson, Rahman, Woodell — "A Multiscale Retinex for Bridging
/// the Gap Between Color Images and the Human Observation of Scenes"
/// (IEEE Trans. Image Processing, 1997)
pub fn retinex_msr(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    sigmas: &[f32],
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "retinex requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let num_scales = sigmas.len() as f32;

    // Accumulate retinex across scales
    let mut retinex = vec![0.0f32; n * 3];

    for &sigma in sigmas {
        let blurred = {
            let r = Rect::new(0, 0, info.width, info.height);
            let mut u = |_: Rect| Ok(pixels.to_vec());
            gaussian_blur_cv(r, &mut u, info, &GaussianBlurCvParams { sigma })?
        };
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let orig = (pixels[pi + c] as f32).max(1.0);
                let surround = (blurred[pi + c] as f32).max(1.0);
                retinex[i * 3 + c] += (orig.ln() - surround.ln()) / num_scales;
            }
        }
    }

    // Normalize to [0, 255]
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in &retinex {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = (max_val - min_val).max(1e-6);

    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let v = (retinex[i * 3 + c] - min_val) / range * 255.0;
            result[pi + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}

/// Multi-Scale Retinex with Color Restoration (MSRCR).
///
/// Extends MSR with a color restoration factor that prevents desaturation:
/// `MSRCR(x,y) = C(x,y) * MSR(x,y)`
/// where `C(x,y) = beta * log(alpha * I_c / sum(I_channels))`
///
/// - `sigmas`: Gaussian scales (typical: &[15.0, 80.0, 250.0])
/// - `alpha`: color restoration nonlinearity (typical: 125.0)
/// - `beta`: color restoration gain (typical: 46.0)
///
/// Reference: Jobson, Rahman, Woodell — "A Multiscale Retinex for Bridging
/// the Gap Between Color Images and the Human Observation of Scenes"
/// (IEEE Trans. Image Processing, 1997)
pub fn retinex_msrcr(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    sigmas: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "retinex requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let num_scales = sigmas.len() as f32;

    // Compute MSR (OpenCV-compatible blur for reference alignment)
    let mut msr = vec![0.0f32; n * 3];
    for &sigma in sigmas {
        let blurred = {
            let r = Rect::new(0, 0, info.width, info.height);
            let mut u = |_: Rect| Ok(pixels.to_vec());
            gaussian_blur_cv(r, &mut u, info, &GaussianBlurCvParams { sigma })?
        };
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let orig = (pixels[pi + c] as f32).max(1.0);
                let surround = (blurred[pi + c] as f32).max(1.0);
                msr[i * 3 + c] += (orig.ln() - surround.ln()) / num_scales;
            }
        }
    }

    // Color restoration: C(x,y) = beta * log(alpha * I_c / sum(I))
    let mut msrcr = vec![0.0f32; n * 3];
    for i in 0..n {
        let pi = i * channels;
        let sum_channels = pixels[pi] as f32 + pixels[pi + 1] as f32 + pixels[pi + 2] as f32;
        let sum_channels = sum_channels.max(1.0);
        for c in 0..3 {
            let ic = (pixels[pi + c] as f32).max(1.0);
            let color_restore = beta * (alpha * ic / sum_channels).ln();
            msrcr[i * 3 + c] = color_restore * msr[i * 3 + c];
        }
    }

    // Normalize to [0, 255]
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in &msrcr {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = (max_val - min_val).max(1e-6);

    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let v = (msrcr[i * 3 + c] - min_val) / range * 255.0;
            result[pi + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}

pub fn weighted_median_val(
    sorted: &[(f32, f32, f32)],
    total_weight: f32,
    val: impl Fn(&(f32, f32, f32)) -> f32,
) -> f32 {
    let half = total_weight / 2.0;
    let mut accum = 0.0f32;
    for item in sorted {
        accum += item.2;
        if accum >= half {
            return val(item);
        }
    }
    val(sorted.last().unwrap())
}


pub fn stackable_box_blur_f32(data: &mut [f32], w: usize, h: usize, sigma: f32) {
    let radii = box_blur_radii_for_gaussian(sigma);
    for &r in &radii {
        box_blur_pass_f32(data, w, h, r);
    }
}

pub fn yvv_blur_1d(buf: &mut [f32], b: &[f64; 4], _m: &[[f64; 3]; 3]) {
    let n = buf.len();
    if n < 4 {
        return;
    }

    // Pad with 3*sigma samples on each side (replicated edge) to let the
    // IIR settle. For the coefficients we use, 3 samples of warm-up on
    // each side is the theoretical minimum, but more padding gives better
    // boundary behavior. Use max(ceil(3*sigma), 32) padding, capped at n.
    // Since we don't have sigma here, use a generous fixed padding.
    let pad = n.min(64);

    let total = pad + n + pad;
    let mut tmp = vec![0.0f64; total];

    // Fill: left pad + data + right pad
    let left_val = buf[0] as f64;
    let right_val = buf[n - 1] as f64;
    for val in tmp.iter_mut().take(pad) {
        *val = left_val;
    }
    for (i, val) in tmp.iter_mut().skip(pad).take(n).enumerate() {
        *val = buf[i] as f64;
    }
    for val in tmp.iter_mut().skip(pad + n).take(pad) {
        *val = right_val;
    }

    // Forward (causal) pass
    let mut y1 = tmp[0];
    let mut y2 = tmp[0];
    let mut y3 = tmp[0];
    for val in tmp.iter_mut() {
        let y = b[0] * *val + b[1] * y1 + b[2] * y2 + b[3] * y3;
        *val = y;
        y3 = y2;
        y2 = y1;
        y1 = y;
    }

    // Backward (anti-causal) pass
    y1 = tmp[total - 1];
    y2 = tmp[total - 1];
    y3 = tmp[total - 1];
    for val in tmp.iter_mut().rev() {
        let y = b[0] * *val + b[1] * y1 + b[2] * y2 + b[3] * y3;
        *val = y;
        y3 = y2;
        y2 = y1;
        y1 = y;
    }

    // Extract the valid region
    for i in 0..n {
        buf[i] = tmp[pad + i] as f32;
    }
}

pub fn yvv_find_constants(sigma: f32) -> ([f64; 4], [[f64; 3]; 3]) {
    let sigma = sigma as f64;
    let k1 = 2.44413;
    let k2 = 1.4281;
    let k3 = 0.422205;

    let q = if sigma >= 2.5 {
        0.98711 * sigma - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    };

    let b0 = 1.57825 + q * (k1 + q * (k2 + q * k3));
    let b1_raw = q * (k1 + q * (2.0 * k2 + q * 3.0 * k3));
    let b2_raw = -k2 * q * q - k3 * 3.0 * q * q * q;
    let b3_raw = q * q * q * k3;

    let a1 = b1_raw / b0;
    let a2 = b2_raw / b0;
    let a3 = b3_raw / b0;

    let b = [1.0 - (a1 + a2 + a3), a1, a2, a3];

    // Right-boundary correction matrix (GEGL's fix_right_boundary)
    let c = 1.0 / ((1.0 + a1 - a2 + a3) * (1.0 + a2 + (a1 - a3) * a3));

    let m = [
        [
            c * (-a3 * (a1 + a3) - a2 + 1.0),
            c * (a3 + a1) * (a2 + a3 * a1),
            c * a3 * (a1 + a3 * a2),
        ],
        [
            c * (a1 + a3 * a2),
            c * (1.0 - a2) * (a2 + a3 * a1),
            c * a3 * (1.0 - a3 * a1 - a3 * a3 - a2),
        ],
        [
            c * (a3 * a1 + a2 + a1 * a1 - a2 * a2),
            c * (a1 * a2 + a3 * a2 * a2 - a1 * a3 * a3 - a3 * a3 * a3 - a3 * a2 + a3),
            c * a3 * (a1 + a3 * a2),
        ],
    ];

    (b, m)
}
