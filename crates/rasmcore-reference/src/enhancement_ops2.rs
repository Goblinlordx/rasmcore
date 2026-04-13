//! Enhancement operation reference implementations — vignette, shadow/highlight,
//! clarity, dehaze, denoising, retinex, frequency separation, and pyramid detail.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Pure math, no dependencies, no optimization.

// ─── Inline Helpers ─────────────────────────────────────────────────────────

/// Rec. 709 luminance from linear RGB.
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// BORDER_REFLECT_101: mirrors at boundary, excluding edge pixel.
fn reflect_101(v: i32, size: i32) -> i32 {
    if size <= 1 { return 0; }
    let mut v = v;
    if v < 0 { v = -v; }
    if v >= size { v = 2 * size - v - 2; }
    v.clamp(0, size - 1)
}

/// Build a 1D Gaussian kernel matching the pipeline convention.
///
/// ksize = round(sigma * 10 + 1) | 1, min 3. Sigma = radius parameter.
fn build_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let ksize = ((sigma * 10.0 + 1.0).round() as usize) | 1;
    let ksize = ksize.max(3);
    let center = ksize / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center as f32;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    kernel
}

/// Separable Gaussian blur on RGBA interleaved f32 data.
///
/// ksize = round(sigma * 10 + 1) | 1, BORDER_REFLECT_101.
/// Processes R, G, B independently; alpha is copied unchanged.
fn gaussian_blur(input: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return input.to_vec();
    }
    let kernel = build_gaussian_kernel(sigma);
    let r = (kernel.len() / 2) as i32;
    let w = w as usize;
    let h = h as usize;

    // Horizontal pass.
    let mut temp = vec![0.0f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sx = reflect_101(x as i32 + ki as i32 - r, w as i32) as usize;
                let idx = (y * w + sx) * 4;
                acc[0] += input[idx] * weight;
                acc[1] += input[idx + 1] * weight;
                acc[2] += input[idx + 2] * weight;
            }
            let oidx = (y * w + x) * 4;
            temp[oidx] = acc[0];
            temp[oidx + 1] = acc[1];
            temp[oidx + 2] = acc[2];
            temp[oidx + 3] = input[oidx + 3]; // alpha
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &weight) in kernel.iter().enumerate() {
                let sy = reflect_101(y as i32 + ki as i32 - r, h as i32) as usize;
                let idx = (sy * w + x) * 4;
                acc[0] += temp[idx] * weight;
                acc[1] += temp[idx + 1] * weight;
                acc[2] += temp[idx + 2] * weight;
            }
            let oidx = (y * w + x) * 4;
            out[oidx] = acc[0];
            out[oidx + 1] = acc[1];
            out[oidx + 2] = acc[2];
            out[oidx + 3] = temp[oidx + 3]; // alpha
        }
    }

    out
}

/// Single-channel Gaussian blur for use by retinex and other per-channel ops.
/// Input/output are flat f32 arrays of length w*h.
fn gaussian_blur_single(input: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return input.to_vec();
    }
    let kernel = build_gaussian_kernel(sigma);
    let r = (kernel.len() / 2) as i32;

    // Horizontal pass.
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &weight) in kernel.iter().enumerate() {
                let sx = reflect_101(x as i32 + ki as i32 - r, w as i32) as usize;
                acc += input[y * w + sx] * weight;
            }
            temp[y * w + x] = acc;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &weight) in kernel.iter().enumerate() {
                let sy = reflect_101(y as i32 + ki as i32 - r, h as i32) as usize;
                acc += temp[sy * w + x] * weight;
            }
            out[y * w + x] = acc;
        }
    }

    out
}

/// Min-filter on a single-channel image (for dark channel prior).
/// Replaces each pixel with the minimum value in its (2*radius+1) neighborhood.
fn min_filter(input: &[f32], w: usize, h: usize, radius: u32) -> Vec<f32> {
    let r = radius as i32;

    // Horizontal min pass.
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            for dx in -r..=r {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                min_val = min_val.min(input[y * w + sx]);
            }
            temp[y * w + x] = min_val;
        }
    }

    // Vertical min pass.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            for dy in -r..=r {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                min_val = min_val.min(temp[sy * w + x]);
            }
            out[y * w + x] = min_val;
        }
    }

    out
}

// ─── Public Operations ──────────────────────────────────────────────────────

/// Elliptical vignette — binary mask + Gaussian blur.
///
/// 1. Build binary elliptical mask (1.0 inside, 0.0 outside).
/// 2. Blur the mask with Gaussian(sigma).
/// 3. Multiply RGB by blurred mask.
pub fn vignette(input: &[f32], w: u32, h: u32, sigma: f32, x_inset: f32, y_inset: f32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rx = (cx - x_inset).max(1.0);
    let ry = (cy - y_inset).max(1.0);

    // Build binary elliptical mask
    let mut mask = vec![0.0f32; wu * hu];
    for y in 0..hu {
        for x in 0..wu {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * wu + x] = 1.0;
            }
        }
    }

    // Blur the mask
    if sigma > 0.0 {
        mask = gaussian_blur_single(&mask, wu, hu, sigma);
    }

    // Multiply RGB by mask
    let mut out = input.to_vec();
    for y in 0..hu {
        for x in 0..wu {
            let factor = mask[y * wu + x];
            let idx = (y * wu + x) * 4;
            out[idx] *= factor;
            out[idx + 1] *= factor;
            out[idx + 2] *= factor;
        }
    }
    out
}

/// Power-law vignette.
///
/// `factor = 1 - strength * (dist / max_dist)^falloff`, clamped >= 0.
/// `dist` = Euclidean distance from center, `max_dist` = distance from center
/// to corner.
pub fn vignette_powerlaw(input: &[f32], w: u32, h: u32, strength: f32, falloff: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();

    if max_dist == 0.0 {
        return out;
    }

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let norm = dist / max_dist;
            let factor = (1.0 - strength * norm.powf(falloff)).max(0.0);

            let idx = (y * w + x) as usize * 4;
            out[idx] *= factor;
            out[idx + 1] *= factor;
            out[idx + 2] *= factor;
        }
    }
    out
}

/// Shadow / highlight recovery — full pipeline formula.
///
/// Uses blurred luminance for smooth weight estimation, quadratic shadow/highlight
/// weights with midtone compression, whitepoint offset, and chroma correction.
pub fn shadow_highlight(input: &[f32], w: u32, h: u32, shadows: f32, highlights: f32) -> Vec<f32> {
    // Default params matching pipeline defaults
    shadow_highlight_full(input, w, h, shadows, highlights, 0.0, 100.0, 50.0, 100.0, 100.0)
}

/// Full shadow/highlight with all parameters.
pub fn shadow_highlight_full(
    input: &[f32], w: u32, h: u32,
    shadows: f32, highlights: f32, whitepoint: f32,
    radius: f32, compress: f32,
    shadows_ccorrect: f32, highlights_ccorrect: f32,
) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let sh = shadows / 100.0;
    let hl = highlights / 100.0;
    let wp = whitepoint;
    let comp = compress / 100.0;
    let sc = shadows_ccorrect / 100.0;
    let hc = highlights_ccorrect / 100.0;

    // Compute per-pixel luminance
    let luma: Vec<f32> = input.chunks_exact(4)
        .map(|px| luminance(px[0], px[1], px[2]))
        .collect();

    // Blur luminance
    let blurred = gaussian_blur_single(&luma, wu, hu, radius);

    let mut out = input.to_vec();
    for i in 0..(wu * hu) {
        let bl = blurred[i];
        let mut sw = (1.0 - bl) * (1.0 - bl);
        let mut hw = bl * bl;
        if comp > 0.0 {
            let mid = 4.0 * bl * (1.0 - bl);
            sw *= 1.0 - comp * mid;
            hw *= 1.0 - comp * mid;
        }

        let luma_adj = sh * sw - hl * hw + wp * 0.01;
        let cur_luma = luma[i].max(1e-10);
        let new_luma = (cur_luma + luma_adj).max(0.0);
        let ratio = new_luma / cur_luma;

        let idx = i * 4;
        for c in 0..3 {
            let v = input[idx + c];
            let chroma = v - cur_luma;
            let sign_c = if chroma >= 0.0 { 1.0 } else { -1.0 };
            let sat_adj = (1.0 + sign_c * sw * (sc - 1.0) + sign_c * hw * (hc - 1.0)).max(0.0);
            out[idx + c] = new_luma + chroma * sat_adj * ratio;
        }
    }
    out
}

/// Clarity (midtone contrast enhancement).
///
/// Gaussian blur input, then apply midtone-weighted local contrast boost:
/// `out = in + amount * 4 * luma * (1 - luma) * (in - blur)`.
/// The midtone weight `4 * luma * (1 - luma)` peaks at luma = 0.5.
pub fn clarity(input: &[f32], w: u32, h: u32, amount: f32, radius: u32) -> Vec<f32> {
    let blurred = gaussian_blur(input, w, h, radius as f32);
    let mut out = input.to_vec();
    let npx = (w * h) as usize;

    for i in 0..npx {
        let idx = i * 4;
        let luma = luminance(input[idx], input[idx + 1], input[idx + 2]);
        let midtone_weight = 4.0 * luma * (1.0 - luma);
        for c in 0..3 {
            let detail = input[idx + c] - blurred[idx + c];
            out[idx + c] = input[idx + c] + amount * midtone_weight * detail;
        }
        // alpha unchanged
    }
    out
}

/// Dehaze via dark channel prior.
///
/// 1. Compute dark channel = min(R,G,B) per pixel, then min-filter over patch.
/// 2. Atmospheric light A = average of pixels corresponding to top 0.1% brightest
///    dark channel values.
/// 3. Transmission `t = 1 - omega * dark(I/A)`, clamped to `t_min`.
/// 4. Recover: `J = (I - A) / max(t, t_min) + A`.
pub fn dehaze(input: &[f32], w: u32, h: u32, patch_radius: u32, omega: f32, t_min: f32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let npx = wu * hu;

    // Step 1: dark channel (min of RGB per pixel).
    let mut dark_raw = vec![0.0f32; npx];
    for i in 0..npx {
        let idx = i * 4;
        dark_raw[i] = input[idx].min(input[idx + 1]).min(input[idx + 2]);
    }
    let dark_channel = min_filter(&dark_raw, wu, hu, patch_radius);

    // Step 2: atmospheric light — top 0.1% brightest dark channel pixels.
    let top_count = ((npx as f32 * 0.001).ceil() as usize).max(1);
    let mut indices: Vec<usize> = (0..npx).collect();
    indices.sort_by(|&a, &b| dark_channel[b].partial_cmp(&dark_channel[a]).unwrap());
    let mut a_sum = [0.0f64; 3];
    for &pi in indices.iter().take(top_count) {
        let idx = pi * 4;
        a_sum[0] += input[idx] as f64;
        a_sum[1] += input[idx + 1] as f64;
        a_sum[2] += input[idx + 2] as f64;
    }
    let atm = [
        (a_sum[0] / top_count as f64) as f32,
        (a_sum[1] / top_count as f64) as f32,
        (a_sum[2] / top_count as f64) as f32,
    ];

    // Step 3: transmission estimate.
    // Normalize input by atmospheric light, compute dark channel, derive t.
    let mut norm_dark = vec![0.0f32; npx];
    for i in 0..npx {
        let idx = i * 4;
        let nr = if atm[0] > 0.0 { input[idx] / atm[0] } else { input[idx] };
        let ng = if atm[1] > 0.0 { input[idx + 1] / atm[1] } else { input[idx + 1] };
        let nb = if atm[2] > 0.0 { input[idx + 2] / atm[2] } else { input[idx + 2] };
        norm_dark[i] = nr.min(ng).min(nb);
    }
    let norm_dark_filtered = min_filter(&norm_dark, wu, hu, patch_radius);

    let mut transmission = vec![0.0f32; npx];
    for i in 0..npx {
        transmission[i] = (1.0 - omega * norm_dark_filtered[i]).max(t_min);
    }

    // Step 4: recover scene radiance.
    let mut out = input.to_vec();
    for i in 0..npx {
        let idx = i * 4;
        let t = transmission[i].max(t_min);
        for c in 0..3 {
            out[idx + c] = (input[idx + c] - atm[c]) / t + atm[c];
        }
        // alpha unchanged
    }
    out
}

/// Non-local means denoising.
///
/// For each pixel, search within `search_radius` neighborhood, compute
/// patch SSD over `(2*patch_radius+1)²` window, weight = exp(-ssd / h²),
/// and output weighted average.
pub fn nlm_denoise(input: &[f32], w: u32, h: u32, h_param: f32, patch_radius: u32, search_radius: u32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let pr = patch_radius as i32;
    let sr = search_radius as i32;
    let h2 = h_param * h_param;
    let patch_area = ((2 * pr + 1) * (2 * pr + 1)) as f32;

    let mut out = input.to_vec();

    for y in 0..hu {
        for x in 0..wu {
            let mut weighted_sum = [0.0f32; 3];
            let mut weight_total = 0.0f32;

            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || nx >= wu as i32 || ny < 0 || ny >= hu as i32 {
                        continue;
                    }

                    // Compute patch SSD.
                    let mut ssd = 0.0f32;
                    for py in -pr..=pr {
                        for px in -pr..=pr {
                            let ax = (x as i32 + px).clamp(0, wu as i32 - 1) as usize;
                            let ay = (y as i32 + py).clamp(0, hu as i32 - 1) as usize;
                            let bx = (nx + px).clamp(0, wu as i32 - 1) as usize;
                            let by = (ny + py).clamp(0, hu as i32 - 1) as usize;

                            let ai = (ay * wu + ax) * 4;
                            let bi = (by * wu + bx) * 4;
                            for c in 0..3 {
                                let d = input[ai + c] - input[bi + c];
                                ssd += d * d;
                            }
                        }
                    }
                    ssd /= patch_area * 3.0;

                    let weight = (-ssd / h2).exp();
                    weight_total += weight;

                    let ni = (ny as usize * wu + nx as usize) * 4;
                    weighted_sum[0] += weight * input[ni];
                    weighted_sum[1] += weight * input[ni + 1];
                    weighted_sum[2] += weight * input[ni + 2];
                }
            }

            let oi = (y * wu + x) * 4;
            if weight_total > 0.0 {
                out[oi] = weighted_sum[0] / weight_total;
                out[oi + 1] = weighted_sum[1] / weight_total;
                out[oi + 2] = weighted_sum[2] / weight_total;
            }
            // alpha unchanged
        }
    }
    out
}

/// Single-Scale Retinex.
///
/// Per channel: `R = log(I + eps) - log(blur(I, sigma) + eps)`, then
/// normalize output to [0, 1].
pub fn retinex_ssr(input: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let npx = wu * hu;

    let mut out = input.to_vec();

    for c in 0..3 {
        // Extract single channel.
        let mut channel = vec![0.0f32; npx];
        for i in 0..npx {
            channel[i] = input[i * 4 + c];
        }

        let blurred = gaussian_blur_single(&channel, wu, hu, sigma);

        // Retinex: log(max(I, 1e-10)) - log(max(blur, 1e-10)).
        let mut retinex = vec![0.0f32; npx];
        for i in 0..npx {
            retinex[i] = channel[i].max(1e-10).ln() - blurred[i].max(1e-10).ln();
        }

        // Normalize to [0, 1].
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in &retinex {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let range = max_val - min_val;
        for i in 0..npx {
            out[i * 4 + c] = if range > 0.0 {
                (retinex[i] - min_val) / range
            } else {
                0.0
            };
        }
    }

    // Alpha preserved (already copied from input.to_vec()).
    out
}

/// Multi-Scale Retinex.
///
/// Average of 3 SSR at different sigma scales, then normalize to [0, 1].
pub fn retinex_msr(input: &[f32], w: u32, h: u32, sigma_s: f32, sigma_m: f32, sigma_l: f32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let npx = wu * hu;

    let mut out = input.to_vec();
    let sigmas = [sigma_s, sigma_m, sigma_l];

    for c in 0..3 {
        let mut channel = vec![0.0f32; npx];
        for i in 0..npx {
            channel[i] = input[i * 4 + c];
        }

        let mut msr = vec![0.0f32; npx];
        for &sigma in &sigmas {
            let blurred = gaussian_blur_single(&channel, wu, hu, sigma);
            for i in 0..npx {
                msr[i] += channel[i].max(1e-10).ln() - blurred[i].max(1e-10).ln();
            }
        }
        for v in msr.iter_mut() {
            *v /= sigmas.len() as f32;
        }

        // Normalize to [0, 1].
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in &msr {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let range = max_val - min_val;
        for i in 0..npx {
            out[i * 4 + c] = if range > 0.0 {
                (msr[i] - min_val) / range
            } else {
                0.0
            };
        }
    }

    out
}

/// Multi-Scale Retinex with Color Restoration (MSRCR).
///
/// MSR + color restoration gain: `gain = beta * ln(alpha * I_c / sum(I_channels))`.
/// Apply gain to MSR output, then normalize to [0, 1].
pub fn retinex_msrcr(
    input: &[f32],
    w: u32,
    h: u32,
    sigma_s: f32,
    sigma_m: f32,
    sigma_l: f32,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let npx = wu * hu;
    let eps = 1e-6f32;
    let sigmas = [sigma_s, sigma_m, sigma_l];

    // Compute MSR per channel (unnormalized).
    let mut msr_channels = vec![vec![0.0f32; npx]; 3];
    for c in 0..3 {
        let mut channel = vec![0.0f32; npx];
        for i in 0..npx {
            channel[i] = input[i * 4 + c];
        }
        for &sigma in &sigmas {
            let blurred = gaussian_blur_single(&channel, wu, hu, sigma);
            for i in 0..npx {
                msr_channels[c][i] += (channel[i] + eps).ln() - (blurred[i] + eps).ln();
            }
        }
        for v in msr_channels[c].iter_mut() {
            *v /= sigmas.len() as f32;
        }
    }

    // Color restoration.
    let mut out = input.to_vec();
    let mut restored = vec![vec![0.0f32; npx]; 3];

    for i in 0..npx {
        let idx = i * 4;
        let channel_sum = input[idx] + input[idx + 1] + input[idx + 2] + eps;
        for c in 0..3 {
            let cr = beta * (alpha * input[idx + c] / channel_sum).ln();
            restored[c][i] = cr * msr_channels[c][i];
        }
    }

    // Normalize each channel to [0, 1].
    for c in 0..3 {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in &restored[c] {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
        let range = max_val - min_val;
        for i in 0..npx {
            out[i * 4 + c] = if range > 0.0 && restored[c][i].is_finite() {
                ((restored[c][i] - min_val) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }

    out
}

/// High-frequency layer via frequency separation.
///
/// `out = (input - blur(input, sigma)) + 0.5`
pub fn frequency_high(input: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    let blurred = gaussian_blur(input, w, h, sigma);
    let mut out = input.to_vec();
    let npx = (w * h) as usize;

    for i in 0..npx {
        let idx = i * 4;
        for c in 0..3 {
            out[idx + c] = (input[idx + c] - blurred[idx + c]) + 0.5;
        }
        // alpha unchanged
    }
    out
}

/// Low-frequency layer via frequency separation.
///
/// `out = blur(input, sigma)`
pub fn frequency_low(input: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    gaussian_blur(input, w, h, sigma)
}

/// Laplacian pyramid detail remapping.
///
/// Build a Laplacian pyramid (downsample, upsample, subtract). Remap detail
/// coefficients: `d' = d * sigma / (sigma + |d|)`. Reconstruct from remapped
/// pyramid.
pub fn pyramid_detail_remap(input: &[f32], w: u32, h: u32, sigma: f32, levels: u32) -> Vec<f32> {
    // Downsample by 2x (box filter average).
    fn downsample(data: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
        let nw = (w + 1) / 2;
        let nh = (h + 1) / 2;
        let mut out = vec![0.0f32; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                let mut count = 0.0f32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let sx = x * 2 + dx as usize;
                        let sy = y * 2 + dy as usize;
                        if sx < w && sy < h {
                            let idx = (sy * w + sx) * 4;
                            for c in 0..4 {
                                acc[c] += data[idx + c];
                            }
                            count += 1.0;
                        }
                    }
                }
                let oidx = (y * nw + x) * 4;
                for c in 0..4 {
                    out[oidx + c] = acc[c] / count;
                }
            }
        }
        (out, nw, nh)
    }

    // Upsample by 2x (nearest neighbor) to target size.
    fn upsample(data: &[f32], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; tw * th * 4];
        for y in 0..th {
            for x in 0..tw {
                let sx = (x / 2).min(sw - 1);
                let sy = (y / 2).min(sh - 1);
                let si = (sy * sw + sx) * 4;
                let oi = (y * tw + x) * 4;
                out[oi] = data[si];
                out[oi + 1] = data[si + 1];
                out[oi + 2] = data[si + 2];
                out[oi + 3] = data[si + 3];
            }
        }
        out
    }

    if levels == 0 {
        return input.to_vec();
    }

    // Build Gaussian pyramid.
    let mut gauss_pyramid: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    gauss_pyramid.push((input.to_vec(), w as usize, h as usize));

    for _ in 0..levels {
        let (prev, pw, ph) = gauss_pyramid.last().unwrap();
        if *pw <= 1 && *ph <= 1 {
            break;
        }
        let (down, nw, nh) = downsample(prev, *pw, *ph);
        gauss_pyramid.push((down, nw, nh));
    }

    let actual_levels = gauss_pyramid.len() - 1;

    // Build Laplacian pyramid (detail layers).
    let mut laplacian: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    for l in 0..actual_levels {
        let (ref fine, fw, fh) = gauss_pyramid[l];
        let (ref coarse, cw, ch) = gauss_pyramid[l + 1];
        let upsampled = upsample(coarse, cw, ch, fw, fh);
        let npx = fw * fh;
        let mut detail = vec![0.0f32; npx * 4];
        for i in 0..npx * 4 {
            detail[i] = fine[i] - upsampled[i];
        }
        laplacian.push((detail, fw, fh));
    }

    // Remap detail coefficients: d' = d * sigma / (sigma + |d|).
    for (detail, dw, dh) in laplacian.iter_mut() {
        let npx = *dw * *dh;
        for i in 0..npx {
            let idx = i * 4;
            for c in 0..3 {
                let d = detail[idx + c];
                detail[idx + c] = d * sigma / (sigma + d.abs());
            }
            // alpha detail left as-is (will reconstruct properly)
        }
    }

    // Reconstruct from coarsest level up.
    let (mut current, mut cw, mut ch) = gauss_pyramid.last().unwrap().clone();

    for l in (0..actual_levels).rev() {
        let (ref detail, dw, dh) = laplacian[l];
        let upsampled = upsample(&current, cw, ch, dw, dh);
        let npx = dw * dh;
        current = vec![0.0f32; npx * 4];
        cw = dw;
        ch = dh;
        for i in 0..npx * 4 {
            current[i] = upsampled[i] + detail[i];
        }
    }

    // Restore original alpha.
    let npx = (w * h) as usize;
    for i in 0..npx {
        current[i * 4 + 3] = input[i * 4 + 3];
    }

    current
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Max absolute RGB difference between two buffers.
    fn max_rgb_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(pa, pb)| (0..3).map(move |c| (pa[c] - pb[c]).abs()))
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn vignette_center_pixel_unchanged() {
        // Center pixel of an odd-sized image should have factor very close to 1.0.
        let w = 33u32;
        let h = 33u32;
        let input = crate::solid(w, h, [0.8, 0.6, 0.4, 1.0]);
        let result = vignette(&input, w, h, 1.0, 0.0, 0.0);

        // Center pixel at (16, 16).
        let idx = (16 * w + 16) as usize * 4;
        let eps = 0.01;
        assert!(
            (result[idx] - 0.8).abs() < eps,
            "center R: got {}, expected 0.8",
            result[idx]
        );
        assert!(
            (result[idx + 1] - 0.6).abs() < eps,
            "center G: got {}, expected 0.6",
            result[idx + 1]
        );
        assert!(
            (result[idx + 2] - 0.4).abs() < eps,
            "center B: got {}, expected 0.4",
            result[idx + 2]
        );
        assert_eq!(result[idx + 3], 1.0, "alpha preserved");
    }

    #[test]
    fn vignette_powerlaw_strength_zero_is_identity() {
        let w = 16u32;
        let h = 16u32;
        let input = crate::noise(w, h, 42);
        let result = vignette_powerlaw(&input, w, h, 0.0, 2.0);
        assert_eq!(input, result);
    }

    #[test]
    fn clarity_amount_zero_is_identity() {
        let w = 16u32;
        let h = 16u32;
        let input = crate::noise(w, h, 7);
        let result = clarity(&input, w, h, 0.0, 3);
        let diff = max_rgb_diff(&input, &result);
        assert!(
            diff < 1e-5,
            "clarity with amount=0 should be identity, got diff {}",
            diff
        );
    }

    #[test]
    fn retinex_ssr_preserves_alpha() {
        let input = vec![
            0.2, 0.4, 0.6, 0.42,
            0.8, 0.5, 0.3, 0.99,
            0.1, 0.9, 0.5, 0.11,
            0.5, 0.5, 0.5, 0.77,
        ];
        let result = retinex_ssr(&input, 2, 2, 1.0);
        assert_eq!(result[3], 0.42);
        assert_eq!(result[7], 0.99);
        assert_eq!(result[11], 0.11);
        assert_eq!(result[15], 0.77);
    }

    #[test]
    fn frequency_low_plus_high_approx_original() {
        // low + (high - 0.5) should reconstruct the original.
        let w = 16u32;
        let h = 16u32;
        let sigma = 2.0;
        let input = crate::noise(w, h, 123);
        let low = frequency_low(&input, w, h, sigma);
        let high = frequency_high(&input, w, h, sigma);

        let npx = (w * h) as usize;
        let mut reconstructed = vec![0.0f32; npx * 4];
        for i in 0..npx {
            let idx = i * 4;
            for c in 0..3 {
                reconstructed[idx + c] = low[idx + c] + (high[idx + c] - 0.5);
            }
            reconstructed[idx + 3] = input[idx + 3];
        }

        let diff = max_rgb_diff(&input, &reconstructed);
        assert!(
            diff < 1e-4,
            "frequency_low + frequency_high should reconstruct original, got diff {}",
            diff
        );
    }

    #[test]
    fn dehaze_clear_image_approx_identity() {
        // An image where all channels are > 0.5 — very little haze.
        // Dehaze should not drastically change it.
        let w = 16u32;
        let h = 16u32;
        let mut input = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = 0.6 + 0.3 * (i as f32 / (w * h - 1) as f32);
            input.extend_from_slice(&[v, v, v, 1.0]);
        }
        let result = dehaze(&input, w, h, 2, 0.5, 0.1);

        // For a clear image the transmission should be high and result close to input.
        let diff = max_rgb_diff(&input, &result);
        assert!(
            diff < 0.3,
            "dehaze on clear image should be near identity, got diff {}",
            diff
        );
    }
}
