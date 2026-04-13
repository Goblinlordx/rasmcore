//! Effect filter reference implementations — noise, grain, aberration, stylization.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Each function documents the formula and expected behavior.
//!
//! These are deliberately naive, unoptimized implementations — pure math,
//! no SIMD, no GPU, no external dependencies. Correct, not fast.

use core::f32::consts::PI;

// ─── Helper Functions ───────────────────────────────────────────────────────

/// Deterministic hash → [0, 1) float. Same algorithm as lib.rs.
#[inline]
fn hash_f32(mut x: u32) -> f32 {
    x = x.wrapping_mul(0x9e3779b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85ebca6b);
    x ^= x >> 13;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Luminance from linear RGB (Rec. 709 coefficients).
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Bilinear sample a single channel from the image.
/// Coordinates are in pixel space (0-based). Out-of-bounds clamps to edge.
#[inline]
fn bilinear_sample(input: &[f32], w: u32, h: u32, x: f32, y: f32, channel: usize) -> f32 {
    let wi = w as i32;
    let hi = h as i32;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sx0 = x0.clamp(0, wi - 1) as usize;
    let sx1 = x1.clamp(0, wi - 1) as usize;
    let sy0 = y0.clamp(0, hi - 1) as usize;
    let sy1 = y1.clamp(0, hi - 1) as usize;

    let stride = w as usize;

    let p00 = input[(sy0 * stride + sx0) * 4 + channel];
    let p10 = input[(sy0 * stride + sx1) * 4 + channel];
    let p01 = input[(sy1 * stride + sx0) * 4 + channel];
    let p11 = input[(sy1 * stride + sx1) * 4 + channel];

    let top = p00 + fx * (p10 - p00);
    let bot = p01 + fx * (p11 - p01);
    top + fy * (bot - top)
}

// ─── 1. Gaussian Noise ─────────────────────────────────────────────────────

/// Add Gaussian noise to each pixel via Box-Muller transform.
///
/// For each pixel, two uniform hash values are converted to a Gaussian
/// sample with mean=0, stddev=amount. Alpha is preserved.
pub fn gaussian_noise(input: &[f32], w: u32, h: u32, amount: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();
    for i in 0..n {
        let base = seed.wrapping_mul(3).wrapping_add(i as u32 * 2);
        let u1 = hash_f32(base).max(1e-10); // avoid log(0)
        let u2 = hash_f32(base.wrapping_add(1));
        let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let noise = gauss * amount;
        output[i * 4] = (input[i * 4] + noise).clamp(0.0, 1.0);
        output[i * 4 + 1] = (input[i * 4 + 1] + noise).clamp(0.0, 1.0);
        output[i * 4 + 2] = (input[i * 4 + 2] + noise).clamp(0.0, 1.0);
        // alpha preserved
    }
    output
}

// ─── 2. Uniform Noise ──────────────────────────────────────────────────────

/// Add uniform noise in [-amount, +amount] to each pixel.
///
/// Each channel gets an independent noise value. Alpha is preserved.
pub fn uniform_noise(input: &[f32], w: u32, h: u32, amount: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();
    for i in 0..n {
        for c in 0..3 {
            let h = hash_f32(seed.wrapping_add(i as u32 * 3 + c as u32));
            let noise = (h * 2.0 - 1.0) * amount;
            output[i * 4 + c] = (input[i * 4 + c] + noise).clamp(0.0, 1.0);
        }
    }
    output
}

// ─── 3. Salt & Pepper Noise ────────────────────────────────────────────────

/// Salt-and-pepper noise: randomly set pixels to black or white.
///
/// For each pixel, if hash < density/2 → black (0,0,0), if hash > 1-density/2 → white (1,1,1).
/// Alpha is preserved.
pub fn salt_pepper_noise(input: &[f32], w: u32, h: u32, density: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();
    let half = density / 2.0;
    for i in 0..n {
        let h = hash_f32(seed.wrapping_add(i as u32));
        if h < half {
            output[i * 4] = 0.0;
            output[i * 4 + 1] = 0.0;
            output[i * 4 + 2] = 0.0;
        } else if h > 1.0 - half {
            output[i * 4] = 1.0;
            output[i * 4 + 1] = 1.0;
            output[i * 4 + 2] = 1.0;
        }
    }
    output
}

// ─── 4. Poisson Noise ──────────────────────────────────────────────────────

/// Approximate Poisson (shot) noise: noise proportional to sqrt(pixel_value).
///
/// For each channel: output = value + gaussian * sqrt(value) * scale.
/// Alpha is preserved.
pub fn poisson_noise(input: &[f32], w: u32, h: u32, scale: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();
    for i in 0..n {
        for c in 0..3 {
            let base = seed.wrapping_mul(7).wrapping_add((i as u32 * 3 + c as u32) * 2);
            let u1 = hash_f32(base).max(1e-10);
            let u2 = hash_f32(base.wrapping_add(1));
            let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let val = input[i * 4 + c].max(0.0);
            let noise = gauss * val.sqrt() * scale;
            output[i * 4 + c] = (val + noise).clamp(0.0, 1.0);
        }
    }
    output
}

// ─── 5. Film Grain ─────────────────────────────────────────────────────────

/// Film grain: Gaussian noise weighted by luminance.
///
/// More grain in midtones, less in shadows/highlights.
/// weight = 4 * luma * (1 - luma). Alpha is preserved.
pub fn film_grain(input: &[f32], w: u32, h: u32, amount: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();
    for i in 0..n {
        let r = input[i * 4];
        let g = input[i * 4 + 1];
        let b = input[i * 4 + 2];
        let luma = luminance(r, g, b).clamp(0.0, 1.0);
        let weight = 4.0 * luma * (1.0 - luma);

        let base = seed.wrapping_mul(5).wrapping_add(i as u32 * 2);
        let u1 = hash_f32(base).max(1e-10);
        let u2 = hash_f32(base.wrapping_add(1));
        let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let noise = gauss * amount * weight;

        output[i * 4] = (r + noise).clamp(0.0, 1.0);
        output[i * 4 + 1] = (g + noise).clamp(0.0, 1.0);
        output[i * 4 + 2] = (b + noise).clamp(0.0, 1.0);
    }
    output
}

// ─── 6. Chromatic Aberration ───────────────────────────────────────────────

/// Chromatic aberration: R shifts outward from center, B inward, G stays.
///
/// shift = dist * strength / max_dist. Nearest-neighbor (round) with BORDER_REFLECT_101.
/// Alpha is taken from the original pixel.
pub fn chromatic_aberration(input: &[f32], w: u32, h: u32, strength: f32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
    let norm_strength = strength / max_dist;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let shift = dist * norm_strength;
            let d = dist.max(1.0);
            let nx = dx / d;
            let ny = dy / d;

            // R shifts outward — bilinear sub-pixel sampling
            output[idx * 4] = bilinear_sample(input, w, h,
                x as f32 + nx * shift, y as f32 + ny * shift, 0);

            // G stays
            output[idx * 4 + 1] = input[idx * 4 + 1];

            // B shifts inward — bilinear sub-pixel sampling
            output[idx * 4 + 2] = bilinear_sample(input, w, h,
                x as f32 - nx * shift, y as f32 - ny * shift, 2);

            // Alpha preserved
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── 7. Chromatic Split ────────────────────────────────────────────────────

/// Translate each RGB channel independently by (dx, dy). Bilinear sample.
///
/// Alpha is taken from the original pixel.
pub fn chromatic_split(
    input: &[f32],
    w: u32,
    h: u32,
    red_dx: f32,
    red_dy: f32,
    green_dx: f32,
    green_dy: f32,
    blue_dx: f32,
    blue_dy: f32,
) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let fx = x as f32;
            let fy = y as f32;

            output[idx * 4] = bilinear_sample(input, w, h, fx + red_dx, fy + red_dy, 0);
            output[idx * 4 + 1] = bilinear_sample(input, w, h, fx + green_dx, fy + green_dy, 1);
            output[idx * 4 + 2] = bilinear_sample(input, w, h, fx + blue_dx, fy + blue_dy, 2);
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── 8. Oil Paint ──────────────────────────────────────────────────────────

/// Oil paint effect: quantize neighborhood to intensity bins, find most
/// frequent bin, output average color of that bin.
///
/// Alpha is preserved from the center pixel.
pub fn oil_paint(input: &[f32], w: u32, h: u32, radius: u32, _levels: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let wu = w as usize;
    let hu = h as usize;
    let mut output = vec![0.0f32; n * 4];
    let r = radius as i64;
    let kw = (2 * r + 1) as usize;
    const BINS: usize = 256;

    for y in 0..h {
        for x in 0..w {
            let oidx = (y * w + x) as usize;
            let mut count = [0u32; BINS];
            let mut max_count = 0u32;
            let mut best_pixel = oidx * 4;

            // Iterate full kernel, clamp to edge (matching IM virtual pixels)
            for v in 0..kw {
                for u in 0..kw {
                    let ny = (y as i64 + v as i64 - r).clamp(0, hu as i64 - 1) as usize;
                    let nx = (x as i64 + u as i64 - r).clamp(0, wu as i64 - 1) as usize;
                    let ni = (ny * wu + nx) * 4;
                    // IM Rec709Luma: direct weighted sum (no gamma since IM sees TIFF as sRGB)
                    let intensity = 0.212656 * input[ni]
                        + 0.715158 * input[ni + 1]
                        + 0.072186 * input[ni + 2];
                    // Round-to-nearest (matching IM's ScaleQuantumToChar)
                    let bin = ((intensity.max(0.0) * 255.0 + 0.5) as usize).min(BINS - 1);
                    count[bin] += 1;
                    if count[bin] > max_count {
                        max_count = count[bin];
                        best_pixel = ni;
                    }
                }
            }

            output[oidx * 4] = input[best_pixel];
            output[oidx * 4 + 1] = input[best_pixel + 1];
            output[oidx * 4 + 2] = input[best_pixel + 2];
            output[oidx * 4 + 3] = input[oidx * 4 + 3];
        }
    }
    output
}

// ─── 9. Charcoal ───────────────────────────────────────────────────────────

/// Charcoal effect: grayscale → Sobel edge detection → clip(1.0) → expand RGB →
/// Gaussian blur → invert.
///
/// Pipeline order matches golden: Sobel→clip→expand RGB→blur(sigma)→invert.
/// Kernel size: ksize = (round(sigma*10+1) | 1).max(3). BORDER_REFLECT_101.
/// Alpha is preserved from the original image.
pub fn charcoal(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    // radius param carries the sigma value (passed as u32 truncation of f32)
    let sigma = radius as f32;
    let sigma = if sigma < 0.5 { 0.5 } else { sigma };
    let n = (w * h) as usize;

    // Step 1: Grayscale
    let mut gray = vec![0.0f32; n];
    for i in 0..n {
        gray[i] = luminance(input[i * 4], input[i * 4 + 1], input[i * 4 + 2]);
    }

    // Step 2: Sobel edge detection with BORDER_REFLECT_101
    let reflect_101 = |v: i32, max: i32| -> usize {
        let mut v = v;
        let m = max - 1;
        if m == 0 { return 0; }
        let period = 2 * m;
        v = v.rem_euclid(period);
        if v > m { v = period - v; }
        v as usize
    };

    let mut edges = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let g = |dx: i32, dy: i32| -> f32 {
                let px = reflect_101(x as i32 + dx, w as i32);
                let py = reflect_101(y as i32 + dy, h as i32);
                gray[py * w as usize + px]
            };
            let gx = -g(-1, -1) + g(1, -1) - 2.0 * g(-1, 0) + 2.0 * g(1, 0) - g(-1, 1) + g(1, 1);
            let gy = -g(-1, -1) - 2.0 * g(0, -1) - g(1, -1) + g(-1, 1) + 2.0 * g(0, 1) + g(1, 1);
            let mag = (gx * gx + gy * gy).sqrt().min(1.0);
            edges[(y * w + x) as usize] = mag;
        }
    }

    // Step 3: Gaussian blur on the edge magnitude — ksize matches golden formula
    // ksize = int(round(sigma * 10.0 + 1.0)) | 1, min 3
    let ksize_raw = (sigma * 10.0 + 1.0).round() as i32;
    let ksize_odd = ksize_raw | 1;
    let ksize = ksize_odd.max(3) as usize;
    let r = (ksize as i32 - 1) / 2;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f32; ksize];
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let d = (i as i32 - r) as f32;
        let v = (-d * d / (2.0 * sigma * sigma)).exp();
        kernel[i] = v;
        sum += v;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    // Horizontal pass (BORDER_REFLECT_101)
    let mut temp = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let mut acc = 0.0f32;
            for k in 0..ksize {
                let sx = reflect_101(x as i32 + k as i32 - r, w as i32);
                acc += edges[y * w as usize + sx] * kernel[k];
            }
            temp[y * w as usize + x] = acc;
        }
    }

    // Vertical pass (BORDER_REFLECT_101)
    let mut blurred = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let mut acc = 0.0f32;
            for k in 0..ksize {
                let sy = reflect_101(y as i32 + k as i32 - r, h as i32);
                acc += temp[sy * w as usize + x] * kernel[k];
            }
            blurred[y * w as usize + x] = acc;
        }
    }

    // Step 4: Invert and write output — grayscale result with original alpha
    let mut output = vec![0.0f32; n * 4];
    for i in 0..n {
        let v = (1.0 - blurred[i]).clamp(0.0, 1.0);
        output[i * 4] = v;
        output[i * 4 + 1] = v;
        output[i * 4 + 2] = v;
        output[i * 4 + 3] = input[i * 4 + 3];
    }
    output
}

// ─── 10. Halftone ──────────────────────────────────────────────────────────

/// Halftone effect: CMYK sine-wave screening.
///
/// RGB → proper CMYK (with K channel); per-channel rotated sin*sin screen
/// at angles C=15°, M=75°, Y=0°, K=45°; freq = π/dot_size;
/// threshold binary; CMYK → RGB recombine.
///
/// Alpha is preserved from the original image.
pub fn halftone(input: &[f32], w: u32, h: u32, dot_size: u32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let n = wu * hu;
    let mut output = vec![0.0f32; n * 4];
    let max_r = dot_size.max(1) as f32;
    let cell = 2.0 * max_r;

    let angles: [f32; 4] = [
        15.0f32.to_radians(),
        75.0f32.to_radians(),
        0.0f32.to_radians(),
        45.0f32.to_radians(),
    ];

    let smooth = |edge0: f32, edge1: f32, x: f32| -> f32 {
        if edge0 >= edge1 { return if x <= edge1 { 1.0 } else { 0.0 }; }
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    };

    for y in 0..hu {
        for x in 0..wu {
            let idx = y * wu + x;
            let mut ink = [0.0f32; 4];

            for (ch, &angle) in angles.iter().enumerate() {
                let (sin_a, cos_a) = angle.sin_cos();
                let sx = x as f32 * cos_a + y as f32 * sin_a;
                let sy = -(x as f32) * sin_a + y as f32 * cos_a;
                let cx = (sx / cell).round() * cell;
                let cy = (sy / cell).round() * cell;

                // Map cell center back to image space
                let img_x = (cx * cos_a + cy * sin_a).clamp(0.0, (wu - 1) as f32) as usize;
                let img_y = (-(cx * (-sin_a)) + cy * cos_a).clamp(0.0, (hu - 1) as f32) as usize;
                let si = (img_y * wu + img_x) * 4;
                let sr = input[si];
                let sg = input[si + 1];
                let sb = input[si + 2];
                let sk = 1.0 - sr.max(sg).max(sb);
                let s_inv_k = if sk < 1.0 { 1.0 / (1.0 - sk) } else { 0.0 };
                let density = match ch {
                    0 => (1.0 - sr - sk) * s_inv_k,
                    1 => (1.0 - sg - sk) * s_inv_k,
                    2 => (1.0 - sb - sk) * s_inv_k,
                    _ => sk,
                };

                let dot_r = max_r * density.max(0.0).sqrt();
                let dist = ((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)).sqrt();
                ink[ch] = smooth(dot_r + 0.5, dot_r - 0.5, dist);
            }

            output[idx * 4] = (1.0 - ink[0]) * (1.0 - ink[3]);
            output[idx * 4 + 1] = (1.0 - ink[1]) * (1.0 - ink[3]);
            output[idx * 4 + 2] = (1.0 - ink[2]) * (1.0 - ink[3]);
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── 11. Glitch ────────────────────────────────────────────────────────────

/// Glitch effect: random horizontal row shifts.
///
/// For each row, if hash < amount, shift the row by a random dx.
/// Deterministic based on seed. Alpha is preserved.
pub fn glitch(input: &[f32], w: u32, h: u32, amount: f32, seed: u32) -> Vec<f32> {
    let mut output = input.to_vec();
    for y in 0..h {
        let row_hash = hash_f32(seed.wrapping_add(y.wrapping_mul(0x45d9f3b)));
        if row_hash < amount {
            // Compute shift amount: map another hash to [-w/2, w/2]
            let shift_hash = hash_f32(seed.wrapping_add(y.wrapping_mul(0x6c62272e)));
            let dx = ((shift_hash * 2.0 - 1.0) * w as f32 * 0.5) as i32;

            for x in 0..w {
                let src_x = ((x as i32 + dx).rem_euclid(w as i32)) as usize;
                let dst_idx = (y * w + x) as usize;
                let src_idx = y as usize * w as usize + src_x;
                output[dst_idx * 4] = input[src_idx * 4];
                output[dst_idx * 4 + 1] = input[src_idx * 4 + 1];
                output[dst_idx * 4 + 2] = input[src_idx * 4 + 2];
                // alpha preserved from source row
                output[dst_idx * 4 + 3] = input[src_idx * 4 + 3];
            }
        }
    }
    output
}

// ─── 12. Light Leak ────────────────────────────────────────────────────────

/// Light leak: additive warm color gradient from a deterministic corner.
///
/// color = (1.0, 0.8, 0.4) * intensity * radial_falloff from chosen corner.
/// Alpha is preserved.
pub fn light_leak(input: &[f32], w: u32, h: u32, intensity: f32, seed: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = input.to_vec();

    // Pick corner based on seed
    let corner = (hash_f32(seed) * 4.0) as u32 % 4;
    let (cx, cy) = match corner {
        0 => (0.0f32, 0.0f32),
        1 => (w as f32 - 1.0, 0.0),
        2 => (0.0, h as f32 - 1.0),
        _ => (w as f32 - 1.0, h as f32 - 1.0),
    };

    let max_dist = ((w as f32 - 1.0).powi(2) + (h as f32 - 1.0).powi(2)).sqrt().max(1e-10);

    for i in 0..n {
        let x = (i % w as usize) as f32;
        let y = (i / w as usize) as f32;
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        let falloff = 1.0 - (dist / max_dist);
        let falloff = falloff * falloff; // quadratic falloff

        let leak = intensity * falloff;
        output[i * 4] = (output[i * 4] + 1.0 * leak).clamp(0.0, 1.0);
        output[i * 4 + 1] = (output[i * 4 + 1] + 0.8 * leak).clamp(0.0, 1.0);
        output[i * 4 + 2] = (output[i * 4 + 2] + 0.4 * leak).clamp(0.0, 1.0);
    }
    output
}

// ─── 13. Mirror Kaleidoscope ───────────────────────────────────────────────

/// Mirror kaleidoscope: reflect coordinates through segment boundaries.
///
/// Maps each pixel's angle to the first segment, mirroring alternating segments.
/// Alpha is preserved.
pub fn mirror_kaleidoscope(input: &[f32], w: u32, h: u32, segments: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    let segments = segments.max(1);
    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;
    let segment_angle = 2.0 * PI / segments as f32;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let mut angle = dy.atan2(dx); // [-PI, PI]
            if angle < 0.0 {
                angle += 2.0 * PI; // [0, 2*PI]
            }

            // Which segment?
            let seg = (angle / segment_angle) as u32;
            let local_angle = angle - seg as f32 * segment_angle;

            // Mirror alternating segments
            let mapped_angle = if seg % 2 == 0 {
                local_angle
            } else {
                segment_angle - local_angle
            };

            // Convert back to cartesian
            let sx = cx + dist * mapped_angle.cos();
            let sy = cy + dist * mapped_angle.sin();

            // Bilinear sample each channel
            for c in 0..3 {
                output[idx * 4 + c] = bilinear_sample(input, w, h, sx, sy, c);
            }
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── 14. Sparse Color ──────────────────────────────────────────────────────

/// Inverse distance weighting interpolation from sparse control points.
///
/// Each point is (x, y, [r, g, b]). weight = 1/dist^2.
/// If a pixel is exactly on a control point, that point's color is used.
/// Alpha is preserved from the original image.
pub fn sparse_color(
    input: &[f32],
    w: u32,
    h: u32,
    points: &[(f32, f32, [f32; 3])],
) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];

    if points.is_empty() {
        return input.to_vec();
    }

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let fx = x as f32;
            let fy = y as f32;

            let mut total_weight = 0.0f32;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut exact = false;

            for &(px, py, color) in points {
                let dx = fx - px;
                let dy = fy - py;
                let dist2 = dx * dx + dy * dy;

                if dist2 < 1e-10 {
                    r = color[0];
                    g = color[1];
                    b = color[2];
                    exact = true;
                    break;
                }

                let w = 1.0 / dist2;
                total_weight += w;
                r += w * color[0];
                g += w * color[1];
                b += w * color[2];
            }

            if !exact && total_weight > 0.0 {
                r /= total_weight;
                g /= total_weight;
                b /= total_weight;
            }

            output[idx * 4] = r.clamp(0.0, 1.0);
            output[idx * 4 + 1] = g.clamp(0.0, 1.0);
            output[idx * 4 + 2] = b.clamp(0.0, 1.0);
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, r: f32, g: f32, b: f32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
        pixels
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn gaussian_noise_amount_zero_identity() {
        let img = solid_image(16, 16, 0.5, 0.3, 0.7);
        let result = gaussian_noise(&img, 16, 16, 0.0, 42);
        assert!(max_diff(&img, &result) < 1e-6, "amount=0 should be identity");
    }

    #[test]
    fn chromatic_aberration_strength_zero_identity() {
        let img = solid_image(16, 16, 0.5, 0.3, 0.7);
        let result = chromatic_aberration(&img, 16, 16, 0.0);
        assert!(max_diff(&img, &result) < 1e-6, "strength=0 should be identity");
    }

    #[test]
    fn oil_paint_solid_color_identity() {
        let img = solid_image(16, 16, 0.5, 0.3, 0.7);
        let result = oil_paint(&img, 16, 16, 3, 8);
        assert!(
            max_diff(&img, &result) < 1e-5,
            "solid color input should produce same solid output"
        );
    }

    #[test]
    fn glitch_amount_zero_identity() {
        let img = solid_image(16, 16, 0.5, 0.3, 0.7);
        let result = glitch(&img, 16, 16, 0.0, 42);
        assert!(max_diff(&img, &result) < 1e-6, "amount=0 should be identity");
    }

    #[test]
    fn gaussian_noise_deterministic() {
        let img = solid_image(8, 8, 0.5, 0.5, 0.5);
        let a = gaussian_noise(&img, 8, 8, 0.1, 123);
        let b = gaussian_noise(&img, 8, 8, 0.1, 123);
        assert_eq!(a, b, "same seed must produce identical output");
    }

    #[test]
    fn uniform_noise_deterministic() {
        let img = solid_image(8, 8, 0.5, 0.5, 0.5);
        let a = uniform_noise(&img, 8, 8, 0.1, 123);
        let b = uniform_noise(&img, 8, 8, 0.1, 123);
        assert_eq!(a, b, "same seed must produce identical output");
    }

    #[test]
    fn salt_pepper_density_zero_identity() {
        let img = solid_image(16, 16, 0.5, 0.3, 0.7);
        let result = salt_pepper_noise(&img, 16, 16, 0.0, 42);
        assert!(max_diff(&img, &result) < 1e-6, "density=0 should be identity");
    }

    #[test]
    fn sparse_color_no_points_identity() {
        let img = solid_image(8, 8, 0.5, 0.3, 0.7);
        let result = sparse_color(&img, 8, 8, &[]);
        assert!(max_diff(&img, &result) < 1e-6, "no points should return input");
    }
}
