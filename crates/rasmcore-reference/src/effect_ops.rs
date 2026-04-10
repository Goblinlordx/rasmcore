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
/// shift = dist_from_center * strength / max_dist. Bilinear sampling.
/// Alpha is taken from the original pixel.
pub fn chromatic_aberration(input: &[f32], w: u32, h: u32, strength: f32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt().max(1e-10);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let shift = dist * strength / max_dist;

            // Normalize direction
            let (ndx, ndy) = if dist > 1e-10 {
                (dx / dist, dy / dist)
            } else {
                (0.0, 0.0)
            };

            // R shifts outward
            let rx = x as f32 + ndx * shift;
            let ry = y as f32 + ndy * shift;
            output[idx * 4] = bilinear_sample(input, w, h, rx, ry, 0);

            // G stays
            output[idx * 4 + 1] = input[idx * 4 + 1];

            // B shifts inward
            let bx = x as f32 - ndx * shift;
            let by = y as f32 - ndy * shift;
            output[idx * 4 + 2] = bilinear_sample(input, w, h, bx, by, 2);

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
pub fn oil_paint(input: &[f32], w: u32, h: u32, radius: u32, levels: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    let levels = levels.max(2);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let mut bin_count = vec![0u32; levels as usize];
            let mut bin_r = vec![0.0f32; levels as usize];
            let mut bin_g = vec![0.0f32; levels as usize];
            let mut bin_b = vec![0.0f32; levels as usize];

            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(h - 1);
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(w - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let ni = (ny * w + nx) as usize;
                    let r = input[ni * 4];
                    let g = input[ni * 4 + 1];
                    let b = input[ni * 4 + 2];
                    let luma = luminance(r, g, b).clamp(0.0, 1.0);
                    let bin = ((luma * (levels - 1) as f32) as u32).min(levels - 1) as usize;
                    bin_count[bin] += 1;
                    bin_r[bin] += r;
                    bin_g[bin] += g;
                    bin_b[bin] += b;
                }
            }

            // Find most frequent bin
            let mut best_bin = 0;
            let mut best_count = 0;
            for b in 0..levels as usize {
                if bin_count[b] > best_count {
                    best_count = bin_count[b];
                    best_bin = b;
                }
            }

            let c = best_count as f32;
            output[idx * 4] = bin_r[best_bin] / c;
            output[idx * 4 + 1] = bin_g[best_bin] / c;
            output[idx * 4 + 2] = bin_b[best_bin] / c;
            output[idx * 4 + 3] = input[idx * 4 + 3];
        }
    }
    output
}

// ─── 9. Charcoal ───────────────────────────────────────────────────────────

/// Charcoal effect: grayscale → Sobel edge detection → invert → Gaussian blur.
///
/// Alpha is preserved from the original image.
pub fn charcoal(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let n = (w * h) as usize;

    // Step 1: Grayscale
    let mut gray = vec![0.0f32; n];
    for i in 0..n {
        gray[i] = luminance(input[i * 4], input[i * 4 + 1], input[i * 4 + 2]);
    }

    // Step 2: Sobel edge detection
    let mut edges = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let g = |dx: i32, dy: i32| -> f32 {
                let px = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let py = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                gray[py * w as usize + px]
            };
            let gx = -g(-1, -1) + g(1, -1) - 2.0 * g(-1, 0) + 2.0 * g(1, 0) - g(-1, 1) + g(1, 1);
            let gy = -g(-1, -1) - 2.0 * g(0, -1) - g(1, -1) + g(-1, 1) + 2.0 * g(0, 1) + g(1, 1);
            let mag = (gx * gx + gy * gy).sqrt().min(1.0);
            edges[(y * w + x) as usize] = mag;
        }
    }

    // Step 3: Invert
    for v in edges.iter_mut() {
        *v = 1.0 - *v;
    }

    // Step 4: Gaussian blur on the inverted edges
    let sigma = radius as f32 / 2.0;
    let sigma = if sigma < 0.5 { 0.5 } else { sigma };
    let r = radius as i32;

    // Build 1D kernel
    let kernel_size = (2 * r + 1) as usize;
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0f32;
    for i in 0..kernel_size {
        let d = (i as i32 - r) as f32;
        let v = (-d * d / (2.0 * sigma * sigma)).exp();
        kernel[i] = v;
        sum += v;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let mut acc = 0.0f32;
            for k in 0..kernel_size {
                let sx = (x as i32 + k as i32 - r).clamp(0, w as i32 - 1) as usize;
                acc += edges[y * w as usize + sx] * kernel[k];
            }
            temp[y * w as usize + x] = acc;
        }
    }

    // Vertical pass
    let mut blurred = vec![0.0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let mut acc = 0.0f32;
            for k in 0..kernel_size {
                let sy = (y as i32 + k as i32 - r).clamp(0, h as i32 - 1) as usize;
                acc += temp[sy * w as usize + x] * kernel[k];
            }
            blurred[y * w as usize + x] = acc;
        }
    }

    // Step 5: Write output — grayscale result with original alpha
    let mut output = vec![0.0f32; n * 4];
    for i in 0..n {
        let v = blurred[i].clamp(0.0, 1.0);
        output[i * 4] = v;
        output[i * 4 + 1] = v;
        output[i * 4 + 2] = v;
        output[i * 4 + 3] = input[i * 4 + 3];
    }
    output
}

// ─── 10. Halftone ──────────────────────────────────────────────────────────

/// Halftone effect: divide image into blocks, compute block luminance,
/// draw a circle of radius proportional to luminance. Output is binary (0/1).
///
/// Alpha is preserved from the original image.
pub fn halftone(input: &[f32], w: u32, h: u32, dot_size: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut output = vec![0.0f32; n * 4];
    let ds = dot_size.max(1) as f32;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;

            // Block center
            let bx = (x as f32 / ds).floor() * ds + ds / 2.0;
            let by = (y as f32 / ds).floor() * ds + ds / 2.0;

            // Average luminance of the block
            let bx0 = ((bx - ds / 2.0) as u32).min(w - 1);
            let by0 = ((by - ds / 2.0) as u32).min(h - 1);
            let bx1 = ((bx + ds / 2.0) as u32).min(w);
            let by1 = ((by + ds / 2.0) as u32).min(h);

            let mut luma_sum = 0.0f32;
            let mut count = 0u32;
            for ny in by0..by1 {
                for nx in bx0..bx1 {
                    let ni = (ny * w + nx) as usize;
                    luma_sum += luminance(input[ni * 4], input[ni * 4 + 1], input[ni * 4 + 2]);
                    count += 1;
                }
            }
            let avg_luma = if count > 0 { luma_sum / count as f32 } else { 0.0 };

            // Darker areas → bigger dots (more ink). Radius proportional to (1 - luma).
            let max_radius = ds / 2.0;
            let dot_radius = max_radius * (1.0 - avg_luma).sqrt();

            // Distance from pixel to block center
            let dist = ((x as f32 - bx) * (x as f32 - bx) + (y as f32 - by) * (y as f32 - by)).sqrt();

            let v = if dist <= dot_radius { 0.0 } else { 1.0 };
            output[idx * 4] = v;
            output[idx * 4 + 1] = v;
            output[idx * 4 + 2] = v;
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
