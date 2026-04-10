//! Advanced spatial filter reference implementations — motion blur, lens blur,
//! tilt-shift, adaptive threshold, Scharr edge detection, smart sharpen.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Each function documents the formula and expected behavior.
//!
//! These are deliberately naive, unoptimized implementations — pure math,
//! no SIMD, no GPU, no external dependencies. Correct, not fast.

// ─── Helper Functions ───────────────────────────────────────────────────────

/// Bilinear sample a single channel from the image.
/// Coordinates are in pixel space (0-based). Out-of-bounds clamps to edge.
#[inline]
fn bilinear_sample(input: &[f32], w: u32, h: u32, x: f32, y: f32, channel: usize) -> f32 {
    let w = w as i32;
    let h = h as i32;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sx0 = x0.clamp(0, w - 1) as usize;
    let sx1 = x1.clamp(0, w - 1) as usize;
    let sy0 = y0.clamp(0, h - 1) as usize;
    let sy1 = y1.clamp(0, h - 1) as usize;

    let stride = w as usize;

    let p00 = input[(sy0 * stride + sx0) * 4 + channel];
    let p10 = input[(sy0 * stride + sx1) * 4 + channel];
    let p01 = input[(sy1 * stride + sx0) * 4 + channel];
    let p11 = input[(sy1 * stride + sx1) * 4 + channel];

    let top = p00 + fx * (p10 - p00);
    let bot = p01 + fx * (p11 - p01);
    top + fy * (bot - top)
}

/// Luminance from linear RGB (Rec. 709 coefficients).
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Smoothstep interpolation. Returns 0 if x <= edge0, 1 if x >= edge1,
/// smooth Hermite interpolation in between.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 >= edge1 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Simple separable Gaussian blur for internal use (tilt-shift pre-blur).
/// Uses the same formula as spatial_ops::gaussian_blur but reimplemented here
/// to avoid cross-module dependency.
fn gaussian_blur_internal(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let sigma = radius as f32 / 3.0;
    let r = radius as i32;
    let size = (2 * radius + 1) as usize;

    // Build 1D kernel
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for i in 0..size {
        let d = i as f32 - radius as f32;
        let val = (-0.5 * d * d / (sigma * sigma)).exp();
        kernel[i] = val;
        sum += val;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    let wi = w as i32;
    let hi = h as i32;
    let npix = (w * h) as usize;

    // Horizontal pass
    let mut temp = vec![0.0f32; npix * 4];
    for y in 0..hi {
        for x in 0..wi {
            let mut acc = [0.0f32; 3];
            for k in 0..size as i32 {
                let sx = (x + k - r).clamp(0, wi - 1);
                let si = (y * wi + sx) as usize * 4;
                let kw = kernel[k as usize];
                acc[0] += input[si] * kw;
                acc[1] += input[si + 1] * kw;
                acc[2] += input[si + 2] * kw;
            }
            let di = (y * wi + x) as usize * 4;
            temp[di] = acc[0];
            temp[di + 1] = acc[1];
            temp[di + 2] = acc[2];
            temp[di + 3] = input[di + 3]; // preserve alpha
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; npix * 4];
    for y in 0..hi {
        for x in 0..wi {
            let mut acc = [0.0f32; 3];
            for k in 0..size as i32 {
                let sy = (y + k - r).clamp(0, hi - 1);
                let si = (sy * wi + x) as usize * 4;
                let kw = kernel[k as usize];
                acc[0] += temp[si] * kw;
                acc[1] += temp[si + 1] * kw;
                acc[2] += temp[si + 2] * kw;
            }
            let di = (y * wi + x) as usize * 4;
            out[di] = acc[0];
            out[di + 1] = acc[1];
            out[di + 2] = acc[2];
            out[di + 3] = input[di + 3]; // preserve alpha
        }
    }

    out
}

// ─── Filter Implementations ─────────────────────────────────────────────────

/// Motion blur — directional blur along an angle.
///
/// For each pixel, samples `length` points along the direction vector
/// `(cos(angle), sin(angle))`, centered on the pixel. Uses bilinear sampling
/// at sub-pixel positions. Edge pixels clamp.
///
/// For `length=0`, returns a copy of the input (identity).
pub fn motion_blur(input: &[f32], w: u32, h: u32, angle_deg: f32, length: u32) -> Vec<f32> {
    if length == 0 {
        return input.to_vec();
    }

    let angle_rad = angle_deg * core::f32::consts::PI / 180.0;
    let dx = angle_rad.cos();
    let dy = angle_rad.sin();
    let n = length as usize;
    let half = n as f32 / 2.0;

    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for s in 0..=n {
                let t = s as f32 - half;
                let sx = x as f32 + t * dx;
                let sy = y as f32 + t * dy;
                for c in 0..3 {
                    acc[c] += bilinear_sample(input, w, h, sx, sy, c);
                }
            }
            let count = (n + 1) as f32;
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0] / count;
            out[di + 1] = acc[1] / count;
            out[di + 2] = acc[2] / count;
            // alpha unchanged
        }
    }
    out
}

/// Zoom blur — radial blur toward/away from center.
///
/// For each pixel, computes direction toward center point, then samples
/// `n = clamp(dist * |factor|, 3, 64)` points along that direction. Averages
/// all samples.
///
/// `factor=0` returns identity (no samples along direction).
pub fn zoom_blur(
    input: &[f32],
    w: u32,
    h: u32,
    center_x: f32,
    center_y: f32,
    factor: f32,
) -> Vec<f32> {
    if factor == 0.0 {
        return input.to_vec();
    }

    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let px = x as f32;
            let py = y as f32;
            let dir_x = center_x - px;
            let dir_y = center_y - py;
            let dist = (dir_x * dir_x + dir_y * dir_y).sqrt();

            let n = (dist * factor.abs()).clamp(3.0, 64.0) as u32;

            let mut acc = [0.0f32; 3];
            for s in 0..=n {
                let t = if n == 0 {
                    0.0
                } else {
                    s as f32 / n as f32
                };
                // Sample from pixel toward center, scaled by factor
                let sx = px + dir_x * t * factor.signum() * (factor.abs().min(1.0));
                let sy = py + dir_y * t * factor.signum() * (factor.abs().min(1.0));
                for c in 0..3 {
                    acc[c] += bilinear_sample(input, w, h, sx, sy, c);
                }
            }
            let count = (n + 1) as f32;
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0] / count;
            out[di + 1] = acc[1] / count;
            out[di + 2] = acc[2] / count;
        }
    }
    out
}

/// Spin blur — rotational blur around a center point.
///
/// For each pixel, rotates around center from `-angle/2` to `+angle/2`.
/// Number of samples: `n = clamp(angle_rad * radius, 3, 64)` where radius
/// is the distance from the pixel to center.
///
/// `angle_deg=0` returns identity.
pub fn spin_blur(
    input: &[f32],
    w: u32,
    h: u32,
    center_x: f32,
    center_y: f32,
    angle_deg: f32,
) -> Vec<f32> {
    if angle_deg == 0.0 {
        return input.to_vec();
    }

    let angle_rad = angle_deg * core::f32::consts::PI / 180.0;
    let half_angle = angle_rad / 2.0;

    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let px = x as f32 - center_x;
            let py = y as f32 - center_y;
            let radius = (px * px + py * py).sqrt();

            let n = (angle_rad.abs() * radius).clamp(3.0, 64.0) as u32;

            let base_angle = py.atan2(px);
            let mut acc = [0.0f32; 3];

            for s in 0..=n {
                let t = if n == 0 {
                    0.0
                } else {
                    s as f32 / n as f32
                };
                let theta = base_angle + (-half_angle + angle_rad * t);
                let sx = center_x + radius * theta.cos();
                let sy = center_y + radius * theta.sin();
                for c in 0..3 {
                    acc[c] += bilinear_sample(input, w, h, sx, sy, c);
                }
            }

            let count = (n + 1) as f32;
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0] / count;
            out[di + 1] = acc[1] / count;
            out[di + 2] = acc[2] / count;
        }
    }
    out
}

/// Lens blur — convolve with disc (circle) kernel.
///
/// Builds a binary disc kernel: for each (dx, dy) in `[-r, r]²`, include if
/// `dx² + dy² <= r²`. Normalizes by the count of included pixels.
///
/// For `radius=0`, returns a copy of the input (identity).
pub fn lens_blur(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }

    let r = radius as i32;
    let r_sq = (r * r) as f32;

    // Build disc kernel
    let mut kernel_offsets: Vec<(i32, i32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r_sq {
                kernel_offsets.push((dx, dy));
            }
        }
    }
    let count = kernel_offsets.len() as f32;

    let wi = w as i32;
    let hi = h as i32;
    let mut out = input.to_vec();

    for y in 0..hi {
        for x in 0..wi {
            let mut acc = [0.0f32; 3];
            for &(dx, dy) in &kernel_offsets {
                let sx = (x + dx).clamp(0, wi - 1) as usize;
                let sy = (y + dy).clamp(0, hi - 1) as usize;
                let si = (sy * w as usize + sx) * 4;
                acc[0] += input[si];
                acc[1] += input[si + 1];
                acc[2] += input[si + 2];
            }
            let di = (y * wi + x) as usize * 4;
            out[di] = acc[0] / count;
            out[di + 1] = acc[1] / count;
            out[di + 2] = acc[2] / count;
        }
    }
    out
}

/// Bokeh blur — shaped kernel blur (hexagonal or disc).
///
/// If `hexagonal` is true, uses a 6-point regular hexagon kernel inscribed
/// in the given radius. If false, falls back to disc kernel (same as lens_blur).
///
/// For `radius=0`, returns a copy of the input (identity).
pub fn bokeh_blur(input: &[f32], w: u32, h: u32, radius: u32, hexagonal: bool) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }

    let r = radius as i32;
    let rf = radius as f32;

    // Build kernel offsets based on shape
    let mut kernel_offsets: Vec<(i32, i32)> = Vec::new();

    if hexagonal {
        // Regular hexagon test: a point (dx, dy) is inside a regular hexagon
        // of circumradius r if it satisfies the intersection of 3 pairs of
        // half-planes. For a flat-top hexagon:
        //   |dy| <= r * sqrt(3)/2
        //   |dy| + |dx| * sqrt(3) <= r * sqrt(3)
        let sqrt3 = 3.0f32.sqrt();
        let half_h = rf * sqrt3 / 2.0;

        for dy in -r..=r {
            for dx in -r..=r {
                let adx = (dx as f32).abs();
                let ady = (dy as f32).abs();
                if ady <= half_h && (ady + adx * sqrt3) <= rf * sqrt3 {
                    kernel_offsets.push((dx, dy));
                }
            }
        }
    } else {
        // Disc kernel — same as lens_blur
        let r_sq = (r * r) as f32;
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx * dx + dy * dy) as f32 <= r_sq {
                    kernel_offsets.push((dx, dy));
                }
            }
        }
    }

    let count = kernel_offsets.len() as f32;
    let wi = w as i32;
    let hi = h as i32;
    let mut out = input.to_vec();

    for y in 0..hi {
        for x in 0..wi {
            let mut acc = [0.0f32; 3];
            for &(dx, dy) in &kernel_offsets {
                let sx = (x + dx).clamp(0, wi - 1) as usize;
                let sy = (y + dy).clamp(0, hi - 1) as usize;
                let si = (sy * w as usize + sx) * 4;
                acc[0] += input[si];
                acc[1] += input[si + 1];
                acc[2] += input[si + 2];
            }
            let di = (y * wi + x) as usize * 4;
            out[di] = acc[0] / count;
            out[di + 1] = acc[1] / count;
            out[di + 2] = acc[2] / count;
        }
    }
    out
}

/// Tilt-shift blur — selective focus with a focused band and blurred regions.
///
/// 1. Pre-blurs the full image with Gaussian blur of `blur_radius`.
/// 2. For each pixel, computes the signed distance from a focus band (a line
///    at `focus_pos` fraction of height, rotated by `angle_deg`, with width
///    `band_size` in pixels).
/// 3. Smoothstep mask from the band edge. Blend: `out = in + t * (blurred - in)`.
///
/// `focus_pos`: 0.0 = top, 1.0 = bottom.
pub fn tilt_shift(
    input: &[f32],
    w: u32,
    h: u32,
    focus_pos: f32,
    band_size: f32,
    blur_radius: u32,
    angle_deg: f32,
) -> Vec<f32> {
    let blurred = gaussian_blur_internal(input, w, h, blur_radius);

    let angle_rad = angle_deg * core::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Focus line center in pixel coordinates
    let center_y = focus_pos * h as f32;
    let center_x = w as f32 / 2.0;

    let half_band = band_size / 2.0;
    // Transition zone beyond the band
    let transition = band_size.max(1.0);

    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            // Vector from focus center
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;

            // Distance from focus line (perpendicular to the rotated line direction)
            // The line direction is (cos_a, sin_a), perpendicular is (-sin_a, cos_a)
            let dist = (dx * (-sin_a) + dy * cos_a).abs();

            // Smoothstep: 0 inside band, 1 far from band
            let t = smoothstep(half_band, half_band + transition, dist);

            let di = (y * w + x) as usize * 4;
            for c in 0..3 {
                out[di + c] = input[di + c] + t * (blurred[di + c] - input[di + c]);
            }
            // alpha unchanged
        }
    }
    out
}

/// Adaptive threshold — local mean thresholding using integral image.
///
/// Computes an integral image of luminance. For each pixel, the local mean
/// is computed over a `(2*radius+1)²` window. Output is 1.0 if the pixel's
/// luminance >= local_mean - offset, else 0.0. All 3 RGB channels get the
/// same value. Alpha preserved.
///
/// Useful for binarizing text/documents with uneven illumination.
pub fn adaptive_threshold(input: &[f32], w: u32, h: u32, radius: u32, offset: f32) -> Vec<f32> {
    let wi = w as usize;
    let hi = h as usize;

    // Build luminance integral image (1-indexed for simpler area queries)
    // integral[y+1][x+1] = sum of luminance for all pixels (0..=x, 0..=y)
    let mut integral = vec![0.0f64; (wi + 1) * (hi + 1)];
    let stride = wi + 1;

    for y in 0..hi {
        let mut row_sum = 0.0f64;
        for x in 0..wi {
            let si = (y * wi + x) * 4;
            let lum = luminance(input[si], input[si + 1], input[si + 2]) as f64;
            row_sum += lum;
            integral[(y + 1) * stride + (x + 1)] =
                row_sum + integral[y * stride + (x + 1)];
        }
    }

    let r = radius as i32;
    let mut out = input.to_vec();

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            // Window bounds (clamped to image)
            let x0 = (x - r).max(0) as usize;
            let y0 = (y - r).max(0) as usize;
            let x1 = (x + r).min(w as i32 - 1) as usize + 1; // exclusive in integral
            let y1 = (y + r).min(h as i32 - 1) as usize + 1;

            let area = ((x1 - x0) * (y1 - y0)) as f64;
            let sum = integral[y1 * stride + x1]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0]
                + integral[y0 * stride + x0];
            let local_mean = (sum / area) as f32;

            let si = (y as usize * wi + x as usize) * 4;
            let lum = luminance(input[si], input[si + 1], input[si + 2]);

            let val = if lum >= local_mean - offset {
                1.0
            } else {
                0.0
            };

            out[si] = val;
            out[si + 1] = val;
            out[si + 2] = val;
            // alpha unchanged
        }
    }
    out
}

/// Scharr edge detection — gradient magnitude using Scharr kernels.
///
/// Gx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
/// Gy = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
///
/// Per channel: `magnitude = sqrt(gx² + gy²) * scale / 32`
/// The `/32` normalizes the Scharr kernel sum (3+10+3 = 16 per side, max response 32).
/// Edge pixels clamp to nearest.
pub fn scharr(input: &[f32], w: u32, h: u32, scale: f32) -> Vec<f32> {
    let wi = w as i32;
    let hi = h as i32;
    let mut out = input.to_vec();

    #[rustfmt::skip]
    let gx_kernel: [[f32; 3]; 3] = [
        [-3.0,  0.0,  3.0],
        [-10.0, 0.0, 10.0],
        [-3.0,  0.0,  3.0],
    ];
    #[rustfmt::skip]
    let gy_kernel: [[f32; 3]; 3] = [
        [-3.0, -10.0, -3.0],
        [ 0.0,   0.0,  0.0],
        [ 3.0,  10.0,  3.0],
    ];

    for y in 0..hi {
        for x in 0..wi {
            let mut mag = [0.0f32; 3];

            for c in 0..3 {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;

                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sx = (x + kx - 1).clamp(0, wi - 1);
                        let sy = (y + ky - 1).clamp(0, hi - 1);
                        let si = (sy * wi + sx) as usize * 4 + c;
                        let val = input[si];
                        gx += val * gx_kernel[ky as usize][kx as usize];
                        gy += val * gy_kernel[ky as usize][kx as usize];
                    }
                }

                mag[c] = (gx * gx + gy * gy).sqrt() * scale / 32.0;
            }

            let di = (y * wi + x) as usize * 4;
            out[di] = mag[0];
            out[di + 1] = mag[1];
            out[di + 2] = mag[2];
            // alpha unchanged
        }
    }
    out
}

/// Smart sharpen — bilateral-based unsharp mask with edge-aware thresholding.
///
/// 1. Computes bilateral blur with `sigma_spatial = radius` and
///    `sigma_range = threshold`.
/// 2. Unsharp mask: `out = in + amount * (in - bilateral(in))`
///
/// The bilateral blur preserves edges, so sharpening is applied primarily to
/// textures/detail rather than strong edges (reducing halo artifacts).
///
/// `amount=0` returns identity.
pub fn smart_sharpen(
    input: &[f32],
    w: u32,
    h: u32,
    amount: f32,
    radius: u32,
    threshold: f32,
) -> Vec<f32> {
    if amount == 0.0 {
        return input.to_vec();
    }

    let sigma_spatial = radius as f32;
    let sigma_range = threshold;

    // Bilateral filter (reimplemented inline to avoid cross-module dependency)
    let blurred = bilateral_filter(input, w, h, sigma_spatial, sigma_range);

    let mut out = input.to_vec();
    for (o, (i, b)) in out
        .chunks_exact_mut(4)
        .zip(input.chunks_exact(4).zip(blurred.chunks_exact(4)))
    {
        o[0] = i[0] + amount * (i[0] - b[0]);
        o[1] = i[1] + amount * (i[1] - b[1]);
        o[2] = i[2] + amount * (i[2] - b[2]);
        // alpha unchanged
    }
    out
}

/// Bilateral filter — edge-preserving smoothing (local implementation).
fn bilateral_filter(
    input: &[f32],
    w: u32,
    h: u32,
    sigma_spatial: f32,
    sigma_range: f32,
) -> Vec<f32> {
    let wi = w as i32;
    let hi = h as i32;
    let radius = (2.0 * sigma_spatial).ceil().max(1.0) as i32;
    let spatial_coeff = if sigma_spatial > 0.0 {
        -0.5 / (sigma_spatial * sigma_spatial)
    } else {
        0.0
    };
    let range_coeff = if sigma_range > 0.0 {
        -0.5 / (sigma_range * sigma_range)
    } else {
        0.0
    };

    let mut out = input.to_vec();

    for y in 0..hi {
        for x in 0..wi {
            let ci = (y * wi + x) as usize * 4;
            let center = [input[ci], input[ci + 1], input[ci + 2]];

            let mut sum = [0.0f32; 3];
            let mut wt = [0.0f32; 3];

            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let sx = (x + kx).clamp(0, wi - 1);
                    let sy = (y + ky).clamp(0, hi - 1);
                    let si = (sy * wi + sx) as usize * 4;

                    let dist_sq = (kx * kx + ky * ky) as f32;
                    let spatial_w = (dist_sq * spatial_coeff).exp();

                    for c in 0..3 {
                        let diff = input[si + c] - center[c];
                        let range_w = (diff * diff * range_coeff).exp();
                        let w = spatial_w * range_w;
                        sum[c] += input[si + c] * w;
                        wt[c] += w;
                    }
                }
            }

            for c in 0..3 {
                out[ci + c] = if wt[c] > 0.0 {
                    sum[c] / wt[c]
                } else {
                    center[c]
                };
            }
        }
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_blur_length0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = motion_blur(&input, 8, 8, 45.0, 0);
        assert_eq!(input, output);
    }

    #[test]
    fn zoom_blur_factor0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = zoom_blur(&input, 8, 8, 4.0, 4.0, 0.0);
        assert_eq!(input, output);
    }

    #[test]
    fn spin_blur_angle0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = spin_blur(&input, 8, 8, 4.0, 4.0, 0.0);
        assert_eq!(input, output);
    }

    #[test]
    fn lens_blur_radius0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = lens_blur(&input, 8, 8, 0);
        assert_eq!(input, output);
    }

    #[test]
    fn adaptive_threshold_uniform_is_white() {
        // Uniform image: every pixel has same luminance, so luma >= mean - offset
        // for any offset >= 0 => all output is 1.0
        let input = crate::solid(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let output = adaptive_threshold(&input, 8, 8, 2, 0.0);
        for i in 0..(8 * 8) {
            let base = i * 4;
            assert_eq!(output[base], 1.0, "pixel {i} R");
            assert_eq!(output[base + 1], 1.0, "pixel {i} G");
            assert_eq!(output[base + 2], 1.0, "pixel {i} B");
            assert_eq!(output[base + 3], 1.0, "pixel {i} A");
        }
    }

    #[test]
    fn scharr_solid_color_is_zero() {
        let input = crate::solid(8, 8, [0.4, 0.6, 0.8, 1.0]);
        let output = scharr(&input, 8, 8, 1.0);
        for i in 0..(8 * 8) {
            let base = i * 4;
            assert!(
                output[base].abs() < 1e-6,
                "pixel {i} R: {}",
                output[base]
            );
            assert!(
                output[base + 1].abs() < 1e-6,
                "pixel {i} G: {}",
                output[base + 1]
            );
            assert!(
                output[base + 2].abs() < 1e-6,
                "pixel {i} B: {}",
                output[base + 2]
            );
        }
    }

    #[test]
    fn motion_blur_preserves_alpha() {
        let mut input = crate::gradient(4, 4);
        for i in 0..(4 * 4) {
            input[i * 4 + 3] = i as f32 / 15.0;
        }
        let output = motion_blur(&input, 4, 4, 0.0, 3);
        for i in 0..(4 * 4) {
            assert_eq!(
                input[i * 4 + 3], output[i * 4 + 3],
                "alpha changed at pixel {i}"
            );
        }
    }

    #[test]
    fn lens_blur_solid_is_identity() {
        let input = crate::solid(8, 8, [0.3, 0.6, 0.9, 1.0]);
        let output = lens_blur(&input, 8, 8, 2);
        crate::assert_parity("lens_blur_solid", &output, &input, 1e-6);
    }

    #[test]
    fn bokeh_blur_disc_matches_lens_blur() {
        let input = crate::noise(8, 8, 42);
        let lens = lens_blur(&input, 8, 8, 2);
        let bokeh = bokeh_blur(&input, 8, 8, 2, false);
        crate::assert_parity("bokeh_disc_vs_lens", &bokeh, &lens, 1e-6);
    }

    #[test]
    fn smart_sharpen_amount0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = smart_sharpen(&input, 8, 8, 0.0, 2, 0.1);
        assert_eq!(input, output);
    }

    #[test]
    fn scharr_detects_vertical_edge() {
        // Create a 4x4 image with a vertical edge: left half black, right half white
        let mut input = vec![0.0f32; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let val = if x >= 2 { 1.0 } else { 0.0 };
                let i = (y * 4 + x) * 4;
                input[i] = val;
                input[i + 1] = val;
                input[i + 2] = val;
                input[i + 3] = 1.0;
            }
        }
        let output = scharr(&input, 4, 4, 1.0);
        // Pixels at x=1 and x=2 should have nonzero gradient
        let edge_pixel = (1 * 4 + 1) * 4; // pixel (1,1)
        assert!(
            output[edge_pixel] > 0.01,
            "Scharr should detect edge, got {}",
            output[edge_pixel]
        );
    }
}
