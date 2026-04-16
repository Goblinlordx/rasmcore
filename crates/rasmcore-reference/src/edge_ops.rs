//! Edge detection and morphology reference implementations.
//!
//! All operations work in **linear f32** space (RGBA, 4 channels interleaved).
//! Alpha is preserved unchanged. Pure math — no SIMD, no GPU, no external crates.
//! Each function documents the formula and intended validation source.

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Sample a pixel channel with clamped (replicate-border) addressing.
#[inline]
fn sample(input: &[f32], w: u32, h: u32, x: i32, y: i32, c: usize) -> f32 {
    let cx = x.clamp(0, w as i32 - 1) as u32;
    let cy = y.clamp(0, h as i32 - 1) as u32;
    input[((cy * w + cx) * 4 + c as u32) as usize]
}

/// Luminance from linear RGB (BT.709 coefficients).
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Sample luminance at (x,y) with clamped borders.
#[inline]
fn sample_lum(input: &[f32], w: u32, h: u32, x: i32, y: i32) -> f32 {
    let r = sample(input, w, h, x, y, 0);
    let g = sample(input, w, h, x, y, 1);
    let b = sample(input, w, h, x, y, 2);
    luminance(r, g, b)
}

// ─── Sobel ─────────────────────────────────────────────────────────────────────

/// Sobel edge detection — gradient magnitude via 3×3 Gx/Gy kernels.
///
/// Computes on luminance (`0.2126*R + 0.7152*G + 0.0722*B`), outputs
/// grayscale magnitude `sqrt(gx² + gy²)` (same value in R, G, B).
///
/// Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
/// Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -morphology Convolve Sobel`
pub fn sobel(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;

            // Gather 3×3 luminance neighborhood
            let l00 = sample_lum(input, w, h, x as i32 - 1, y as i32 - 1);
            let l10 = sample_lum(input, w, h, x as i32, y as i32 - 1);
            let l20 = sample_lum(input, w, h, x as i32 + 1, y as i32 - 1);
            let l01 = sample_lum(input, w, h, x as i32 - 1, y as i32);
            let l21 = sample_lum(input, w, h, x as i32 + 1, y as i32);
            let l02 = sample_lum(input, w, h, x as i32 - 1, y as i32 + 1);
            let l12 = sample_lum(input, w, h, x as i32, y as i32 + 1);
            let l22 = sample_lum(input, w, h, x as i32 + 1, y as i32 + 1);

            let gx = -l00 + l20 - 2.0 * l01 + 2.0 * l21 - l02 + l22;
            let gy = -l00 - 2.0 * l10 - l20 + l02 + 2.0 * l12 + l22;
            let mag = (gx * gx + gy * gy).sqrt();

            out[idx] = mag;
            out[idx + 1] = mag;
            out[idx + 2] = mag;
            out[idx + 3] = input[idx + 3]; // alpha preserved
        }
    }
    out
}

/// Internal: compute Sobel gradient magnitude and direction per-pixel (grayscale).
/// Returns (magnitude, angle_in_radians) buffers, each w*h.
fn sobel_gradient(input: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let l00 = sample_lum(input, w, h, x as i32 - 1, y as i32 - 1);
            let l10 = sample_lum(input, w, h, x as i32, y as i32 - 1);
            let l20 = sample_lum(input, w, h, x as i32 + 1, y as i32 - 1);
            let l01 = sample_lum(input, w, h, x as i32 - 1, y as i32);
            let l21 = sample_lum(input, w, h, x as i32 + 1, y as i32);
            let l02 = sample_lum(input, w, h, x as i32 - 1, y as i32 + 1);
            let l12 = sample_lum(input, w, h, x as i32, y as i32 + 1);
            let l22 = sample_lum(input, w, h, x as i32 + 1, y as i32 + 1);

            let gx = -l00 + l20 - 2.0 * l01 + 2.0 * l21 - l02 + l22;
            let gy = -l00 - 2.0 * l10 - l20 + l02 + 2.0 * l12 + l22;
            mag[idx] = (gx * gx + gy * gy).sqrt();
            dir[idx] = gy.atan2(gx);
        }
    }
    (mag, dir)
}

// ─── Laplacian ─────────────────────────────────────────────────────────────────

/// Laplacian edge detection — second derivative approximation on luminance.
///
/// Kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
/// Output is absolute value of convolution result, written as grayscale
/// (same value in R, G, B).
///
/// Validated against: OpenCV `cv::Laplacian` with ksize=1
pub fn laplacian(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;

            let center = sample_lum(input, w, h, x as i32, y as i32);
            let top = sample_lum(input, w, h, x as i32, y as i32 - 1);
            let bottom = sample_lum(input, w, h, x as i32, y as i32 + 1);
            let left = sample_lum(input, w, h, x as i32 - 1, y as i32);
            let right = sample_lum(input, w, h, x as i32 + 1, y as i32);

            let lap = (4.0 * center - top - bottom - left - right).abs();

            out[idx] = lap;
            out[idx + 1] = lap;
            out[idx + 2] = lap;
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

// ─── Canny ─────────────────────────────────────────────────────────────────────

/// Simplified Canny edge detection.
///
/// Steps:
///   1. Gaussian blur (sigma=1.4, 5×5 kernel)
///   2. Sobel gradient magnitude + direction (on luminance)
///   3. Non-maximum suppression
///   4. Double threshold + hysteresis edge tracking
///
/// Output: binary edge mask (0.0 or 1.0 in RGB channels, alpha preserved).
///
/// Validated against: OpenCV `cv::Canny` (approximate — Canny is implementation-sensitive)
pub fn canny(
    input: &[f32],
    w: u32,
    h: u32,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<f32> {
    let n = (w * h) as usize;

    // Step 1: Gaussian blur, sigma=1.4, 5×5 approximation
    let blurred = gaussian_blur(input, w, h, 1.4);

    // Step 2: Sobel gradient on blurred image
    let (mag, dir) = sobel_gradient(&blurred, w, h);

    // Step 3: Non-maximum suppression
    let mut nms = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let angle = dir[idx];

            // Quantize angle to 4 directions: 0°, 45°, 90°, 135°
            let a = angle.to_degrees().rem_euclid(180.0);
            let (dx, dy): (i32, i32) = if a < 22.5 || a >= 157.5 {
                (1, 0) // horizontal
            } else if a < 67.5 {
                (1, 1) // 45°
            } else if a < 112.5 {
                (0, 1) // vertical
            } else {
                (-1, 1) // 135°
            };

            let x1 = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
            let y1 = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
            let x2 = (x as i32 - dx).clamp(0, w as i32 - 1) as u32;
            let y2 = (y as i32 - dy).clamp(0, h as i32 - 1) as u32;

            let m = mag[idx];
            let m1 = mag[(y1 * w + x1) as usize];
            let m2 = mag[(y2 * w + x2) as usize];

            if m >= m1 && m >= m2 {
                nms[idx] = m;
            }
        }
    }

    // Step 4: Double threshold + hysteresis
    // Classify: 0=suppressed, 1=weak, 2=strong
    let mut edge_type = vec![0u8; n];
    for i in 0..n {
        if nms[i] >= high_threshold {
            edge_type[i] = 2;
        } else if nms[i] >= low_threshold {
            edge_type[i] = 1;
        }
    }

    // Hysteresis: promote weak edges connected to strong edges
    let mut changed = true;
    while changed {
        changed = false;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if edge_type[idx] != 1 {
                    continue;
                }
                // Check 8-connected neighbors for strong edge
                'neighbors: for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            let ni = (ny as u32 * w + nx as u32) as usize;
                            if edge_type[ni] == 2 {
                                edge_type[idx] = 2;
                                changed = true;
                                break 'neighbors;
                            }
                        }
                    }
                }
            }
        }
    }

    // Build output: strong edges = 1.0, everything else = 0.0
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let pidx = idx * 4;
            let v = if edge_type[idx] == 2 { 1.0 } else { 0.0 };
            out[pidx] = v;
            out[pidx + 1] = v;
            out[pidx + 2] = v;
            out[pidx + 3] = input[pidx + 3];
        }
    }
    out
}

/// Simple Gaussian blur for Canny preprocessing.
/// Generates a (2*r+1)×(2*r+1) kernel from sigma, applies as separable convolution.
fn gaussian_blur(input: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as i32;
    let size = (2 * r + 1) as usize;

    // Build 1D kernel
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for i in 0..size {
        let d = (i as i32 - r) as f32;
        let v = (-d * d / (2.0 * sigma * sigma)).exp();
        kernel[i] = v;
        sum += v;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut acc = 0.0f32;
                for i in 0..size {
                    let sx = x as i32 + i as i32 - r;
                    acc += kernel[i] * sample(input, w, h, sx, y as i32, c);
                }
                temp[idx + c] = acc;
            }
            temp[idx + 3] = input[idx + 3];
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut acc = 0.0f32;
                for i in 0..size {
                    let sy = y as i32 + i as i32 - r;
                    acc += kernel[i] * sample(&temp, w, h, x as i32, sy, c);
                }
                out[idx + c] = acc;
            }
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

// ─── Threshold ─────────────────────────────────────────────────────────────────

/// Binary threshold on luminance.
///
/// Formula: `lum = 0.2126*R + 0.7152*G + 0.0722*B`
/// Output: 1.0 if lum >= level, else 0.0 (all 3 channels same).
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -threshold {level*100}%`
pub fn threshold(input: &[f32], _w: u32, _h: u32, level: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for px_i in 0..(input.len() / 4) {
        let idx = px_i * 4;
        let lum = luminance(input[idx], input[idx + 1], input[idx + 2]);
        let v = if lum >= level { 1.0 } else { 0.0 };
        out[idx] = v;
        out[idx + 1] = v;
        out[idx + 2] = v;
        out[idx + 3] = input[idx + 3]; // alpha preserved
    }
    out
}

// ─── Otsu ──────────────────────────────────────────────────────────────────────

/// Otsu's automatic threshold.
///
/// Computes optimal threshold from luminance histogram (256 bins),
/// minimizes intra-class variance (equivalently maximizes inter-class variance),
/// then applies binary threshold at that level.
///
/// Validated against: OpenCV `cv::threshold` with THRESH_BINARY | THRESH_OTSU
pub fn otsu_threshold(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let n = (w * h) as usize;

    // Build luminance histogram (256 bins, values clamped to [0,1])
    let mut histogram = [0u32; 256];
    for i in 0..n {
        let pi = i * 4;
        let lum = luminance(input[pi], input[pi + 1], input[pi + 2]).clamp(0.0, 1.0);
        let bin = (lum * 255.0).round() as usize;
        let bin = bin.min(255);
        histogram[bin] += 1;
    }

    // Otsu's method: find threshold that maximizes inter-class variance
    let total = n as f64;
    let mut sum_all = 0.0f64;
    for (i, &count) in histogram.iter().enumerate() {
        sum_all += i as f64 * count as f64;
    }

    let mut best_threshold = 0usize;
    let mut best_variance = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for (t, &count) in histogram.iter().enumerate() {
        w0 += count as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * count as f64;
        let mean0 = sum0 / w0;
        let mean1 = (sum_all - sum0) / w1;
        let variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
        if variance > best_variance {
            best_variance = variance;
            best_threshold = t;
        }
    }

    // Place threshold between bins (matching OpenCV behavior)
    let level = (best_threshold as f32 + 0.5) / 255.0;
    threshold(input, w, h, level)
}

// ─── Morphological Operations ──────────────────────────────────────────────────

/// Morphological dilation — per-channel max in a square window.
///
/// Window size: (2*radius+1) × (2*radius+1)
/// Output pixel = max of all pixels in the neighborhood for each RGB channel.
/// Alpha is preserved from the center pixel.
///
/// Validated against: ImageMagick 7.1.1 `-morphology Dilate Square:{radius}`
pub fn dilate(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut max_val = f32::NEG_INFINITY;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let v = sample(input, w, h, x as i32 + dx, y as i32 + dy, c);
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out[idx + c] = max_val;
            }
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

/// Morphological erosion — per-channel min in a square window.
///
/// Window size: (2*radius+1) × (2*radius+1)
/// Output pixel = min of all pixels in the neighborhood for each RGB channel.
/// Alpha is preserved from the center pixel.
///
/// Validated against: ImageMagick 7.1.1 `-morphology Erode Square:{radius}`
pub fn erode(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut min_val = f32::INFINITY;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let v = sample(input, w, h, x as i32 + dx, y as i32 + dy, c);
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out[idx + c] = min_val;
            }
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

/// Morphological opening — erode then dilate.
///
/// Opening removes small bright features (noise) while preserving shape.
///
/// Validated against: ImageMagick 7.1.1 `-morphology Open Square:{radius}`
pub fn morph_open(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let eroded = erode(input, w, h, radius);
    dilate(&eroded, w, h, radius)
}

/// Morphological closing — dilate then erode.
///
/// Closing fills small dark features (holes) while preserving shape.
///
/// Validated against: ImageMagick 7.1.1 `-morphology Close Square:{radius}`
pub fn morph_close(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let dilated = dilate(input, w, h, radius);
    erode(&dilated, w, h, radius)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a w×h solid image.
    fn solid(w: u32, h: u32, r: f32, g: f32, b: f32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            out.extend_from_slice(&[r, g, b, 1.0]);
        }
        out
    }

    /// Helper: create a binary checkerboard pattern.
    fn checkerboard(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                out.extend_from_slice(&[v, v, v, 1.0]);
            }
        }
        out
    }

    #[test]
    fn sobel_solid_is_zero() {
        let img = solid(8, 8, 0.5, 0.3, 0.7);
        let result = sobel(&img, 8, 8);
        for (i, px) in result.chunks_exact(4).enumerate() {
            assert!(px[0].abs() < 1e-6, "sobel R at pixel {i} should be zero, got {}", px[0]);
            assert!(px[1].abs() < 1e-6, "sobel G at pixel {i} should be zero, got {}", px[1]);
            assert!(px[2].abs() < 1e-6, "sobel B at pixel {i} should be zero, got {}", px[2]);
            assert!((px[3] - 1.0).abs() < 1e-6, "alpha should be preserved");
        }
    }

    #[test]
    fn threshold_zero_all_white() {
        // Any pixel with lum > 0 should pass threshold at 0.0
        let img = solid(4, 4, 0.5, 0.5, 0.5);
        let result = threshold(&img, 4, 4, 0.0);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 1.0, "threshold(0.0) should produce all white");
            assert_eq!(px[1], 1.0);
            assert_eq!(px[2], 1.0);
        }
    }

    #[test]
    fn threshold_one_all_black() {
        // lum(0.5, 0.5, 0.5) = 0.5 < 1.0 → black
        let img = solid(4, 4, 0.5, 0.5, 0.5);
        let result = threshold(&img, 4, 4, 1.0);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 0.0, "threshold(1.0) should produce all black");
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
        }
    }

    #[test]
    fn threshold_preserves_alpha() {
        let mut img = solid(2, 2, 1.0, 1.0, 1.0);
        img[3] = 0.5; // first pixel alpha
        let result = threshold(&img, 2, 2, 0.5);
        assert_eq!(result[3], 0.5, "alpha must be preserved");
    }

    #[test]
    fn dilate_radius_zero_is_identity() {
        let img = checkerboard(4, 4);
        let result = dilate(&img, 4, 4, 0);
        assert_eq!(img, result, "dilate with radius=0 must be identity");
    }

    #[test]
    fn erode_radius_zero_is_identity() {
        let img = checkerboard(4, 4);
        let result = erode(&img, 4, 4, 0);
        assert_eq!(img, result, "erode with radius=0 must be identity");
    }

    #[test]
    fn dilate_then_erode_equals_opening() {
        let img = checkerboard(8, 8);
        let radius = 1;

        let opening = morph_open(&img, 8, 8, radius);

        // Manually: erode then dilate
        let eroded = erode(&img, 8, 8, radius);
        let manual = dilate(&eroded, 8, 8, radius);

        assert_eq!(opening, manual, "morph_open must equal dilate(erode(x))");
    }

    #[test]
    fn close_equals_dilate_then_erode() {
        let img = checkerboard(8, 8);
        let radius = 1;

        let closing = morph_close(&img, 8, 8, radius);

        let dilated = dilate(&img, 8, 8, radius);
        let manual = erode(&dilated, 8, 8, radius);

        assert_eq!(closing, manual, "morph_close must equal erode(dilate(x))");
    }

    #[test]
    fn laplacian_solid_is_zero() {
        let img = solid(8, 8, 0.4, 0.6, 0.2);
        let result = laplacian(&img, 8, 8);
        for (i, px) in result.chunks_exact(4).enumerate() {
            assert!(px[0].abs() < 1e-6, "laplacian on solid should be zero at pixel {i}");
        }
    }

    #[test]
    fn sobel_detects_edge() {
        // Left half black, right half white — boundary should have strong gradient
        let w = 8u32;
        let h = 8u32;
        let mut input = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0.0 } else { 1.0 };
                input.extend_from_slice(&[v, v, v, 1.0]);
            }
        }

        let result = sobel(&input, w, h);

        // Pixel at boundary (x=3, y=4) should have nonzero magnitude
        let edge_idx = ((4 * w + 3) * 4) as usize;
        assert!(result[edge_idx] > 0.1, "sobel should detect edge at boundary");

        // Interior pixel far from edge (x=0, y=4) should be near zero
        let interior_idx = ((4 * w + 0) * 4) as usize;
        assert!(result[interior_idx] < 0.01, "interior pixel should be near zero");
    }

    #[test]
    fn otsu_bimodal() {
        // Half black, half white — Otsu should separate them cleanly
        let mut img = Vec::with_capacity(16 * 4);
        for i in 0..16 {
            let v = if i < 8 { 0.0 } else { 1.0 };
            img.extend_from_slice(&[v, v, v, 1.0]);
        }
        let result = otsu_threshold(&img, 4, 4);

        // Dark pixels should be 0
        for i in 0..8 {
            assert_eq!(result[i * 4], 0.0, "dark pixel {i} should be 0");
        }
        // Bright pixels should be 1
        for i in 8..16 {
            assert_eq!(result[i * 4], 1.0, "bright pixel {i} should be 1");
        }
    }
}
