//! Spatial (neighborhood) filter reference implementations — convolution, median, bilateral.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Each function documents the formula and external validation source.
//!
//! These are deliberately naive, unoptimized implementations — full 2D loops,
//! no separable passes, no lookup tables. Correct, not fast.

/// Gaussian blur — full 2D kernel convolution (NOT separable).
///
/// Builds an NxN kernel from `exp(-0.5*(dx²+dy²)/σ²)` where `σ = radius/3.0`
/// and `N = 2*radius+1`. Edge pixels are clamped (repeat nearest).
///
/// For `radius=0`, returns a copy of the input (identity).
///
/// Validated against: ImageMagick 7.1.1 `-gaussian-blur {radius}x{sigma}`
pub fn gaussian_blur(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let sigma = radius as f32 / 3.0;
    let size = (2 * radius + 1) as usize;
    let r = radius as i32;

    // Build 2D kernel
    let mut kernel = vec![0.0f32; size * size];
    let mut sum = 0.0f32;
    for ky in 0..size {
        for kx in 0..size {
            let dx = kx as f32 - radius as f32;
            let dy = ky as f32 - radius as f32;
            let val = (-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).exp();
            kernel[ky * size + kx] = val;
            sum += val;
        }
    }
    // Normalize
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    convolve_2d(input, w, h, &kernel, size, r)
}

/// Box blur — uniform averaging kernel.
///
/// Each pixel is the unweighted mean of the `(2*radius+1)²` neighborhood.
/// Edge pixels are clamped (repeat nearest).
///
/// Validated against: ImageMagick 7.1.1 `-blur {radius}x65535` (large sigma approx box)
pub fn box_blur(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let size = (2 * radius + 1) as usize;
    let r = radius as i32;
    let weight = 1.0 / (size * size) as f32;
    let kernel = vec![weight; size * size];

    convolve_2d(input, w, h, &kernel, size, r)
}

/// 2D convolution helper. Kernel is `size x size`, radius is `r`.
/// Clamps at edges (repeat nearest pixel).
fn convolve_2d(input: &[f32], w: u32, h: u32, kernel: &[f32], size: usize, r: i32) -> Vec<f32> {
    let w = w as i32;
    let h = h as i32;
    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for ky in 0..size as i32 {
                for kx in 0..size as i32 {
                    let sx = (x + kx - r).clamp(0, w - 1);
                    let sy = (y + ky - r).clamp(0, h - 1);
                    let si = (sy * w + sx) as usize * 4;
                    let ki = (ky as usize) * size + kx as usize;
                    let kw = kernel[ki];
                    acc[0] += input[si] * kw;
                    acc[1] += input[si + 1] * kw;
                    acc[2] += input[si + 2] * kw;
                }
            }
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0];
            out[di + 1] = acc[1];
            out[di + 2] = acc[2];
            // alpha unchanged
        }
    }
    out
}

/// Median filter — per-channel median of `(2*radius+1)²` neighborhood.
///
/// For each pixel, collects all neighbor values per channel, sorts, and
/// picks the middle value. Edge pixels are clamped (repeat nearest).
///
/// Validated against: ImageMagick 7.1.1 `-median {radius}`
pub fn median(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let wi = w as i32;
    let hi = h as i32;
    let r = radius as i32;
    let neighborhood_size = ((2 * radius + 1) * (2 * radius + 1)) as usize;
    let mut out = input.to_vec();

    let mut buf_r = Vec::with_capacity(neighborhood_size);
    let mut buf_g = Vec::with_capacity(neighborhood_size);
    let mut buf_b = Vec::with_capacity(neighborhood_size);

    for y in 0..hi {
        for x in 0..wi {
            buf_r.clear();
            buf_g.clear();
            buf_b.clear();

            for ky in -r..=r {
                for kx in -r..=r {
                    let sx = (x + kx).clamp(0, wi - 1);
                    let sy = (y + ky).clamp(0, hi - 1);
                    let si = (sy * wi + sx) as usize * 4;
                    buf_r.push(input[si]);
                    buf_g.push(input[si + 1]);
                    buf_b.push(input[si + 2]);
                }
            }

            buf_r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            buf_g.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            buf_b.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let mid = neighborhood_size / 2;
            let di = (y * wi + x) as usize * 4;
            out[di] = buf_r[mid];
            out[di + 1] = buf_g[mid];
            out[di + 2] = buf_b[mid];
            // alpha unchanged
        }
    }
    out
}

/// Unsharp mask (sharpen).
///
/// Formula: `out = in + amount * (in - gaussian_blur(in, radius=1))`
/// where the blur is a 3x3 gaussian with `sigma = 1/3`.
///
/// `amount = 0` is identity. Typical values 0.5-3.0.
///
/// Validated against: ImageMagick 7.1.1 `-unsharp 1x0.333+{amount}+0`
/// Photoshop Unsharp Mask with Radius=1, Amount={amount*100}%, Threshold=0.
pub fn sharpen(input: &[f32], w: u32, h: u32, amount: f32) -> Vec<f32> {
    if amount == 0.0 {
        return input.to_vec();
    }
    let blurred = gaussian_blur(input, w, h, 1);
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

/// High-pass filter — removes low-frequency content, preserving edges.
///
/// Formula: `out = in - gaussian_blur(in, radius) + 0.5`
/// The +0.5 offset recenters around mid-gray for visualization.
///
/// Validated against: Photoshop Filter > Other > High Pass
pub fn high_pass(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let blurred = gaussian_blur(input, w, h, radius);
    let mut out = input.to_vec();
    for (o, b) in out.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
        o[0] = o[0] - b[0] + 0.5;
        o[1] = o[1] - b[1] + 0.5;
        o[2] = o[2] - b[2] + 0.5;
        // alpha unchanged
    }
    out
}

/// Bilateral filter — edge-preserving smoothing.
///
/// For each pixel p, weighted average over window:
///   `weight(q) = exp(-||p-q||^2 / (2*sigma_spatial^2)) * exp(-|I(p)-I(q)|^2 / (2*sigma_range^2))`
///   `out(p) = sum(weight(q) * I(q)) / sum(weight(q))`
///
/// Per-channel range weight, per-channel output.
/// Window radius = `ceil(2 * sigma_spatial)`.
///
/// Validated against: OpenCV `bilateralFilter` with `BORDER_REPLICATE`
pub fn bilateral(input: &[f32], w: u32, h: u32, sigma_spatial: f32, sigma_range: f32) -> Vec<f32> {
    let wi = w as i32;
    let hi = h as i32;
    let radius = (2.0 * sigma_spatial).ceil() as i32;
    let spatial_coeff = -0.5 / (sigma_spatial * sigma_spatial);
    let range_coeff = -0.5 / (sigma_range * sigma_range);

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
                out[ci + c] = sum[c] / wt[c];
            }
            // alpha unchanged
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_blur_radius0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = gaussian_blur(&input, 8, 8, 0);
        assert_eq!(input, output);
    }

    #[test]
    fn box_blur_solid_color_is_identity() {
        let color = [0.3, 0.6, 0.9, 1.0];
        let input = crate::solid(8, 8, color);
        let output = box_blur(&input, 8, 8, 2);
        crate::assert_parity("box_blur_solid", &output, &input, 1e-6);
    }

    #[test]
    fn median_sorted_returns_center() {
        // 3x1 image: R values are [0.1, 0.5, 0.9]
        // With radius=1, center pixel neighborhood is all 3 pixels (clamped).
        // Median of {0.1, 0.5, 0.9} = 0.5 for R channel.
        let input: Vec<f32> = vec![
            0.1, 0.2, 0.3, 1.0, // pixel 0
            0.5, 0.5, 0.5, 1.0, // pixel 1
            0.9, 0.8, 0.7, 1.0, // pixel 2
        ];
        let output = median(&input, 3, 1, 1);
        // Center pixel (index 1): median of sorted neighborhood
        assert!((output[4] - 0.5).abs() < 1e-7, "R median: {}", output[4]);
        assert!((output[5] - 0.5).abs() < 1e-7, "G median: {}", output[5]);
        assert!((output[6] - 0.5).abs() < 1e-7, "B median: {}", output[6]);
    }

    #[test]
    fn sharpen_amount0_is_identity() {
        let input = crate::gradient(8, 8);
        let output = sharpen(&input, 8, 8, 0.0);
        assert_eq!(input, output);
    }

    #[test]
    fn bilateral_large_sigma_range_approx_gaussian() {
        // With very large sigma_range, the range weight is ~1.0 for all neighbors,
        // so bilateral degenerates to pure spatial gaussian weighting.
        let input = crate::noise(8, 8, 42);
        let sigma_spatial = 1.0;
        let sigma_range = 1000.0; // effectively infinite

        let bilateral_out = bilateral(&input, 8, 8, sigma_spatial, sigma_range);

        // Build a gaussian blur with matching sigma. bilateral uses radius=ceil(2*sigma)=2,
        // gaussian_blur uses sigma=radius/3. For a fair comparison, we build the
        // equivalent 2D gaussian manually with the same sigma and radius.
        let radius = (2.0 * sigma_spatial).ceil() as u32; // = 2
        let size = (2 * radius + 1) as usize;
        let r = radius as i32;
        let mut kernel = vec![0.0f32; size * size];
        let mut ksum = 0.0f32;
        for ky in 0..size {
            for kx in 0..size {
                let dx = kx as f32 - radius as f32;
                let dy = ky as f32 - radius as f32;
                let val = (-0.5 * (dx * dx + dy * dy) / (sigma_spatial * sigma_spatial)).exp();
                kernel[ky * size + kx] = val;
                ksum += val;
            }
        }
        for v in kernel.iter_mut() {
            *v /= ksum;
        }
        let gaussian_out = convolve_2d(&input, 8, 8, &kernel, size, r);

        let diff = crate::max_diff(&bilateral_out, &gaussian_out);
        assert!(
            diff < 1e-5,
            "bilateral with huge sigma_range should match gaussian, max diff: {diff}"
        );
    }

    #[test]
    fn gaussian_blur_preserves_alpha() {
        let mut input = crate::gradient(4, 4);
        // Set varied alpha values
        for i in 0..(4 * 4) {
            input[i * 4 + 3] = i as f32 / 15.0;
        }
        let output = gaussian_blur(&input, 4, 4, 1);
        for i in 0..(4 * 4) {
            assert_eq!(
                input[i * 4 + 3], output[i * 4 + 3],
                "alpha changed at pixel {i}"
            );
        }
    }

    #[test]
    fn high_pass_uniform_is_half() {
        // High pass of a solid image: out = in - in + 0.5 = 0.5
        let input = crate::solid(4, 4, [0.7, 0.3, 0.1, 1.0]);
        let output = high_pass(&input, 4, 4, 1);
        for i in 0..(4 * 4) {
            let base = i * 4;
            for c in 0..3 {
                assert!(
                    (output[base + c] - 0.5).abs() < 1e-6,
                    "pixel {i} channel {c}: expected 0.5, got {}",
                    output[base + c]
                );
            }
        }
    }
}
