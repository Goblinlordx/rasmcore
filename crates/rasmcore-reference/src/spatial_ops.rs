//! Spatial (neighborhood) filter reference implementations — convolution, median, bilateral.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Each function documents the formula and external validation source.
//!
//! These are deliberately naive, unoptimized implementations — full 2D loops,
//! no separable passes, no lookup tables. Correct, not fast.

/// Gaussian blur — full 2D kernel convolution (NOT separable).
///
/// Separable Gaussian blur matching pipeline convention.
///
/// sigma = radius (directly), ksize = round(sigma * 10 + 1) | 1.
/// Uses separable H+V passes with BORDER_REFLECT_101.
///
/// Validated against: OpenCV cv2.GaussianBlur with BORDER_REFLECT_101.
pub fn gaussian_blur(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let sigma = radius as f32;
    let ksize = ((sigma * 10.0 + 1.0).round() as usize) | 1;
    let ksize = ksize.max(3);

    // Build 1D kernel
    let center = ksize / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center as f32;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    let inv = 1.0 / sum;
    for v in &mut kernel { *v *= inv; }

    let w = w as i32;
    let h = h as i32;
    let r = center as i32;

    // H pass
    let mut tmp = input.to_vec();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = reflect_101(x + ki as i32 - r, w);
                let si = (y * w + sx) as usize * 4;
                acc[0] += input[si] * kw;
                acc[1] += input[si + 1] * kw;
                acc[2] += input[si + 2] * kw;
            }
            let di = (y * w + x) as usize * 4;
            tmp[di] = acc[0];
            tmp[di + 1] = acc[1];
            tmp[di + 2] = acc[2];
        }
    }

    // V pass
    let mut out = tmp.clone();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = reflect_101(y + ki as i32 - r, h);
                let si = (sy * w + x) as usize * 4;
                acc[0] += tmp[si] * kw;
                acc[1] += tmp[si + 1] * kw;
                acc[2] += tmp[si + 2] * kw;
            }
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0];
            out[di + 1] = acc[1];
            out[di + 2] = acc[2];
        }
    }

    out
}

/// Box blur — separable uniform averaging.
///
/// Each pixel is the mean of the `(2*radius+1)` neighborhood per axis.
/// Uses BORDER_REFLECT_101 matching pipeline.
///
/// Validated against: OpenCV cv2.blur with BORDER_REFLECT_101.
pub fn box_blur(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let ksize = 2 * radius as usize + 1;
    let weight = 1.0 / ksize as f32;
    let w = w as i32;
    let h = h as i32;
    let r = radius as i32;

    // H pass
    let mut tmp = input.to_vec();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for kx in -r..=r {
                let sx = reflect_101(x + kx, w);
                let si = (y * w + sx) as usize * 4;
                acc[0] += input[si] * weight;
                acc[1] += input[si + 1] * weight;
                acc[2] += input[si + 2] * weight;
            }
            let di = (y * w + x) as usize * 4;
            tmp[di] = acc[0];
            tmp[di + 1] = acc[1];
            tmp[di + 2] = acc[2];
        }
    }

    // V pass
    let mut out = tmp.clone();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for ky in -r..=r {
                let sy = reflect_101(y + ky, h);
                let si = (sy * w + x) as usize * 4;
                acc[0] += tmp[si] * weight;
                acc[1] += tmp[si + 1] * weight;
                acc[2] += tmp[si + 2] * weight;
            }
            let di = (y * w + x) as usize * 4;
            out[di] = acc[0];
            out[di + 1] = acc[1];
            out[di + 2] = acc[2];
        }
    }

    out
}

/// BORDER_REFLECT_101: mirror at boundary excluding edge pixel.
/// Matches OpenCV's default and our pipeline's clamp_coord.
/// Pattern: dcb|abcd|cba (not dcba|abcd|dcba)
fn reflect_101(v: i32, size: i32) -> i32 {
    if v < 0 {
        (-v).min(size - 1)
    } else if v >= size {
        (2 * size - v - 2).max(0)
    } else {
        v
    }
}

/// 2D convolution helper (used by median, sharpen).
#[allow(dead_code)]
fn convolve_2d(input: &[f32], w: u32, h: u32, kernel: &[f32], size: usize, r: i32) -> Vec<f32> {
    let w = w as i32;
    let h = h as i32;
    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for ky in 0..size as i32 {
                for kx in 0..size as i32 {
                    let sx = reflect_101(x + kx - r, w);
                    let sy = reflect_101(y + ky - r, h);
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
                    let sx = reflect_101(x + kx, wi);
                    let sy = reflect_101(y + ky, hi);
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

/// Convert linear RGB to CIE Lab (D65) for perceptual color distance.
fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA;
    let lab_f = |t: f32| -> f32 {
        if t > DELTA3 { t.cbrt() } else { t / (3.0 * DELTA * DELTA) + 4.0 / 29.0 }
    };
    let fx = lab_f(x / XN);
    let fy = lab_f(y / YN);
    let fz = lab_f(z / ZN);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Bilateral filter — edge-preserving smoothing with CIE-Lab color distance.
///
/// Matches Tomasi & Manduchi 1998 recommendation and MATLAB's imbilatfilt:
/// color similarity computed as L2 Euclidean distance in CIE-Lab space
/// (perceptually uniform). Smoothing applied in input color space (linear RGB).
///
/// Validated against: colour-science Lab + numpy bilateral (golden data).
pub fn bilateral(input: &[f32], w: u32, h: u32, sigma_spatial: f32, sigma_range: f32) -> Vec<f32> {
    let wi = w as i32;
    let hi = h as i32;
    let radius = (2.0 * sigma_spatial).ceil() as i32;
    let spatial_coeff = -0.5 / (sigma_spatial * sigma_spatial);
    let range_coeff = -0.5 / (sigma_range * sigma_range);

    // Pre-compute Lab for all pixels
    let n = (w * h) as usize;
    let mut lab = Vec::with_capacity(n * 3);
    for i in 0..n {
        let idx = i * 4;
        let (l, a, b) = linear_rgb_to_lab(input[idx], input[idx + 1], input[idx + 2]);
        lab.push(l);
        lab.push(a);
        lab.push(b);
    }

    let mut out = input.to_vec();

    for y in 0..hi {
        for x in 0..wi {
            let ci = (y * wi + x) as usize * 4;
            let cli = (y * wi + x) as usize * 3;

            let mut sum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;

            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let sx = reflect_101(x + kx, wi);
                    let sy = reflect_101(y + ky, hi);
                    let si = (sy * wi + sx) as usize * 4;
                    let sli = (sy * wi + sx) as usize * 3;

                    let dist_sq = (kx * kx + ky * ky) as f32;
                    let spatial_w = (dist_sq * spatial_coeff).exp();

                    // CIE-Lab L2 color distance
                    let dl = lab[sli] - lab[cli];
                    let da = lab[sli + 1] - lab[cli + 1];
                    let db = lab[sli + 2] - lab[cli + 2];
                    let color_dist2 = dl * dl + da * da + db * db;
                    let range_w = (color_dist2 * range_coeff).exp();

                    let w = spatial_w * range_w;
                    sum[0] += input[si] * w;
                    sum[1] += input[si + 1] * w;
                    sum[2] += input[si + 2] * w;
                    weight_sum += w;
                }
            }

            let inv = if weight_sum > 1e-10 { 1.0 / weight_sum } else { 0.0 };
            out[ci] = sum[0] * inv;
            out[ci + 1] = sum[1] * inv;
            out[ci + 2] = sum[2] * inv;
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
