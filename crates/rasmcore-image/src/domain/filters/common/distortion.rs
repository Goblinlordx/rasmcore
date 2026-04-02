//! Distortion helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Unified distortion engine. Each filter provides its inverse transform
/// and Jacobian as closures. This function handles:
/// - Computing required source rect (bounding box or uniform overlap)
/// - Requesting expanded upstream pixels
/// - Running the sampling loop in image-space coordinates (not tile-local)
/// - Producing output for the requested tile
pub fn apply_distortion(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo, // FULL image dimensions -- never overwrite with tile dims
    overlap: DistortionOverlap,
    sampling: DistortionSampling,
    // inverse_fn: (output_x, output_y) -> (source_x, source_y) in image space
    inverse_fn: &dyn Fn(f32, f32) -> (f32, f32),
    // jacobian_fn: (output_x, output_y) -> 2x2 Jacobian matrix
    // Ignored for Bilinear sampling mode.
    jacobian_fn: &dyn Fn(f32, f32) -> crate::domain::ewa::Jacobian,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width;
    let h = info.height;

    // Compute required source rect
    let source_rect = match overlap {
        DistortionOverlap::Uniform(px) => request.expand_uniform(px, w, h),
        DistortionOverlap::FullImage => Rect::new(0, 0, w, h),
    };

    let pixels = upstream(source_rect)?;
    let ch = channels(info.format);
    let src_w = source_rect.width as usize;
    let src_h = source_rect.height as usize;
    let sample_fmt = crate::domain::ewa::SampleFormat::from_pixel_format(info.format);
    let bpc = sample_fmt.bytes_per_channel();

    let sampler =
        crate::domain::ewa::EwaSampler::new_with_format(&pixels, src_w, src_h, ch, sample_fmt);

    // Output buffer — sized by actual bytes per pixel (ch * bytes_per_channel)
    let out_w = request.width as usize;
    let out_h = request.height as usize;
    let out_bpp = ch * bpc;
    let mut out = vec![0u8; out_w * out_h * out_bpp];

    // Iterate in IMAGE-SPACE coordinates (not tile-local)
    for oy in 0..out_h {
        let img_y = (request.y as usize + oy) as f32;
        for ox in 0..out_w {
            let img_x = (request.x as usize + ox) as f32;

            let (sx, sy) = inverse_fn(img_x, img_y);
            let local_sx = sx - source_rect.x as f32;
            let local_sy = sy - source_rect.y as f32;

            let pixel_off = (oy * out_w + ox) * out_bpp;
            for c in 0..ch {
                // Sample returns value in native format range
                let v = match sampling {
                    DistortionSampling::Bilinear => {
                        sampler.bilinear_pub(local_sx, local_sy, c)
                    }
                    DistortionSampling::Ewa => {
                        let j = jacobian_fn(img_x, img_y);
                        sampler.sample(local_sx, local_sy, &j, c)
                    }
                    DistortionSampling::EwaClamp => {
                        let j = jacobian_fn(img_x, img_y);
                        sampler.sample_clamp(local_sx, local_sy, &j, c)
                    }
                };
                // Write channel value in native format
                let ch_off = pixel_off + c * bpc;
                match sample_fmt {
                    crate::domain::ewa::SampleFormat::U8 => {
                        out[ch_off] = v.round().clamp(0.0, 255.0) as u8;
                    }
                    crate::domain::ewa::SampleFormat::U16 => {
                        let u = v.round().clamp(0.0, 65535.0) as u16;
                        out[ch_off..ch_off + 2].copy_from_slice(&u.to_le_bytes());
                    }
                    crate::domain::ewa::SampleFormat::F16 => {
                        let h = half::f16::from_f32(v);
                        out[ch_off..ch_off + 2].copy_from_slice(&h.to_le_bytes());
                    }
                    crate::domain::ewa::SampleFormat::F32 => {
                        out[ch_off..ch_off + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Invert a 3x3 matrix (row-major). Returns None if singular.
pub fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

/// Public wrapper for integration tests.
pub fn invert_3x3_public(m: &[f64; 9]) -> Option<[f64; 9]> {
    invert_3x3(m)
}

/// Create a padded copy of the image with reflected borders.
///
/// Eliminates per-pixel boundary checks — interior pixels use direct indexing.
pub fn pad_reflect(pixels: &[u8], w: usize, h: usize, channels: usize, pad: usize) -> Vec<u8> {
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    let mut out = vec![0u8; pw * ph * channels];

    for py in 0..ph {
        let sy = reflect(py as i32 - pad as i32, h);
        for px in 0..pw {
            let sx = reflect(px as i32 - pad as i32, w);
            let src = (sy * w + sx) * channels;
            let dst = (py * pw + px) * channels;
            out[dst..dst + channels].copy_from_slice(&pixels[src..src + channels]);
        }
    }
    out
}

/// Reflect-edge coordinate clamping.
pub fn reflect(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

/// BORDER_REFLECT_101: reflect at boundary without duplicating edge pixel.
/// Matches OpenCV's default border mode.
#[inline(always)]
pub fn reflect101(idx: isize, size: isize) -> isize {
    if idx < 0 {
        -idx
    } else if idx >= size {
        2 * size - 2 - idx
    } else {
        idx
    }
}

/// BORDER_REFLECT_101 with clamping for small sizes.
/// Handles the case where a single reflection is insufficient (e.g., idx=-2 with size=2).
#[inline]
pub fn reflect101_safe(idx: isize, size: isize) -> usize {
    if size <= 1 {
        return 0;
    }
    let mut i = idx;
    // Bring into range [-(size-1), 2*(size-1)] first
    let cycle = 2 * (size - 1);
    if i < 0 {
        i = -i;
    }
    if i >= cycle {
        i %= cycle;
    }
    if i >= size {
        i = cycle - i;
    }
    i as usize
}

pub fn solve_homography_4pt(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<[f64; 9]> {
    // Build 8×8 system A*x = b, where x = [c00, c01, c02, c10, c11, c12, c20, c21]
    // and c22 = 1 (assumed).
    let mut a = [0.0f64; 8 * 8];
    let mut b = [0.0f64; 8];

    for i in 0..4 {
        let (sx, sy) = (src[i].0 as f64, src[i].1 as f64);
        let (dx, dy) = (dst[i].0 as f64, dst[i].1 as f64);

        // Row i: x-equation → c00*sx + c01*sy + c02 - c20*sx*dx - c21*sy*dx = dx
        a[i * 8] = sx;
        a[i * 8 + 1] = sy;
        a[i * 8 + 2] = 1.0;
        // a[i*8+3..5] = 0 (c10, c11, c12 terms)
        a[i * 8 + 6] = -sx * dx;
        a[i * 8 + 7] = -sy * dx;
        b[i] = dx;

        // Row i+4: y-equation → c10*sx + c11*sy + c12 - c20*sx*dy - c21*sy*dy = dy
        // a[(i+4)*8+0..2] = 0 (c00, c01, c02 terms)
        a[(i + 4) * 8 + 3] = sx;
        a[(i + 4) * 8 + 4] = sy;
        a[(i + 4) * 8 + 5] = 1.0;
        a[(i + 4) * 8 + 6] = -sx * dy;
        a[(i + 4) * 8 + 7] = -sy * dy;
        b[i + 4] = dy;
    }

    // Solve via Gaussian elimination with partial pivoting (DECOMP_LU)
    let n = 8usize;
    let mut aug = [0.0f64; 8 * 9]; // augmented [A|b]
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular — would need SVD fallback
        }
        if max_row != col {
            for c in 0..(n + 1) {
                aug.swap(col * (n + 1) + c, max_row * (n + 1) + c);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for c in col..(n + 1) {
            aug[col * (n + 1) + c] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * (n + 1) + col];
            for c in col..(n + 1) {
                aug[row * (n + 1) + c] -= factor * aug[col * (n + 1) + c];
            }
        }
    }

    // Extract solution: x[i] = aug[i][n]
    // Pack into 3×3 with M[8] = 1.0 (c22 = 1)
    #[allow(clippy::identity_op, clippy::erasing_op)]
    Some([
        aug[0 * (n + 1) + n], // c00
        aug[1 * (n + 1) + n], // c01
        aug[2 * (n + 1) + n], // c02
        aug[3 * (n + 1) + n], // c10
        aug[4 * (n + 1) + n], // c11
        aug[5 * (n + 1) + n], // c12
        aug[6 * (n + 1) + n], // c20
        aug[7 * (n + 1) + n], // c21
        1.0,                  // c22
    ])
}

/// Solve a 3x3 perspective transform from 4 point correspondences.
///
/// Matches OpenCV's `getPerspectiveTransform` exactly:
/// - Formulates A*x = b with c22=1 assumption
/// - Solves via LU decomposition (Gaussian elimination with partial pivoting)
/// - Row ordering: x-equations first (rows 0–3), then y-equations (rows 4–7)
///
/// Returns the 3×3 matrix (row-major) mapping src → dst.
/// Reference: OpenCV 4.x modules/imgproc/src/imgwarp.cpp getPerspectiveTransform
/// Public wrapper for integration tests.
pub fn solve_homography_4pt_public(
    src: &[(f32, f32); 4],
    dst: &[(f32, f32); 4],
) -> Option<[f64; 9]> {
    solve_homography_4pt(src, dst)
}

/// Solve overdetermined linear system via normal equations (A^T A x = A^T b).
/// Uses Cholesky-like Gaussian elimination on the normal equations.
pub fn solve_least_squares(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // Form A^T A (n×n) and A^T b (n)
    let mut ata = vec![0.0f64; n * n];
    let mut atb = vec![0.0f64; n];

    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0f64;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
            ata[j * n + i] = sum;
        }
        let mut sum = 0.0f64;
        for k in 0..m {
            sum += a[k * n + i] * b[k];
        }
        atb[i] = sum;
    }

    // Gaussian elimination with partial pivoting
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = ata[i * n + j];
        }
        aug[i * (n + 1) + n] = atb[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }

    x
}

/// Convenience wrapper for non-tiled Gaussian vignette.
pub fn vignette_full(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
    x_inset: u32,
    y_inset: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = Rect::new(0, 0, info.width, info.height);
    let mut u = |_: Rect| Ok(pixels.to_vec());
    vignette(
        r,
        &mut u,
        info,
        &VignetteParams {
            sigma,
            x_inset,
            y_inset,
            full_width: info.width,
            full_height: info.height,
            tile_offset_x: 0,
            tile_offset_y: 0,
        },
    )
}

