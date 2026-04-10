use super::{ImageError, ImageInfo, Rgb, find_nearest};

/// Floyd-Steinberg error-diffusion dithering.
///
/// Distributes quantization error to neighboring pixels:
///   * 7/16 → right (scan direction)
///   * 3/16 → below-left
///   * 5/16 → below
///   * 1/16 → below-right
///
/// Uses left-to-right scan (matching ImageMagick and Pillow).
pub fn dither_floyd_steinberg(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let n = w * h;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    // ImageMagick-compatible Floyd-Steinberg with:
    //   - Q16 scale (0–65535) in f64 for error accumulation
    //   - Serpentine scan (even rows L→R, odd rows R→L)
    //   - Two-row error buffer (current + previous), swapped per row
    //   - Error stored AFTER quantization, accumulated BEFORE next pixel
    const Q: f64 = 65535.0;
    let diffusion: f64 = 1.0;

    // Palette at Q16
    let pal_q16: Vec<[f64; 3]> = palette
        .iter()
        .map(|c| [c.r as f64 * 257.0, c.g as f64 * 257.0, c.b as f64 * 257.0])
        .collect();

    let mut out = vec![0u8; n * 3];

    // Two error rows: current (being computed) and previous (from last row)
    let mut current = vec![[0.0f64; 3]; w];
    let mut previous = vec![[0.0f64; 3]; w];

    for y in 0..h {
        // Serpentine: even rows L→R (v=1), odd rows R→L (v=-1)
        let forward = (y & 1) == 0;
        let v: i32 = if forward { 1 } else { -1 };

        for xi in 0..w {
            let u = if forward { xi } else { w - 1 - xi };

            // Start with original pixel at Q16
            let idx = y * w + u;
            let mut px = [
                pixels[idx * 3] as f64 * 257.0,
                pixels[idx * 3 + 1] as f64 * 257.0,
                pixels[idx * 3 + 2] as f64 * 257.0,
            ];

            // Add 7/16 error from previous pixel in current row
            if xi > 0 {
                let prev_u = (u as i32 - v) as usize;
                for c in 0..3 {
                    px[c] += 7.0 * diffusion * current[prev_u][c] / 16.0;
                }
            }

            // Add errors from previous row
            if y > 0 {
                // 1/16 from diagonal ahead
                if xi < w - 1 {
                    let next_u = (u as i32 + v) as usize;
                    for c in 0..3 {
                        px[c] += diffusion * previous[next_u][c] / 16.0;
                    }
                }
                // 5/16 from directly above
                for c in 0..3 {
                    px[c] += 5.0 * diffusion * previous[u][c] / 16.0;
                }
                // 3/16 from diagonal behind
                if xi > 0 {
                    let prev_u = (u as i32 - v) as usize;
                    for c in 0..3 {
                        px[c] += 3.0 * diffusion * previous[prev_u][c] / 16.0;
                    }
                }
            }

            // ClampPixel
            for px_val in &mut px {
                *px_val = px_val.clamp(0.0, Q);
            }

            // Nearest color at Q16
            let mut best_idx = 0;
            let mut best_dist = f64::MAX;
            for (i, c) in pal_q16.iter().enumerate() {
                let dr = px[0] - c[0];
                let dg = px[1] - c[1];
                let db = px[2] - c[2];
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            let nearest = palette[best_idx];
            out[idx * 3] = nearest.r;
            out[idx * 3 + 1] = nearest.g;
            out[idx * 3 + 2] = nearest.b;

            // Store error for this pixel
            for c in 0..3 {
                current[u][c] = px[c] - pal_q16[best_idx][c];
            }
        }

        // Swap rows: current becomes previous
        std::mem::swap(&mut current, &mut previous);
        for c in current.iter_mut() {
            *c = [0.0; 3];
        }
    }

    Ok(out)
}

/// Ordered dithering using a Bayer matrix.
///
/// `matrix_size` must be 2, 4, or 8.
pub fn dither_ordered(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
    matrix_size: usize,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let n = w * h;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let matrix = match matrix_size {
        2 => &BAYER_2X2[..],
        4 => &BAYER_4X4[..],
        8 => &BAYER_8X8[..],
        _ => {
            return Err(ImageError::InvalidParameters(
                "matrix_size must be 2, 4, or 8".into(),
            ));
        }
    };
    let ms = matrix_size;
    let scale = 255.0 / (ms * ms) as f32;

    let mut out = vec![0u8; n * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let threshold = matrix[(y % ms) * ms + (x % ms)] as f32 * scale - 128.0;

            let r = (pixels[idx * 3] as f32 + threshold).clamp(0.0, 255.0) as i32;
            let g = (pixels[idx * 3 + 1] as f32 + threshold).clamp(0.0, 255.0) as i32;
            let b = (pixels[idx * 3 + 2] as f32 + threshold).clamp(0.0, 255.0) as i32;

            let nearest = find_nearest(r, g, b, palette);
            out[idx * 3] = nearest.r;
            out[idx * 3 + 1] = nearest.g;
            out[idx * 3 + 2] = nearest.b;
        }
    }

    Ok(out)
}

// ─── Bayer Matrices ────────────────────────────────────────────────────────

#[rustfmt::skip]
const BAYER_2X2: [u8; 4] = [
    0, 2,
    3, 1,
];

#[rustfmt::skip]
const BAYER_4X4: [u8; 16] = [
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5,
];

#[rustfmt::skip]
const BAYER_8X8: [u8; 64] = [
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
];
