//! Edge helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Canny edge detection (internal — returns raw Gray8 bytes).
pub fn canny(pixels: &[u8], info: &ImageInfo, config: &CannyParams) -> Result<Vec<u8>, ImageError> {
    let low_threshold = config.low_threshold;
    let high_threshold = config.high_threshold;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| canny(p8, i8, config));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    // Step 1: Convert to grayscale
    let gray = to_grayscale(pixels, channels);

    // Step 2: Sobel gradient magnitude and direction
    // Note: no internal blur — matches OpenCV cv2.Canny behavior.
    // Caller should pre-blur if desired (e.g., GaussianBlur then Canny).
    let mut magnitude = vec![0.0f32; w * h];
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p01 = padded[r0 + x + 1] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p21 = padded[r2 + x + 1] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            // L1 gradient magnitude (matches OpenCV default: |gx| + |gy|)
            magnitude[y * w + x] = gx.abs() + gy.abs();
        }
    }

    // Step 4: Non-maximum suppression (matches OpenCV's tangent-ratio method)
    //
    // OpenCV uses TG22 = tan(22.5°) * 2^15 = 13573 to classify angles into
    // 3 bins without atan2. Asymmetric comparison (> on one side, >= on other)
    // ensures consistent tie-breaking.
    let mut nms = vec![0u8; w * h]; // 0=suppressed, 1=weak candidate, 2=strong edge

    // Recompute Sobel components for NMS direction (need signed gx, gy)
    let padded_nms = pad_reflect(&gray, w, h, 1, 1);
    let pw_nms = w + 2;

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let mag = magnitude[y * w + x];
            if mag <= low_threshold {
                continue;
            }

            let r0 = y * pw_nms;
            let r1 = (y + 1) * pw_nms;
            let r2 = (y + 2) * pw_nms;
            let p00 = padded_nms[r0 + x] as f32;
            let p01 = padded_nms[r0 + x + 1] as f32;
            let p02 = padded_nms[r0 + x + 2] as f32;
            let p10 = padded_nms[r1 + x] as f32;
            let p12 = padded_nms[r1 + x + 2] as f32;
            let p20 = padded_nms[r2 + x] as f32;
            let p21 = padded_nms[r2 + x + 1] as f32;
            let p22 = padded_nms[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            // OpenCV tangent-ratio NMS: TG22 = tan(22.5°) * 2^15 = 13573
            let ax = gx.abs();
            let ay = gy.abs();
            let tg22x = ax * 13573.0;
            let y_shifted = ay * 32768.0; // ay << 15

            let is_max = if y_shifted < tg22x {
                // Near-horizontal edge: compare left/right
                mag > magnitude[y * w + x - 1] && mag >= magnitude[y * w + x + 1]
            } else {
                let tg67x = tg22x + ax * 65536.0; // tg22x + (ax << 16)
                if y_shifted > tg67x {
                    // Near-vertical edge: compare up/down
                    mag > magnitude[(y - 1) * w + x] && mag >= magnitude[(y + 1) * w + x]
                } else {
                    // Diagonal edge: compare diagonal neighbors
                    let s: i32 = if (gx < 0.0) != (gy < 0.0) { -1 } else { 1 };
                    mag > magnitude[(y - 1) * w + (x as i32 - s) as usize]
                        && mag > magnitude[(y + 1) * w + (x as i32 + s) as usize]
                }
            };

            if is_max {
                if mag >= high_threshold {
                    nms[y * w + x] = 2; // strong edge
                } else {
                    nms[y * w + x] = 1; // weak candidate
                }
            }
        }
    }

    // Step 5: Hysteresis thresholding (stack-based BFS, matches OpenCV)
    let mut out = vec![0u8; w * h];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    // Seed stack with strong edges
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if nms[y * w + x] == 2 {
                out[y * w + x] = 255;
                stack.push((x, y));
            }
        }
    }

    // BFS: extend strong edges to connected weak edges
    while let Some((x, y)) = stack.pop() {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                if nx < w && ny < h && nms[ny * w + nx] == 1 && out[ny * w + nx] == 0 {
                    out[ny * w + nx] = 255;
                    nms[ny * w + nx] = 2; // mark as visited
                    stack.push((nx, ny));
                }
            }
        }
    }

    Ok(out)
}

pub fn emboss_impl(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    // Standard emboss kernel: directional highlight along the diagonal
    #[rustfmt::skip]
    let kernel: [f32; 9] = [
        -2.0, -1.0,  0.0,
        -1.0,  1.0,  1.0,
         0.0,  1.0,  2.0,
    ];
    {
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        convolve(
            r,
            &mut u,
            info,
            &kernel,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 1.0,
            },
        )
    }
}

/// Laplacian edge detection (internal — returns raw Gray8 bytes).
pub fn laplacian(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, laplacian);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let gray = to_grayscale(pixels, channels);
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p11 = padded[r1 + x + 1] as f32;
            let p20 = padded[r2 + x] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            // OpenCV Laplacian ksize=3: kernel [2,0,2; 0,-8,0; 2,0,2]
            let lap = 2.0 * p00 + 2.0 * p02 - 8.0 * p11 + 2.0 * p20 + 2.0 * p22;
            out[y * w + x] = lap.abs().min(255.0) as u8;
        }
    }
    Ok(out)
}

/// Scharr edge detection (internal — returns raw Gray8 bytes).
pub fn scharr(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, scharr);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let gray = to_grayscale(pixels, channels);
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p01 = padded[r0 + x + 1] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p21 = padded[r2 + x + 1] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            // Scharr Gx = [[-3,0,3],[-10,0,10],[-3,0,3]]
            let gx = -3.0 * p00 + 3.0 * p02 - 10.0 * p10 + 10.0 * p12 - 3.0 * p20 + 3.0 * p22;
            // Scharr Gy = [[-3,-10,-3],[0,0,0],[3,10,3]]
            let gy = -3.0 * p00 - 10.0 * p01 - 3.0 * p02 + 3.0 * p20 + 10.0 * p21 + 3.0 * p22;

            out[y * w + x] = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
        }
    }
    Ok(out)
}

/// Sobel edge detection (internal — returns raw Gray8 bytes).
pub fn sobel(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, sobel);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    let gray = to_grayscale(pixels, channels);

    // Pad with 1-pixel reflected border to eliminate boundary checks
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw; // row above (in padded coords, offset by pad=1 → y+1-1 = y)
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            // Direct Sobel — unrolled 3x3, no loop
            // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
            let p00 = padded[r0 + x] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;

            // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
            let p01 = padded[r0 + x + 1] as f32;
            let p21 = padded[r2 + x + 1] as f32;

            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            out[y * w + x] = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
        }
    }
    Ok(out)
}

