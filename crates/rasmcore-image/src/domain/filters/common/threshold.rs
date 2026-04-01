//! Threshold helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Adaptive threshold — per-pixel threshold computed from local neighborhood.
///
/// For each pixel, the threshold is the (mean or Gaussian-weighted mean) of a
/// block_size × block_size neighborhood, minus constant C.
///
/// Reference: OpenCV cv2.adaptiveThreshold.
pub fn adaptive_threshold(
    pixels: &[u8],
    info: &ImageInfo,
    max_value: u8,
    method: AdaptiveMethod,
    block_size: u32,
    c: f64,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "adaptive threshold requires Gray8 input".into(),
        ));
    }
    if block_size.is_multiple_of(2) || block_size < 3 {
        return Err(ImageError::InvalidParameters(
            "block_size must be odd and >= 3".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let r = (block_size / 2) as usize;

    match method {
        AdaptiveMethod::Mean => {
            // Integer box mean via integral image — BORDER_REPLICATE (matches OpenCV exactly).
            // OpenCV adaptiveThreshold uses: boxFilter(src, mean, CV_8U, ..., BORDER_REPLICATE)
            // then: pixel > (mean - C) in integer arithmetic.
            let box_mean = box_mean_u8_replicate(pixels, w, h, r);
            let c_int = c.round() as i16;
            let mut result = vec![0u8; w * h];
            for i in 0..(w * h) {
                // Signed comparison — threshold can be negative (OpenCV does NOT clamp to 0)
                let thresh = box_mean[i] as i16 - c_int;
                result[i] = if (pixels[i] as i16) > thresh {
                    max_value
                } else {
                    0
                };
            }
            Ok(result)
        }
        AdaptiveMethod::Gaussian => {
            // Gaussian-weighted mean — use separable Gaussian blur
            let sigma = 0.3 * ((block_size as f64 - 1.0) * 0.5 - 1.0) + 0.8;
            let local_mean = gaussian_blur_f64(pixels, w, h, block_size as usize, sigma);
            let mut result = vec![0u8; w * h];
            for i in 0..(w * h) {
                let thresh = local_mean[i] - c;
                result[i] = if (pixels[i] as f64) > thresh {
                    max_value
                } else {
                    0
                };
            }
            Ok(result)
        }
    }
}

/// Compute Otsu's optimal threshold for a grayscale image.
///
/// Maximizes inter-class variance between foreground and background.
/// Returns the threshold value [0, 255].
///
/// Reference: OpenCV cv2.threshold(..., THRESH_OTSU).
pub fn otsu_threshold(pixels: &[u8], info: &ImageInfo) -> Result<u8, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "otsu requires Gray8 input".into(),
        ));
    }

    let n = pixels.len() as f64;
    if n == 0.0 {
        return Ok(0);
    }

    // Build histogram
    let mut hist = [0u32; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }

    // Compute total mean
    let mut total_sum = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        total_sum += i as f64 * h as f64;
    }

    let mut best_thresh = 0u8;
    let mut best_var = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for (t, &ht) in hist.iter().enumerate() {
        w0 += ht as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = n - w0;
        if w1 == 0.0 {
            break;
        }

        sum0 += t as f64 * ht as f64;
        let mu0 = sum0 / w0;
        let mu1 = (total_sum - sum0) / w1;
        let between_var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if between_var > best_var {
            best_var = between_var;
            best_thresh = t as u8;
        }
    }

    Ok(best_thresh)
}

/// Compute triangle threshold for a grayscale image.
///
/// Draws a line from the histogram peak to the farthest end,
/// then finds the bin with maximum perpendicular distance to that line.
///
/// Reference: OpenCV cv2.threshold(..., THRESH_TRIANGLE).
pub fn triangle_threshold(pixels: &[u8], info: &ImageInfo) -> Result<u8, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "triangle requires Gray8 input".into(),
        ));
    }

    let mut hist = [0u32; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }

    // Find histogram bounds and peak
    let mut left = 0usize;
    let mut right = 255usize;
    while left < 256 && hist[left] == 0 {
        left += 1;
    }
    while right > 0 && hist[right] == 0 {
        right -= 1;
    }
    if left >= right {
        return Ok(left as u8);
    }

    let mut peak = left;
    for i in left..=right {
        if hist[i] > hist[peak] {
            peak = i;
        }
    }

    // Determine which side is longer — the line goes from peak to the far end
    let flip = (peak - left) < (right - peak);
    let (a, b) = if flip { (peak, right) } else { (left, peak) };

    // Line from (a, hist[a]) to (b, hist[b])
    let ax = a as f64;
    let ay = hist[a] as f64;
    let bx = b as f64;
    let by = hist[b] as f64;

    // Find bin with max perpendicular distance to the line
    let line_len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    if line_len < 1e-10 {
        return Ok(peak as u8);
    }

    let mut best_dist = 0.0f64;
    let mut best_t = a;
    let range = if a < b { a..=b } else { b..=a };
    for t in range {
        let px = t as f64;
        let py = hist[t] as f64;
        // Perpendicular distance from point to line
        let dist = ((by - ay) * px - (bx - ax) * py + bx * ay - by * ax).abs() / line_len;
        if dist > best_dist {
            best_dist = dist;
            best_t = t;
        }
    }

    Ok(best_t as u8)
}

