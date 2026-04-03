//! Filter: harris_corners (category: analysis)
//!
//! Harris corner detection — computes the Harris response at each pixel
//! and returns corners via non-maximum suppression.
//!
//! R = det(M) - k * trace(M)^2
//! where M is the structure tensor [Ixx, Ixy; Ixy, Iyy] smoothed over a block.
//!
//! Reference: Harris & Stephens 1988 "A combined corner and edge detector"

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Parameters for Harris corner detection.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "harris_corners", category = "analysis", group = "analysis", variant = "harris_corners", reference = "Harris & Stephens 1988")]
pub struct HarrisCornersParams {
    /// Harris sensitivity parameter (typically 0.04-0.06)
    #[param(min = 0.01, max = 0.3, step = 0.01, default = 0.04)]
    pub k: f32,
    /// Response threshold (corners with R > threshold are kept)
    #[param(min = 0.0, max = 1000000.0, step = 100.0, default = 10000.0)]
    pub threshold: f32,
    /// Structure tensor smoothing window size (must be odd, typically 3 or 5)
    #[param(min = 3, max = 31, step = 2, default = 3)]
    pub block_size: u32,
    /// Non-maximum suppression window radius
    #[param(min = 1, max = 15, step = 1, default = 1)]
    pub nms_radius: u32,
}

/// A detected corner point.
#[derive(Debug, Clone, Copy)]
pub struct CornerPoint {
    pub x: u32,
    pub y: u32,
    pub response: f32,
}

/// Compute Harris corner response map and detect corners.
///
/// Returns a list of corner points sorted by descending response strength.
pub fn harris_corners(
    pixels: &[u8],
    info: &ImageInfo,
    k: f32,
    threshold: f32,
    block_size: u32,
    nms_radius: u32,
) -> Result<Vec<CornerPoint>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    // Convert to grayscale
    let gray = to_grayscale(pixels, channels);

    // Compute Sobel gradients Gx, Gy (signed f32, not magnitude)
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;

    let mut ix = vec![0.0f32; w * h];
    let mut iy = vec![0.0f32; w * h];

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

            // Sobel Gx: [[-1,0,1],[-2,0,2],[-1,0,1]]
            ix[y * w + x] = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            // Sobel Gy: [[-1,-2,-1],[0,0,0],[1,2,1]]
            iy[y * w + x] = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;
        }
    }

    // Compute structure tensor products
    let mut ixx: Vec<f32> = ix.iter().map(|&x| x * x).collect();
    let mut ixy: Vec<f32> = ix.iter().zip(iy.iter()).map(|(&x, &y)| x * y).collect();
    let mut iyy: Vec<f32> = iy.iter().map(|&y| y * y).collect();

    // Box-filter the structure tensor components over block_size window
    let half = block_size as usize / 2;
    ixx = box_filter_f32(&ixx, w, h, half);
    ixy = box_filter_f32(&ixy, w, h, half);
    iyy = box_filter_f32(&iyy, w, h, half);

    // Compute Harris response: R = det(M) - k * trace(M)^2
    let mut response = vec![0.0f32; w * h];
    for i in 0..w * h {
        let det = ixx[i] * iyy[i] - ixy[i] * ixy[i];
        let trace = ixx[i] + iyy[i];
        response[i] = det - k * trace * trace;
    }

    // Non-maximum suppression
    let nms = nms_radius as usize;
    let mut corners = Vec::new();
    for y in nms..h.saturating_sub(nms) {
        for x in nms..w.saturating_sub(nms) {
            let r = response[y * w + x];
            if r <= threshold {
                continue;
            }
            // Check if this pixel is the local maximum in nms window
            let mut is_max = true;
            'nms: for ny in y.saturating_sub(nms)..=(y + nms).min(h - 1) {
                for nx in x.saturating_sub(nms)..=(x + nms).min(w - 1) {
                    if (ny, nx) != (y, x) && response[ny * w + nx] >= r {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                corners.push(CornerPoint {
                    x: x as u32,
                    y: y as u32,
                    response: r,
                });
            }
        }
    }

    // Sort by descending response
    corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));

    Ok(corners)
}

/// Simple box filter for f32 data — O(1) per pixel via running sums.
fn box_filter_f32(data: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return data.to_vec();
    }
    let mut temp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];

    // Horizontal pass
    for y in 0..h {
        let row = y * w;
        let mut sum = 0.0f32;
        // Init window
        for x in 0..=radius.min(w - 1) {
            sum += data[row + x];
        }
        for x in 0..w {
            temp[row + x] = sum;
            let add = x + radius + 1;
            let rem = x.wrapping_sub(radius);
            if add < w {
                sum += data[row + add];
            }
            if rem < w {
                sum -= data[row + rem];
            }
        }
    }

    // Vertical pass
    for x in 0..w {
        let mut sum = 0.0f32;
        for y in 0..=radius.min(h - 1) {
            sum += temp[y * w + x];
        }
        for y in 0..h {
            out[y * w + x] = sum;
            let add = y + radius + 1;
            let rem = y.wrapping_sub(radius);
            if add < h {
                sum += temp[add * w + x];
            }
            if rem < h {
                sum -= temp[rem * w + x];
            }
        }
    }

    out
}

/// Registered filter — renders Harris corners as white dots on black Gray8 canvas.


impl CpuFilter for HarrisCornersParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };

    let corners = harris_corners(
        &pixels,
        info,
        self.k,
        self.threshold,
        self.block_size,
        self.nms_radius,
    )?;

    // Render corners as white 3x3 crosses on black canvas
    let w = info.width as usize;
    let h = info.height as usize;
    let mut out = vec![0u8; w * h];
    for c in &corners {
        let cx = c.x as usize;
        let cy = c.y as usize;
        // Center pixel
        out[cy * w + cx] = 255;
        // Cross arms (1 pixel each direction)
        if cx > 0 { out[cy * w + cx - 1] = 255; }
        if cx + 1 < w { out[cy * w + cx + 1] = 255; }
        if cy > 0 { out[(cy - 1) * w + cx] = 255; }
        if cy + 1 < h { out[(cy + 1) * w + cx] = 255; }
    }
    Ok(out)
}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rectangle(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> (Vec<u8>, ImageInfo) {
        let mut pixels = vec![0u8; (w * h) as usize];
        for y in y0..=y1 {
            for x in x0..=x1 {
                pixels[(y * w + x) as usize] = 255;
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn detects_rectangle_corners() {
        // White rectangle on black background — should detect 4 corners
        let (pixels, info) = make_rectangle(64, 64, 15, 15, 48, 48);
        let corners = harris_corners(&pixels, &info, 0.04, 100.0, 3, 3).unwrap();

        // Should find corners near the 4 rectangle vertices
        assert!(
            corners.len() >= 4,
            "Expected at least 4 corners, got {}",
            corners.len()
        );

        // Verify corners are near the expected positions (within 5 pixels)
        let expected = [(15, 15), (48, 15), (15, 48), (48, 48)];
        for (ex, ey) in &expected {
            let found = corners.iter().any(|c| {
                let dx = (c.x as i32 - *ex as i32).unsigned_abs();
                let dy = (c.y as i32 - *ey as i32).unsigned_abs();
                dx <= 5 && dy <= 5
            });
            assert!(found, "Expected corner near ({ex}, {ey}) not found");
        }
        eprintln!("  harris rectangle: {} corners detected", corners.len());
    }

    #[test]
    fn no_corners_on_flat_image() {
        let pixels = vec![128u8; 64 * 64];
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let corners = harris_corners(&pixels, &info, 0.04, 100.0, 3, 1).unwrap();
        assert!(corners.is_empty(), "Flat image should have no corners");
    }

    #[test]
    fn corners_sorted_by_response() {
        let (pixels, info) = make_rectangle(64, 64, 10, 10, 50, 50);
        let corners = harris_corners(&pixels, &info, 0.04, 100.0, 3, 1).unwrap();
        for w in corners.windows(2) {
            assert!(w[0].response >= w[1].response, "Corners not sorted by response");
        }
    }
}
