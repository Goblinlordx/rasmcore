//! Morphology helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Build an anti-aliased elliptical mask via 8×8 supersampling at boundary pixels.
///
/// Interior pixels get 1.0, exterior get 0.0, boundary pixels (where the ellipse
/// edge crosses the pixel) get the fraction of 64 sub-pixel samples that fall
/// inside the ellipse. Sub-pixel samples span [col-0.5, col+0.5) × [row-0.5, row+0.5)
/// around the integer pixel coordinate, matching ImageMagick's rasterization
/// convention where the pixel center is at integer (col, row).
pub fn build_aa_ellipse_mask(w: usize, h: usize, cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<f64> {
    const N: usize = 8; // 8×8 = 64 sub-pixel samples
    let inv_rx = 1.0 / rx;
    let inv_ry = 1.0 / ry;

    let mut mask = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            // Check all four corners of the pixel [-0.5,+0.5] around center (col, row)
            let mut corners_inside = 0u8;
            for &(dx, dy) in &[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                let xn = (col as f64 + dx - cx) * inv_rx;
                let yn = (row as f64 + dy - cy) * inv_ry;
                if xn * xn + yn * yn <= 1.0 {
                    corners_inside += 1;
                }
            }

            if corners_inside == 4 {
                mask[row * w + col] = 1.0;
            } else if corners_inside == 0 {
                // All corners outside — check center in case the arc passes through
                let xn = (col as f64 - cx) * inv_rx;
                let yn = (row as f64 - cy) * inv_ry;
                if xn * xn + yn * yn <= 1.0 {
                    // Center inside, corners outside — supersample
                    let mut count = 0u32;
                    for sy in 0..N {
                        let py = (row as f64 - 0.5 + (sy as f64 + 0.5) / N as f64 - cy) * inv_ry;
                        let py2 = py * py;
                        for sx in 0..N {
                            let px =
                                (col as f64 - 0.5 + (sx as f64 + 0.5) / N as f64 - cx) * inv_rx;
                            if px * px + py2 <= 1.0 {
                                count += 1;
                            }
                        }
                    }
                    mask[row * w + col] = count as f64 / (N * N) as f64;
                }
            } else {
                // Mixed corners — boundary pixel, supersample
                let mut count = 0u32;
                for sy in 0..N {
                    let py = (row as f64 - 0.5 + (sy as f64 + 0.5) / N as f64 - cy) * inv_ry;
                    let py2 = py * py;
                    for sx in 0..N {
                        let px = (col as f64 - 0.5 + (sx as f64 + 0.5) / N as f64 - cx) * inv_rx;
                        if px * px + py2 <= 1.0 {
                            count += 1;
                        }
                    }
                }
                mask[row * w + col] = count as f64 / (N * N) as f64;
            }
        }
    }
    mask
}

/// Dilate: output pixel = maximum over structuring element neighborhood.
pub fn dilate(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    morph_op(pixels, info, ksize, shape, false)
}

/// Erode: output pixel = minimum over structuring element neighborhood.
///
/// For grayscale: per-pixel minimum. For RGB: per-channel minimum.
/// Matches OpenCV `cv2.erode` with `BORDER_REFLECT_101`.
pub fn erode(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    morph_op(pixels, info, ksize, shape, true)
}

/// Find the median value by scanning the histogram until cumulative count
/// reaches the target position.
#[inline]
pub fn find_median_in_hist(hist: &[u32; 256], target: usize) -> u8 {
    let mut cumulative = 0u32;
    for (val, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as usize > target {
            return val as u8;
        }
    }
    255
}

/// Find the value at the given rank position by scanning the histogram.
#[inline]
pub fn find_rank_in_hist(hist: &[u32; 256], target: usize) -> u8 {
    let mut cumulative = 0u32;
    for (val, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as usize > target {
            return val as u8;
        }
    }
    255
}

/// Generate a flat disc kernel of the given radius.
///
/// All pixels whose center falls within the circle of `radius` get weight 1.0.
/// Returns `(kernel, side_length)` where `side_length = 2 * radius + 1`.
pub fn make_disc_kernel(radius: u32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let r2 = (radius as f32 + 0.5) * (radius as f32 + 0.5);
    let mut kernel = vec![0.0f32; side * side];

    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            if dx * dx + dy * dy <= r2 {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a flat regular hexagonal kernel of the given radius.
///
/// Uses the pointy-top hexagon inscribed in a circle of `radius`. A pixel
/// is inside the hexagon if it satisfies all 6 half-plane constraints of the
/// regular hexagon with circumradius `radius + 0.5`.
pub fn make_hex_kernel(radius: u32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let cr = radius as f32 + 0.5; // circumradius
    let mut kernel = vec![0.0f32; side * side];

    // Regular hexagon (pointy-top): a point (dx, dy) is inside if
    // |dy| <= cr * sqrt(3)/2  AND  |dy| * 0.5 + |dx| * sqrt(3)/2 <= cr * sqrt(3)/2
    let h = cr * (3.0_f32.sqrt() / 2.0);

    for y in 0..side {
        for x in 0..side {
            let dx = (x as f32 - center).abs();
            let dy = (y as f32 - center).abs();
            if dy <= h && dy * 0.5 + dx * (3.0_f32.sqrt() / 2.0) <= h {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a regular polygon kernel with N sides, rotated by angle degrees.
pub fn make_polygon_kernel(radius: u32, sides: u32, rotation_deg: f32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let cr = radius as f32 + 0.5;
    let rot = rotation_deg.to_radians();
    let n = sides as f32;
    let mut kernel = vec![0.0f32; side * side];

    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > cr {
                continue;
            }
            // Check if point is inside the regular polygon
            // A point is inside if for each edge, it's on the interior side
            let angle = dy.atan2(dx) - rot;
            // Angular distance to nearest vertex
            let sector = (angle * n / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
            let half_angle = std::f32::consts::PI / n;
            // The polygon edge at this sector has distance: cr * cos(half_angle) / cos(...)
            let sector_angle = (sector - 0.5).abs() * 2.0 * half_angle;
            let edge_dist = cr * half_angle.cos() / sector_angle.cos().max(0.001);
            if dist <= edge_dist {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a structuring element as a boolean mask.
pub fn make_structuring_element(shape: MorphShape, kw: usize, kh: usize) -> Vec<bool> {
    let mut se = vec![false; kw * kh];
    let cx = kw / 2;
    let cy = kh / 2;
    match shape {
        MorphShape::Rect => {
            se.fill(true);
        }
        MorphShape::Cross => {
            for y in 0..kh {
                for x in 0..kw {
                    se[y * kw + x] = x == cx || y == cy;
                }
            }
        }
        MorphShape::Ellipse => {
            // Exact match with OpenCV getStructuringElement(MORPH_ELLIPSE).
            // From OpenCV source (morph.dispatch.cpp):
            //   r = ksize.height/2, c = ksize.width/2
            //   inv_r2 = 1.0/(r*r)
            //   for row i: j = c if dy==0, else round(sqrt(c²*(1 - dy²*inv_r2)))
            //   fill from c-j to c+j
            let r = (kh / 2) as f64;
            let c = (kw / 2) as f64;
            let inv_r2 = if r > 0.0 { 1.0 / (r * r) } else { 0.0 };
            for y in 0..kh {
                let dy = y as f64 - r;
                let j = if dy != 0.0 {
                    let t = c * c * (1.0 - dy * dy * inv_r2);
                    (t.max(0.0).sqrt()).round() as isize
                } else {
                    c as isize
                }
                .max(0) as usize;
                let x_start = cx.saturating_sub(j);
                let x_end = (cx + j + 1).min(kw);
                for x in x_start..x_end {
                    se[y * kw + x] = true;
                }
            }
        }
    }
    se
}

/// Histogram sliding-window median (Huang algorithm) for large radii.
///
/// Maintains a 256-bin histogram. When sliding horizontally, removes the
/// leftmost column and adds the rightmost column — O(2*diameter) per pixel
/// instead of O(diameter^2).
pub fn median_histogram(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    radius: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = radius as i32;
    let diameter = (2 * r + 1) as usize;
    let median_pos = (diameter * diameter) / 2;
    let mut out = vec![0u8; pixels.len()];

    for c in 0..channels {
        for y in 0..h {
            let mut hist = [0u32; 256];
            let mut _count = 0u32;

            // Initialize histogram for first window in this row
            for ky in -r..=r {
                let sy = reflect(y as i32 + ky, h);
                for kx in -r..=r {
                    let sx = reflect(kx, w);
                    hist[pixels[(sy * w + sx) * channels + c] as usize] += 1;
                    _count += 1;
                }
            }

            // Find median for first pixel
            out[y * w * channels + c] = find_median_in_hist(&hist, median_pos);

            // Slide right across the row
            for x in 1..w {
                // Remove leftmost column (x - r - 1)
                let old_x = x as i32 - r - 1;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(old_x, w);
                    let val = pixels[(sy * w + sx) * channels + c] as usize;
                    hist[val] -= 1;
                    _count -= 1;
                }

                // Add rightmost column (x + r)
                let new_x = x as i32 + r;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(new_x, w);
                    let val = pixels[(sy * w + sx) * channels + c] as usize;
                    hist[val] += 1;
                    _count += 1;
                }

                out[(y * w + x) * channels + c] = find_median_in_hist(&hist, median_pos);
            }
        }
    }
    Ok(out)
}

/// Sorting-based median for small radii (radius <= 2).
pub fn median_sort(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    radius: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = radius as i32;
    let window_size = ((2 * r + 1) * (2 * r + 1)) as usize;
    let median_pos = window_size / 2;
    let mut out = vec![0u8; pixels.len()];
    let mut window = Vec::with_capacity(window_size);

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                window.clear();
                for ky in -r..=r {
                    for kx in -r..=r {
                        let sy = reflect(y as i32 + ky, h);
                        let sx = reflect(x as i32 + kx, w);
                        window.push(pixels[(sy * w + sx) * channels + c]);
                    }
                }
                window.sort_unstable();
                out[(y * w + x) * channels + c] = window[median_pos];
            }
        }
    }
    Ok(out)
}

/// Black-hat: closing - input. Extracts small dark features.
pub fn morph_blackhat(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let closed = morph_close(pixels, info, ksize, shape)?;
    Ok(closed
        .iter()
        .zip(pixels.iter())
        .map(|(&c, &p)| c.saturating_sub(p))
        .collect())
}

/// Morphological closing: dilate then erode. Fills small dark holes.
pub fn morph_close(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let dilated = dilate(pixels, info, ksize, shape)?;
    erode(&dilated, info, ksize, shape)
}

/// Morphological gradient: dilate - erode. Highlights edges.
pub fn morph_gradient(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let dilated = dilate(pixels, info, ksize, shape)?;
    let eroded = erode(pixels, info, ksize, shape)?;
    Ok(dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d.saturating_sub(e))
        .collect())
}

/// Core morphological operation (erode=min, dilate=max).
pub fn morph_op(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
    is_erode: bool,
) -> Result<Vec<u8>, ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "morphology on {other:?} not supported"
            )));
        }
    };
    let w = info.width as usize;
    let h = info.height as usize;
    let kw = ksize as usize;
    let kh = ksize as usize;
    let kx = kw / 2;
    let ky = kh / 2;
    let se = make_structuring_element(shape, kw, kh);

    let process_ch = if info.format == PixelFormat::Rgba8 {
        3
    } else {
        ch
    };
    let mut out = pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            for c in 0..process_ch {
                let mut val = if is_erode { 255u8 } else { 0u8 };
                for ky2 in 0..kh {
                    for kx2 in 0..kw {
                        if !se[ky2 * kw + kx2] {
                            continue;
                        }
                        // Reflect101 boundary
                        let sy = reflect101(y as isize + ky2 as isize - ky as isize, h as isize)
                            as usize;
                        let sx = reflect101(x as isize + kx2 as isize - kx as isize, w as isize)
                            as usize;
                        let p = pixels[(sy * w + sx) * ch + c];
                        if is_erode {
                            val = val.min(p);
                        } else {
                            val = val.max(p);
                        }
                    }
                }
                out[(y * w + x) * ch + c] = val;
            }
        }
    }
    Ok(out)
}

/// Morphological opening: erode then dilate. Removes small bright spots.
pub fn morph_open(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let eroded = erode(pixels, info, ksize, shape)?;
    dilate(&eroded, info, ksize, shape)
}

pub fn morph_shape_from_u32(v: u32) -> MorphShape {
    match v {
        1 => MorphShape::Ellipse,
        2 => MorphShape::Cross,
        _ => MorphShape::Rect,
    }
}

/// Top-hat: input - opening. Extracts small bright features.
pub fn morph_tophat(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let opened = morph_open(pixels, info, ksize, shape)?;
    Ok(pixels
        .iter()
        .zip(opened.iter())
        .map(|(&p, &o)| p.saturating_sub(o))
        .collect())
}

