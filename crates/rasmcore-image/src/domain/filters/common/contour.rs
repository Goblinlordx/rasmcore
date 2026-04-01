//! Contour helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Approximate a contour polygon using the Douglas-Peucker algorithm.
///
/// Simplifies the contour by removing points within `epsilon` distance
/// of the line between endpoints. Larger epsilon = fewer points.
///
/// Reference: Douglas & Peucker (1973)
pub fn approx_poly(contour: &[(i32, i32)], epsilon: f64) -> Vec<(i32, i32)> {
    if contour.len() <= 2 || epsilon <= 0.0 {
        return contour.to_vec();
    }
    let mut keep = vec![false; contour.len()];
    keep[0] = true;
    keep[contour.len() - 1] = true;
    douglas_peucker(contour, 0, contour.len() - 1, epsilon, &mut keep);
    contour
        .iter()
        .zip(keep.iter())
        .filter_map(|(&pt, &k)| if k { Some(pt) } else { None })
        .collect()
}

/// Compute the axis-aligned bounding rectangle of a contour.
///
/// Returns (x, y, width, height) where (x, y) is the top-left corner.
pub fn bounding_rect(contour: &[(i32, i32)]) -> (i32, i32, i32, i32) {
    if contour.is_empty() {
        return (0, 0, 0, 0);
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in contour {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
}

/// Compute the area of a contour using the shoelace formula.
///
/// Returns the absolute area. For a closed polygon, this gives the
/// enclosed area. Matches `cv2.contourArea`.
pub fn contour_area(contour: &[(i32, i32)]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }
    let mut area: f64 = 0.0;
    let n = contour.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += contour[i].0 as f64 * contour[j].1 as f64;
        area -= contour[j].0 as f64 * contour[i].1 as f64;
    }
    area.abs() / 2.0
}

/// Compute the perimeter (arc length) of a contour.
///
/// Sums the Euclidean distances between consecutive points.
/// If `closed` is true, also adds the distance from last to first point.
pub fn contour_perimeter(contour: &[(i32, i32)], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    let mut perim: f64 = 0.0;
    for i in 0..contour.len() - 1 {
        let dx = (contour[i + 1].0 - contour[i].0) as f64;
        let dy = (contour[i + 1].1 - contour[i].1) as f64;
        perim += (dx * dx + dy * dy).sqrt();
    }
    if closed {
        let dx = (contour[0].0 - contour.last().unwrap().0) as f64;
        let dy = (contour[0].1 - contour.last().unwrap().1) as f64;
        perim += (dx * dx + dy * dy).sqrt();
    }
    perim
}

pub fn douglas_peucker(points: &[(i32, i32)], start: usize, end: usize, epsilon: f64, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let (sx, sy) = (points[start].0 as f64, points[start].1 as f64);
    let (ex, ey) = (points[end].0 as f64, points[end].1 as f64);
    let line_len = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt();

    let mut max_dist = 0.0;
    let mut max_idx = start;

    for (i, &(ptx, pty)) in points.iter().enumerate().skip(start + 1).take(end - start - 1) {
        let (px, py) = (ptx as f64, pty as f64);
        let dist = if line_len < 1e-10 {
            ((px - sx).powi(2) + (py - sy).powi(2)).sqrt()
        } else {
            ((ey - sy) * px - (ex - sx) * py + ex * sy - ey * sx).abs() / line_len
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        keep[max_idx] = true;
        douglas_peucker(points, start, max_idx, epsilon, keep);
        douglas_peucker(points, max_idx, end, epsilon, keep);
    }
}

pub fn line_intersection(l1: &LineSegment, l2: &LineSegment) -> Option<(f32, f32)> {
    let d1x = (l1.x2 - l1.x1) as f32;
    let d1y = (l1.y2 - l1.y1) as f32;
    let d2x = (l2.x2 - l2.x1) as f32;
    let d2y = (l2.y2 - l2.y1) as f32;

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-6 {
        return None;
    }

    let t = ((l2.x1 - l1.x1) as f32 * d2y - (l2.y1 - l1.y1) as f32 * d2x) / denom;
    Some((l1.x1 as f32 + t * d1x, l1.y1 as f32 + t * d1y))
}

/// Trace a single border starting at (start_x, start_y) using Moore boundary tracing.
pub fn trace_border(
    img: &mut [i32],
    pw: usize,
    start_x: i32,
    start_y: i32,
    dx: &[i32; 8],
    dy: &[i32; 8],
    nbd: i32,
) -> Vec<(i32, i32)> {
    let mut contour = Vec::new();
    contour.push((start_x, start_y));

    // Find the first foreground neighbor going clockwise from the west direction
    let start_dir = 4; // start searching from west (the background side)
    let mut found_dir = None;
    for i in 0..8 {
        let d = (start_dir + i) % 8;
        let nx = start_x + dx[d];
        let ny = start_y + dy[d];
        let ni = ny as usize * pw + nx as usize;
        if img[ni] != 0 {
            found_dir = Some(d);
            break;
        }
    }

    let Some(mut dir) = found_dir else {
        // Isolated pixel
        img[start_y as usize * pw + start_x as usize] = -nbd;
        return contour;
    };

    let mut cx = start_x + dx[dir];
    let mut cy = start_y + dy[dir];

    if cx == start_x && cy == start_y {
        // Single pixel contour
        img[start_y as usize * pw + start_x as usize] = -nbd;
        return contour;
    }

    // Mark the start pixel
    img[start_y as usize * pw + start_x as usize] = nbd;

    let second_x = cx;
    let second_y = cy;

    loop {
        contour.push((cx, cy));
        img[cy as usize * pw + cx as usize] = nbd;

        // Search clockwise from (dir + 5) % 8 = opposite of arrival + 1
        let search_start = (dir + 5) % 8;
        let mut next_dir = None;
        for i in 0..8 {
            let d = (search_start + i) % 8;
            let nx = cx + dx[d];
            let ny = cy + dy[d];
            let ni = ny as usize * pw + nx as usize;
            if img[ni] != 0 {
                next_dir = Some(d);
                break;
            }
        }

        let Some(nd) = next_dir else {
            break; // shouldn't happen for a valid contour
        };

        let nx = cx + dx[nd];
        let ny = cy + dy[nd];
        dir = nd;

        // Termination: we've returned to start and the next step is the second point
        if nx == start_x && ny == start_y && cx == second_x && cy == second_y {
            break;
        }
        // Also terminate if we've returned to start
        if nx == start_x && ny == start_y {
            break;
        }

        cx = nx;
        cy = ny;

        // Safety: prevent infinite loops
        if contour.len() > (pw * img.len() / pw) {
            break;
        }
    }

    contour
}

/// Estimate vanishing point from line segments using weighted median of
/// pairwise intersections, weighted by product of segment lengths.
pub fn estimate_vanishing_point(lines: &[(LineSegment, f32)]) -> Option<(f32, f32)> {
    if lines.len() < 2 {
        return None;
    }

    let mut intersections: Vec<(f32, f32, f32)> = Vec::new();

    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            let (l1, w1) = &lines[i];
            let (l2, w2) = &lines[j];
            if let Some((ix, iy)) = line_intersection(l1, l2) {
                intersections.push((ix, iy, w1 * w2));
            }
        }
    }

    if intersections.is_empty() {
        return None;
    }

    let total_weight: f32 = intersections.iter().map(|&(_, _, w)| w).sum();
    if total_weight <= 0.0 {
        return None;
    }

    let mut sorted_x = intersections.clone();
    sorted_x.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let median_x = weighted_median_val(&sorted_x, total_weight, |p| p.0);

    let mut sorted_y = intersections;
    sorted_y.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let median_y = weighted_median_val(&sorted_y, total_weight, |p| p.1);

    Some((median_x, median_y))
}

/// Progressive Probabilistic Hough Transform (PPHT) on a binary edge image.
///
/// Implements the exact algorithm from OpenCV's `HoughLinesProbabilistic`
/// (Matas et al., 2000). Key properties:
/// - Random pixel processing order via seeded PRNG (deterministic given seed)
/// - Incremental accumulator with vote decrement on consumed pixels
/// - Fixed-point line walking with gap tolerance
/// - Chebyshev (L∞) length check for min_length
///
/// Parameters match cv2.HoughLinesP:
/// - `rho`: distance resolution of the accumulator in pixels
/// - `theta`: angle resolution in radians
/// - `threshold`: accumulator threshold — only lines with votes > threshold are returned
/// - `min_line_length`: minimum line segment length (Chebyshev / L∞ metric)
/// - `max_line_gap`: maximum gap between points on the same line segment
/// - `seed`: PRNG seed for deterministic output (use 0 for OpenCV default)
///
/// Reference: OpenCV 4.x modules/imgproc/src/hough.cpp HoughLinesProbabilistic
#[allow(clippy::too_many_arguments)]
pub fn hough_lines_p(
    pixels: &[u8],
    info: &ImageInfo,
    rho: f32,
    theta: f32,
    threshold: i32,
    min_line_length: i32,
    max_line_gap: i32,
    seed: u64,
) -> Result<Vec<LineSegment>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::InvalidInput(
            "hough_lines_p requires Gray8 binary edge map".into(),
        ));
    }

    let width = info.width as i32;
    let height = info.height as i32;
    let irho = 1.0f32 / rho;

    // Compute numangle: discrete theta bins from [0, pi)
    let mut numangle = ((std::f64::consts::PI / theta as f64).floor() as i32) + 1;
    if numangle > 1 {
        let last_theta = (numangle - 1) as f64 * theta as f64;
        if (std::f64::consts::PI - last_theta).abs() < theta as f64 / 2.0 {
            numangle -= 1;
        }
    }

    // numrho: rho bins covering [-(w+h), +(w+h)]
    let numrho = (((width + height) * 2 + 1) as f64 / rho as f64).round() as i32;

    // Precompute trig table: trigtab[n*2] = cos(n*theta)*irho, trigtab[n*2+1] = sin(n*theta)*irho
    let mut trigtab = vec![0.0f32; (numangle * 2) as usize];
    for n in 0..numangle {
        let ang = n as f64 * theta as f64;
        trigtab[(n * 2) as usize] = (ang.cos() * irho as f64) as f32;
        trigtab[(n * 2 + 1) as usize] = (ang.sin() * irho as f64) as f32;
    }

    // Accumulator: numangle rows × numrho cols
    let mut accum = vec![0i32; (numangle * numrho) as usize];

    // Mask: tracks live edge pixels
    let mut mask = vec![0u8; (width * height) as usize];

    // Collect non-zero pixels
    let mut nzloc: Vec<(i32, i32)> = Vec::new(); // (x, y)
    for y in 0..height {
        for x in 0..width {
            if pixels[(y * width + x) as usize] != 0 {
                mask[(y * width + x) as usize] = 1;
                nzloc.push((x, y));
            }
        }
    }
    let mut count = nzloc.len();

    let mut rng = CvRng::new(if seed == 0 { u64::MAX } else { seed });
    let mut lines = Vec::new();
    let line_walk_shift: i32 = 16;

    // Main PPHT loop: process pixels in random order
    while count > 0 {
        // Pick random pixel, swap-remove
        let idx = rng.uniform(count as u32) as usize;
        let (j, i) = nzloc[idx]; // j=x, i=y
        nzloc[idx] = nzloc[count - 1];
        count -= 1;

        // Skip if already consumed
        if mask[(i * width + j) as usize] == 0 {
            continue;
        }

        // Vote: increment accumulator for all angle bins
        let mut max_val = threshold - 1;
        let mut max_n: i32 = 0;
        for n in 0..numangle {
            let r = (j as f32 * trigtab[(n * 2) as usize]
                + i as f32 * trigtab[(n * 2 + 1) as usize])
                .round() as i32
                + (numrho - 1) / 2;
            if r >= 0 && r < numrho {
                let idx_acc = (n * numrho + r) as usize;
                accum[idx_acc] += 1;
                if accum[idx_acc] > max_val {
                    max_val = accum[idx_acc];
                    max_n = n;
                }
            }
        }

        // If no bin reached threshold, continue
        if max_val < threshold {
            continue;
        }

        // Line walking: extract segment along the peak (rho, theta)
        // Line direction perpendicular to normal: a = -sin*irho, b = cos*irho
        let a = -trigtab[(max_n * 2 + 1) as usize];
        let b = trigtab[(max_n * 2) as usize];

        let mut x0 = j as i64;
        let mut y0 = i as i64;
        let dx0: i64;
        let dy0: i64;
        let xflag: bool;

        if a.abs() > b.abs() {
            xflag = true;
            dx0 = if a > 0.0 { 1 } else { -1 };
            dy0 = (b as f64 * (1i64 << line_walk_shift) as f64 / a.abs() as f64).round() as i64;
            y0 = (y0 << line_walk_shift) + (1i64 << (line_walk_shift - 1));
        } else {
            xflag = false;
            dy0 = if b > 0.0 { 1 } else { -1 };
            dx0 = (a as f64 * (1i64 << line_walk_shift) as f64 / b.abs() as f64).round() as i64;
            x0 = (x0 << line_walk_shift) + (1i64 << (line_walk_shift - 1));
        }

        // Walk in both directions to find segment endpoints
        let mut line_end = [(0i32, 0i32); 2];
        #[allow(clippy::needless_range_loop)]
        for k in 0..2usize {
            let mut gap = 0i32;
            let (mut x, mut y) = (x0, y0);
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };

            loop {
                let (j1, i1) = if xflag {
                    (x as i32, (y >> line_walk_shift) as i32)
                } else {
                    ((x >> line_walk_shift) as i32, y as i32)
                };

                if j1 < 0 || j1 >= width || i1 < 0 || i1 >= height {
                    break;
                }

                if mask[(i1 * width + j1) as usize] != 0 {
                    gap = 0;
                    line_end[k] = (j1, i1);
                } else {
                    gap += 1;
                    if gap > max_line_gap {
                        break;
                    }
                }

                x += dx;
                y += dy;
            }
        }

        // Length check: Chebyshev distance (L∞), matching OpenCV
        let good_line = (line_end[1].0 - line_end[0].0).abs() >= min_line_length
            || (line_end[1].1 - line_end[0].1).abs() >= min_line_length;

        // Second walk: consume pixels along the line, decrement votes if good
        #[allow(clippy::needless_range_loop)]
        for k in 0..2usize {
            let (mut x, mut y) = (x0, y0);
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };

            loop {
                let (j1, i1) = if xflag {
                    (x as i32, (y >> line_walk_shift) as i32)
                } else {
                    ((x >> line_walk_shift) as i32, y as i32)
                };

                if j1 < 0 || j1 >= width || i1 < 0 || i1 >= height {
                    break;
                }

                let midx = (i1 * width + j1) as usize;
                if mask[midx] != 0 {
                    if good_line {
                        // Decrement accumulator for ALL angle bins this pixel voted for
                        for n in 0..numangle {
                            let r = (j1 as f32 * trigtab[(n * 2) as usize]
                                + i1 as f32 * trigtab[(n * 2 + 1) as usize])
                                .round() as i32
                                + (numrho - 1) / 2;
                            if r >= 0 && r < numrho {
                                accum[(n * numrho + r) as usize] -= 1;
                            }
                        }
                    }
                    mask[midx] = 0; // Always consume, even if not good
                }

                // Stop at the endpoint found in first walk
                if i1 == line_end[k].1 && j1 == line_end[k].0 {
                    break;
                }

                x += dx;
                y += dy;
            }
        }

        if good_line {
            lines.push(LineSegment {
                x1: line_end[0].0,
                y1: line_end[0].1,
                x2: line_end[1].0,
                y2: line_end[1].1,
            });
        }
    }

    Ok(lines)
}

