//! Analysis helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Connected component labeling on a binary (thresholded) grayscale image.
///
/// Returns a label map where each pixel has the label of its connected component
/// (0 = background, 1..N = component labels). Matches `cv2.connectedComponents`.
///
/// `connectivity`: 4 or 8 (default 8).
/// Input must be binary: 0 = background, non-zero = foreground.
pub fn connected_components(
    pixels: &[u8],
    info: &ImageInfo,
    connectivity: u32,
) -> Result<(Vec<u32>, u32), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "connected_components requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;

    let mut labels = vec![0u32; w * h];
    let mut parent = vec![0u32; w * h + 1]; // union-find
    let mut next_label: u32 = 1;

    // Initialize union-find
    for (i, p) in parent.iter_mut().enumerate() {
        *p = i as u32;
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize]; // path compression
            x = parent[x as usize];
        }
        x
    }

    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Pass 1: assign provisional labels
    for y in 0..h {
        for x in 0..w {
            if pixels[y * w + x] == 0 {
                continue; // background
            }

            let mut neighbors = Vec::with_capacity(4);

            // Check neighbors based on connectivity
            if y > 0 && pixels[(y - 1) * w + x] != 0 {
                neighbors.push(labels[(y - 1) * w + x]); // above
            }
            if x > 0 && pixels[y * w + x - 1] != 0 {
                neighbors.push(labels[y * w + x - 1]); // left
            }
            if connectivity == 8 {
                if y > 0 && x > 0 && pixels[(y - 1) * w + x - 1] != 0 {
                    neighbors.push(labels[(y - 1) * w + x - 1]); // above-left
                }
                if y > 0 && x + 1 < w && pixels[(y - 1) * w + x + 1] != 0 {
                    neighbors.push(labels[(y - 1) * w + x + 1]); // above-right
                }
            }

            if neighbors.is_empty() {
                labels[y * w + x] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[y * w + x] = min_label;
                for &n in &neighbors {
                    if n != min_label {
                        union(&mut parent, n, min_label);
                    }
                }
            }
        }
    }

    // Pass 2: resolve labels
    let mut label_map = vec![0u32; next_label as usize];
    let mut num_labels: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            if labels[y * w + x] > 0 {
                let root = find(&mut parent, labels[y * w + x]);
                if label_map[root as usize] == 0 {
                    num_labels += 1;
                    label_map[root as usize] = num_labels;
                }
                labels[y * w + x] = label_map[root as usize];
            }
        }
    }

    Ok((labels, num_labels))
}

/// Euclidean distance transform — distance from each pixel to nearest zero pixel.
///
/// Input: grayscale image where 0 = background, >0 = foreground.
/// Output: grayscale image where each pixel = distance to nearest background pixel.
/// Uses two-pass Rosenfeld-Pfaltz algorithm.
/// Reference: cv2.distanceTransform (OpenCV 4.13, DIST_L2).
pub fn distance_transform(pixels: &[u8], info: &ImageInfo) -> Result<Vec<f64>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "distance transform requires Gray8".into(),
        ));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let inf = (w + h) as f64;

    // Initialize: 0 for background, infinity for foreground
    let mut dist = vec![0.0f64; w * h];
    for i in 0..w * h {
        dist[i] = if pixels[i] == 0 { 0.0 } else { inf };
    }

    // Forward pass: top-left to bottom-right
    for y in 0..h {
        for x in 0..w {
            if dist[y * w + x] == 0.0 {
                continue;
            }
            if y > 0 {
                dist[y * w + x] = dist[y * w + x].min(dist[(y - 1) * w + x] + 1.0);
            }
            if x > 0 {
                dist[y * w + x] = dist[y * w + x].min(dist[y * w + x - 1] + 1.0);
            }
            // Diagonal
            if y > 0 && x > 0 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y - 1) * w + x - 1] + std::f64::consts::SQRT_2);
            }
            if y > 0 && x < w - 1 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y - 1) * w + x + 1] + std::f64::consts::SQRT_2);
            }
        }
    }

    // Backward pass: bottom-right to top-left
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if dist[y * w + x] == 0.0 {
                continue;
            }
            if y < h - 1 {
                dist[y * w + x] = dist[y * w + x].min(dist[(y + 1) * w + x] + 1.0);
            }
            if x < w - 1 {
                dist[y * w + x] = dist[y * w + x].min(dist[y * w + x + 1] + 1.0);
            }
            if y < h - 1 && x < w - 1 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y + 1) * w + x + 1] + std::f64::consts::SQRT_2);
            }
            if y < h - 1 && x > 0 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y + 1) * w + x - 1] + std::f64::consts::SQRT_2);
            }
        }
    }

    Ok(dist)
}

/// Trace external contours of foreground regions in a binary Gray8 image.
///
/// Uses a simplified Suzuki-Abe border following algorithm to extract ordered
/// boundary point sequences from a binary image (0 = background, non-zero = foreground).
///
/// Returns a list of contours, each being an ordered list of (x, y) boundary points.
/// Only external (outer) contours are returned — no hierarchy.
///
/// Reference: Suzuki & Abe (1985), "Topological Structural Analysis of Digitized
/// Binary Images by Border Following"
pub fn find_contours(pixels: &[u8], info: &ImageInfo) -> Result<Vec<Vec<(i32, i32)>>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "find_contours requires Gray8 input".into(),
        ));
    }
    let w = info.width as i32;
    let h = info.height as i32;

    // Work on a copy with 1-pixel border padding (simplifies boundary checks)
    let pw = (w + 2) as usize;
    let ph = (h + 2) as usize;
    let mut img = vec![0i32; pw * ph];
    for y in 0..h as usize {
        for x in 0..w as usize {
            if pixels[y * w as usize + x] != 0 {
                img[(y + 1) * pw + (x + 1)] = 1;
            }
        }
    }

    // 8-connectivity neighbor offsets (clockwise from right)
    // Index: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
    let dx: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    let dy: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

    let mut contours = Vec::new();
    let mut nbd: i32 = 1; // current border sequential number

    for y in 1..ph as i32 - 1 {
        for x in 1..pw as i32 - 1 {
            let idx = y as usize * pw + x as usize;
            // Detect outer border start: pixel is foreground and left neighbor is background
            if img[idx] == 1 && img[idx - 1] == 0 {
                nbd += 1;
                let contour = trace_border(&mut img, pw, x, y, &dx, &dy, nbd);
                if !contour.is_empty() {
                    // Convert from padded coordinates back to original
                    let original: Vec<(i32, i32)> =
                        contour.iter().map(|&(cx, cy)| (cx - 1, cy - 1)).collect();
                    contours.push(original);
                }
            }
            // Mark visited foreground pixels to avoid re-tracing
            if img[idx] != 0 && img[idx].abs() <= 1 {
                // Already traced or will be traced
            }
        }
    }

    Ok(contours)
}

/// Flood fill from a seed point with configurable tolerance and connectivity.
///
/// Fills connected pixels within `tolerance` of the seed pixel's value with
/// `new_val`. Returns the modified image and the number of pixels filled.
///
/// Matches `cv2.floodFill` behavior for grayscale images.
pub fn flood_fill(
    pixels: &[u8],
    info: &ImageInfo,
    seed_x: u32,
    seed_y: u32,
    new_val: u8,
    tolerance: u8,
    connectivity: u32,
) -> Result<(Vec<u8>, u32), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "flood_fill requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let sx = seed_x as usize;
    let sy = seed_y as usize;
    if sx >= w || sy >= h {
        return Err(ImageError::InvalidParameters(
            "seed point out of bounds".into(),
        ));
    }

    let mut result = pixels.to_vec();
    let seed_val = pixels[sy * w + sx];
    let lo = seed_val.saturating_sub(tolerance);
    let hi = seed_val.saturating_add(tolerance);

    let mut visited = vec![false; w * h];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((sx, sy));
    visited[sy * w + sx] = true;
    let mut filled: u32 = 0;

    while let Some((cx, cy)) = queue.pop_front() {
        let val = pixels[cy * w + cx];
        if val < lo || val > hi {
            continue;
        }
        result[cy * w + cx] = new_val;
        filled += 1;

        // 4-connectivity neighbors
        let neighbors: &[(i32, i32)] = if connectivity == 8 {
            &[
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
        } else {
            &[(0, -1), (-1, 0), (1, 0), (0, 1)]
        };

        for &(dx, dy) in neighbors {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let ni = ny as usize * w + nx as usize;
                if !visited[ni] {
                    visited[ni] = true;
                    let nval = pixels[ni];
                    if nval >= lo && nval <= hi {
                        queue.push_back((nx as usize, ny as usize));
                    }
                }
            }
        }
    }

    Ok((result, filled))
}

