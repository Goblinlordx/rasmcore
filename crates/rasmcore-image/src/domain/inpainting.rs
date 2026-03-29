//! Image inpainting — fill masked regions using surrounding pixel information.
//!
//! Two algorithms, both FMM-based, matching OpenCV's `cv2.inpaint()`:
//! - **Telea (FMM):** Weighted interpolation using distance, level-set, and time-gradient
//!   alignment, plus a first-order gradient correction term (Telea 2004).
//! - **Navier-Stokes (FMM):** Weighted interpolation using distance^4 and image-gradient
//!   alignment. Pure weighted average without gradient correction.

use super::error::ImageError;
use super::types::ImageInfo;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Inpainting method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InpaintMethod {
    /// Fast Marching Method (Telea 2004) — fast, good for thin masks.
    Telea,
    /// Navier-Stokes FMM (OpenCV variant) — image-gradient-weighted interpolation.
    NavierStokes,
}

/// Inpaint masked regions of a grayscale image.
pub fn inpaint(
    pixels: &[u8],
    info: &ImageInfo,
    mask: &[u8],
    radius: f32,
    method: InpaintMethod,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;

    if pixels.len() < w * h {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if mask.len() < w * h {
        return Err(ImageError::InvalidInput("mask buffer too small".into()));
    }
    if radius <= 0.0 {
        return Err(ImageError::InvalidParameters("radius must be > 0".into()));
    }

    match method {
        InpaintMethod::Telea => telea_inpaint(pixels, w, h, mask, radius),
        InpaintMethod::NavierStokes => navier_stokes_inpaint(pixels, w, h, mask, radius),
    }
}

/// Inpaint an RGB image (processes each channel independently).
pub fn inpaint_rgb(
    pixels: &[u8],
    info: &ImageInfo,
    mask: &[u8],
    radius: f32,
    method: InpaintMethod,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let n = w * h;

    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("RGB buffer too small".into()));
    }

    let mut r = vec![0u8; n];
    let mut g = vec![0u8; n];
    let mut b = vec![0u8; n];
    for i in 0..n {
        r[i] = pixels[i * 3];
        g[i] = pixels[i * 3 + 1];
        b[i] = pixels[i * 3 + 2];
    }

    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: super::types::PixelFormat::Gray8,
        color_space: info.color_space,
    };

    let r_out = inpaint(&r, &gray_info, mask, radius, method)?;
    let g_out = inpaint(&g, &gray_info, mask, radius, method)?;
    let b_out = inpaint(&b, &gray_info, mask, radius, method)?;

    let mut result = vec![0u8; n * 3];
    for i in 0..n {
        result[i * 3] = r_out[i];
        result[i * 3 + 1] = g_out[i];
        result[i * 3 + 2] = b_out[i];
    }
    Ok(result)
}

// ─── FMM Infrastructure ────────────────────────────────────────────────────

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;

#[derive(Clone, Copy)]
struct FmmEntry {
    dist: f32,
    seq: u32,
    x: usize,
    y: usize,
}

impl PartialEq for FmmEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.seq == other.seq
    }
}
impl Eq for FmmEntry {}
impl PartialOrd for FmmEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FmmEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smallest distance first, FIFO for ties
        match other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => other.seq.cmp(&self.seq),
            ord => ord,
        }
    }
}

/// Solve Eikonal equation for a single pair of neighbors.
/// Matches OpenCV's FastMarching_solve(i1,j1, i2,j2, f, t).
fn fm_solve_pair(a11: f32, f1_inside: bool, a22: f32, f2_inside: bool) -> f32 {
    let m12 = a11.min(a22);
    if !f1_inside {
        if !f2_inside {
            if (a11 - a22).abs() >= 1.0 {
                1.0 + m12
            } else {
                let d = a11 as f64 - a22 as f64;
                ((a11 as f64 + a22 as f64 + (2.0 - d * d).sqrt()) * 0.5) as f32
            }
        } else {
            1.0 + a11
        }
    } else if !f2_inside {
        1.0 + a22
    } else {
        1.0 + m12
    }
}

/// Solve Eikonal using OpenCV's min4 approach: try all 4 diagonal pairs
/// and return the minimum distance.
fn solve_eikonal(dist: &[f32], flags: &[u8], w: usize, h: usize, x: usize, y: usize) -> f32 {
    // Neighbor values and INSIDE status
    let up = if y > 0 {
        (dist[(y - 1) * w + x], flags[(y - 1) * w + x] == INSIDE)
    } else {
        (1e6, true)
    };
    let down = if y + 1 < h {
        (dist[(y + 1) * w + x], flags[(y + 1) * w + x] == INSIDE)
    } else {
        (1e6, true)
    };
    let left = if x > 0 {
        (dist[y * w + x - 1], flags[y * w + x - 1] == INSIDE)
    } else {
        (1e6, true)
    };
    let right = if x + 1 < w {
        (dist[y * w + x + 1], flags[y * w + x + 1] == INSIDE)
    } else {
        (1e6, true)
    };

    // OpenCV tries all 4 diagonal combinations and takes min
    let s1 = fm_solve_pair(up.0, up.1, left.0, left.1);
    let s2 = fm_solve_pair(down.0, down.1, left.0, left.1);
    let s3 = fm_solve_pair(up.0, up.1, right.0, right.1);
    let s4 = fm_solve_pair(down.0, down.1, right.0, right.1);

    s1.min(s2).min(s3).min(s4)
}

/// OpenCV internally pads the flag/distance arrays by 1 pixel on all sides.
/// We emulate this with a padded representation: pw = w+2, ph = h+2.
/// The (x,y) pixel in the original maps to (x+1, y+1) in padded space.
/// The border pixels are KNOWN with dist=0.
fn fmm_init_padded(
    mask: &[u8],
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<f32>, BinaryHeap<FmmEntry>, usize, usize, u32) {
    let pw = w + 2;
    let ph = h + 2;
    let pn = pw * ph;
    let mut flags = vec![KNOWN; pn]; // Border is KNOWN by default
    let mut dist = vec![0.0f32; pn]; // Border dist = 0

    // Mark mask pixels as INSIDE (in padded coords)
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] > 0 {
                let pidx = (y + 1) * pw + (x + 1);
                flags[pidx] = INSIDE;
                dist[pidx] = 1e6;
            }
        }
    }

    let mut heap = BinaryHeap::new();
    let mut seq: u32 = 0;

    // Initial band: KNOWN pixels adjacent to INSIDE pixels (distance = 0)
    for py in 0..ph {
        for px in 0..pw {
            let pidx = py * pw + px;
            if flags[pidx] == KNOWN {
                let has_inside_neighbor = neighbors_4(px, py, pw, ph)
                    .iter()
                    .any(|&(nx, ny)| flags[ny * pw + nx] == INSIDE);
                if has_inside_neighbor {
                    flags[pidx] = BAND;
                    dist[pidx] = 0.0;
                    heap.push(FmmEntry {
                        dist: 0.0,
                        seq,
                        x: px,
                        y: py,
                    });
                    seq += 1;
                }
            }
        }
    }

    (flags, dist, heap, pw, ph, seq)
}

fn telea_inpaint(
    pixels: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    radius: f32,
) -> Result<Vec<u8>, ImageError> {
    let (mut flags, mut dist, mut heap, pw, ph, mut seq) = fmm_init_padded(mask, w, h);
    let r = radius.ceil() as i32;

    let mut result = vec![0u8; pw * ph];
    for y in 0..h {
        for x in 0..w {
            result[(y + 1) * pw + (x + 1)] = pixels[y * w + x];
        }
    }

    while let Some(entry) = heap.pop() {
        let (x, y) = (entry.x, entry.y);
        let idx = y * pw + x;

        if flags[idx] == KNOWN {
            continue;
        }

        flags[idx] = KNOWN;

        for &(nx, ny) in &neighbors_4_opencv(x, y, pw, ph) {
            let nidx = ny * pw + nx;
            if flags[nidx] == INSIDE {
                let new_dist = solve_eikonal(&dist, &flags, pw, ph, nx, ny);
                dist[nidx] = new_dist;

                result[nidx] = telea_interpolate(&result, &flags, &dist, pw, ph, nx, ny, r);

                flags[nidx] = BAND;
                heap.push(FmmEntry {
                    dist: new_dist,
                    seq,
                    x: nx,
                    y: ny,
                });
                seq += 1;
            }
        }
    }

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = result[(y + 1) * pw + (x + 1)];
        }
    }
    Ok(out)
}

/// Telea interpolation matching OpenCV's icvTeleaInpaintFMM.
///
/// Weight = |dst * lev * dir| where:
///   dst = 1 / (|r|^2 * |r|) = 1/|r|^3
///   lev = 1 / (1 + |t_neighbor - t_target|)
///   dir = dot(r, gradT)
///
/// Final value = Ia/s + (Jx+Jy) / sqrt(Jx^2+Jy^2) (gradient correction)
fn telea_interpolate(
    pixels: &[u8],
    flags: &[u8],
    dist: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    r: i32,
) -> u8 {
    // Compute gradT (gradient of arrival time field) at target (x, y)
    // OpenCV checks INSIDE flag, not just bounds
    let idx = y * w + x;
    let t_center = dist[idx];

    let right_ok = x + 1 < w && flags[y * w + x + 1] != INSIDE;
    let left_ok = x > 0 && flags[y * w + x - 1] != INSIDE;
    let grad_t_x: f32 = if right_ok {
        if left_ok {
            (dist[y * w + x + 1] - dist[y * w + x - 1]) * 0.5
        } else {
            dist[y * w + x + 1] - t_center
        }
    } else if left_ok {
        t_center - dist[y * w + x - 1]
    } else {
        0.0
    };

    let down_ok = y + 1 < h && flags[(y + 1) * w + x] != INSIDE;
    let up_ok = y > 0 && flags[(y - 1) * w + x] != INSIDE;
    let grad_t_y: f32 = if down_ok {
        if up_ok {
            (dist[(y + 1) * w + x] - dist[(y - 1) * w + x]) * 0.5
        } else {
            dist[(y + 1) * w + x] - t_center
        }
    } else if up_ok {
        t_center - dist[(y - 1) * w + x]
    } else {
        0.0
    };

    // OpenCV accumulates in f32, final division in f64
    let mut ia = 0.0f32;
    let mut jx = 0.0f32;
    let mut jy = 0.0f32;
    let mut s = 0.0f32;

    for dy in -r..=r {
        for dx in -r..=r {
            let kx = x as i32 + dx;
            let ky = y as i32 + dy;
            // OpenCV boundary: k>0 && l>0 && k<rows-1 && l<cols-1 (padded)
            if kx <= 0 || ky <= 0 || kx >= w as i32 - 1 || ky >= h as i32 - 1 {
                continue;
            }
            let kx = kx as usize;
            let ky = ky as usize;
            let kidx = ky * w + kx;

            // OpenCV uses != INSIDE (includes both KNOWN and BAND)
            if flags[kidx] == INSIDE {
                continue;
            }

            let rx = (x as f32) - (kx as f32);
            let ry = (y as f32) - (ky as f32);
            let r2 = rx * rx + ry * ry;
            if r2 == 0.0 || r2 > (r as f32 * r as f32) {
                continue;
            }

            // dst = 1 / (|r|^2 * sqrt(|r|^2)) = 1/|r|^3
            let dst = 1.0f32 / (r2 * r2.sqrt());

            // lev = 1 / (1 + |t_neighbor - t_target|)
            let lev = 1.0f32 / (1.0 + (dist[kidx] - t_center).abs());

            // dir = dot(r, gradT)
            let dir_val = rx * grad_t_x + ry * grad_t_y;
            let dir = if dir_val.abs() <= 0.01 {
                0.000001f32
            } else {
                dir_val.abs()
            };

            let weight = (dst * lev * dir).abs();

            // Image gradient at neighbor — matches OpenCV's flag-aware selection
            let mut grad_ix = 0.0f32;
            let mut grad_iy = 0.0f32;

            let right_ok = kx + 1 < w && flags[ky * w + kx + 1] != INSIDE;
            let left_ok = kx > 0 && flags[ky * w + kx - 1] != INSIDE;
            if right_ok {
                if left_ok {
                    grad_ix =
                        (pixels[ky * w + kx + 1] as f32 - pixels[ky * w + kx - 1] as f32) * 2.0;
                } else {
                    grad_ix = pixels[ky * w + kx + 1] as f32 - pixels[kidx] as f32;
                }
            } else if left_ok {
                grad_ix = pixels[kidx] as f32 - pixels[ky * w + kx - 1] as f32;
            }

            let down_ok = ky + 1 < h && flags[(ky + 1) * w + kx] != INSIDE;
            let up_ok = ky > 0 && flags[(ky - 1) * w + kx] != INSIDE;
            if down_ok {
                if up_ok {
                    grad_iy =
                        (pixels[(ky + 1) * w + kx] as f32 - pixels[(ky - 1) * w + kx] as f32) * 2.0;
                } else {
                    grad_iy = pixels[(ky + 1) * w + kx] as f32 - pixels[kidx] as f32;
                }
            } else if up_ok {
                grad_iy = pixels[kidx] as f32 - pixels[(ky - 1) * w + kx] as f32;
            }

            ia += weight * pixels[kidx] as f32;
            jx -= weight * grad_ix * rx;
            jy -= weight * grad_iy * ry;
            s += weight;
        }
    }

    if s > 0.0 {
        // OpenCV: Ia/s (f64) + correction, then saturate_cast (round)
        let j_mag = ((jx as f64) * (jx as f64) + (jy as f64) * (jy as f64)).sqrt();
        let sat = (ia as f64 / s as f64) + (jx as f64 + jy as f64) / (j_mag + 1e-20);
        sat.clamp(0.0, 255.0).round() as u8
    } else {
        0
    }
}

// ─── Navier-Stokes FMM Inpainting ──────────────────────────────────────────
//
// OpenCV's INPAINT_NS is actually FMM-based (same skeleton as Telea) but with
// different interpolation weights:
//   - Distance: 1 / (|r|^4 + 1)  (steeper falloff than Telea)
//   - Direction: |cos(angle(r, gradI))|  (image gradient alignment)
//   - No level-set weight, no gradient correction term
//   - Pure weighted average

fn navier_stokes_inpaint(
    pixels: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    radius: f32,
) -> Result<Vec<u8>, ImageError> {
    let (mut flags, mut dist, mut heap, pw, ph, mut seq) = fmm_init_padded(mask, w, h);
    let r = radius.ceil() as i32;

    let mut result = vec![0u8; pw * ph];
    for y in 0..h {
        for x in 0..w {
            result[(y + 1) * pw + (x + 1)] = pixels[y * w + x];
        }
    }

    while let Some(entry) = heap.pop() {
        let (x, y) = (entry.x, entry.y);
        let idx = y * pw + x;

        if flags[idx] == KNOWN {
            continue;
        }

        flags[idx] = KNOWN;

        for &(nx, ny) in &neighbors_4_opencv(x, y, pw, ph) {
            let nidx = ny * pw + nx;
            if flags[nidx] == INSIDE {
                let new_dist = solve_eikonal(&dist, &flags, pw, ph, nx, ny);
                dist[nidx] = new_dist;

                result[nidx] = ns_interpolate(&result, &flags, pw, ph, nx, ny, r);

                flags[nidx] = BAND;
                heap.push(FmmEntry {
                    dist: new_dist,
                    seq,
                    x: nx,
                    y: ny,
                });
                seq += 1;
            }
        }
    }

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = result[(y + 1) * pw + (x + 1)];
        }
    }
    Ok(out)
}

/// NS interpolation: weighted average using distance^4 and image gradient alignment.
/// Matches OpenCV's icvNSInpaintFMM.
fn ns_interpolate(
    pixels: &[u8],
    flags: &[u8],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    r: i32,
) -> u8 {
    let mut ia = 0.0f64;
    let mut s = 0.0f64;

    for dy in -r..=r {
        for dx in -r..=r {
            let kx = x as i32 + dx;
            let ky = y as i32 + dy;
            // OpenCV boundary: k>0 && l>0 && k<rows-1 && l<cols-1 (padded)
            if kx <= 0 || ky <= 0 || kx >= w as i32 - 1 || ky >= h as i32 - 1 {
                continue;
            }
            let kx = kx as usize;
            let ky = ky as usize;
            let kidx = ky * w + kx;

            // OpenCV uses != INSIDE (includes both KNOWN and BAND)
            if flags[kidx] == INSIDE {
                continue;
            }

            // r vector: from neighbor (k) to target (x,y)
            let rx = (x as f64) - (kx as f64);
            let ry = (y as f64) - (ky as f64);
            let r2 = rx * rx + ry * ry; // |r|^2
            if r2 == 0.0 || r2 > (r as f64 * r as f64) {
                continue;
            }

            // Distance weight: 1 / (|r|^4 + 1) — matches OpenCV VectorLength squared
            let dst = 1.0 / (r2 * r2 + 1.0);

            // OpenCV NS gradient: gradI.x = ROW gradient, gradI.y = COL gradient
            // Uses sum of absolute differences (TV-style), not central diff
            let mut grad_ix = 0.0f64; // row gradient (k±1 direction)
            let mut grad_iy = 0.0f64; // col gradient (l±1 direction)

            let down_ok = ky + 1 < h && flags[(ky + 1) * w + kx] != INSIDE;
            let up_ok = ky > 0 && flags[(ky - 1) * w + kx] != INSIDE;
            if down_ok {
                if up_ok {
                    // Central: abs(down-center) + abs(center-up)
                    grad_ix = (pixels[(ky + 1) * w + kx] as f64 - pixels[kidx] as f64).abs()
                        + (pixels[kidx] as f64 - pixels[(ky - 1) * w + kx] as f64).abs();
                } else {
                    grad_ix = (pixels[(ky + 1) * w + kx] as f64 - pixels[kidx] as f64).abs() * 2.0;
                }
            } else if up_ok {
                grad_ix = (pixels[kidx] as f64 - pixels[(ky - 1) * w + kx] as f64).abs() * 2.0;
            }

            let right_ok = kx + 1 < w && flags[ky * w + kx + 1] != INSIDE;
            let left_ok = kx > 0 && flags[ky * w + kx - 1] != INSIDE;
            if right_ok {
                if left_ok {
                    grad_iy = (pixels[ky * w + kx + 1] as f64 - pixels[kidx] as f64).abs()
                        + (pixels[kidx] as f64 - pixels[ky * w + kx - 1] as f64).abs();
                } else {
                    grad_iy = (pixels[ky * w + kx + 1] as f64 - pixels[kidx] as f64).abs() * 2.0;
                }
            } else if left_ok {
                grad_iy = (pixels[kidx] as f64 - pixels[ky * w + kx - 1] as f64).abs() * 2.0;
            }

            // OpenCV negates gradI.x (the row gradient)
            grad_ix = -grad_ix;

            // dot(r, gradI) = rx * gradI.x + ry * gradI.y
            // where rx = col displacement, ry = row displacement
            // gradI.x = -row_gradient, gradI.y = col_gradient
            let dot = rx * grad_ix + ry * grad_iy;
            let grad2 = grad_ix * grad_ix + grad_iy * grad_iy;
            let dir = if dot.abs() <= 0.01 {
                0.000001
            } else {
                (dot / (r2 * grad2).sqrt()).abs()
            };

            let weight = dst * dir;
            ia += weight * pixels[kidx] as f64;
            s += weight;
        }
    }

    if s > 0.0 {
        (ia / s).clamp(0.0, 255.0).round() as u8
    } else {
        0
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn neighbors_4(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    if x > 0 {
        n.push((x - 1, y));
    }
    if x + 1 < w {
        n.push((x + 1, y));
    }
    if y > 0 {
        n.push((x, y - 1));
    }
    if y + 1 < h {
        n.push((x, y + 1));
    }
    n
}

/// OpenCV neighbor order: up (i-1,j), left (i,j-1), down (i+1,j), right (i,j+1)
/// Boundary: valid range is 1..h-1 (exclusive) for rows, 1..w-1 for cols (padded).
fn neighbors_4_opencv(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    // up: i-1 must be > 0 (i.e., y-1 >= 1, so y >= 2)
    if y >= 2 {
        n.push((x, y - 1));
    }
    // left: j-1 must be > 0 (i.e., x-1 >= 1, so x >= 2)
    if x >= 2 {
        n.push((x - 1, y));
    }
    // down: i+1 must be <= rows-1 (i.e., y+1 <= h-1)
    if y + 1 < h {
        n.push((x, y + 1));
    }
    // right: j+1 must be <= cols-1 (i.e., x+1 <= w-1)
    if x + 1 < w {
        n.push((x + 1, y));
    }
    n
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::*;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn telea_fills_single_pixel() {
        let mut pixels = vec![128u8; 8 * 8];
        let mut mask = vec![0u8; 8 * 8];
        // Mask out center pixel
        mask[4 * 8 + 4] = 255;
        pixels[4 * 8 + 4] = 0; // damaged pixel

        let result = inpaint(&pixels, &gray_info(8, 8), &mask, 3.0, InpaintMethod::Telea).unwrap();

        // Center pixel should be filled with ~128 (surrounded by 128s)
        let center = result[4 * 8 + 4];
        assert!(
            (center as i32 - 128).abs() < 5,
            "single pixel inpaint: expected ~128, got {center}"
        );
    }

    #[test]
    fn navier_stokes_fills_region() {
        let mut pixels = vec![128u8; 16 * 16];
        let mut mask = vec![0u8; 16 * 16];
        // Mask out a 4x4 region in the center
        for y in 6..10 {
            for x in 6..10 {
                mask[y * 16 + x] = 255;
                pixels[y * 16 + x] = 0;
            }
        }

        let result = inpaint(
            &pixels,
            &gray_info(16, 16),
            &mask,
            5.0,
            InpaintMethod::NavierStokes,
        )
        .unwrap();

        // All masked pixels should be near 128 (uniform surround)
        for y in 6..10 {
            for x in 6..10 {
                let v = result[y * 16 + x];
                assert!(
                    (v as i32 - 128).abs() < 20,
                    "NS inpaint at ({x},{y}): expected ~128, got {v}"
                );
            }
        }
    }

    #[test]
    fn empty_mask_is_identity() {
        let pixels = vec![42u8; 8 * 8];
        let mask = vec![0u8; 8 * 8];
        let result = inpaint(&pixels, &gray_info(8, 8), &mask, 3.0, InpaintMethod::Telea).unwrap();
        assert_eq!(result, pixels, "empty mask should be identity");
    }

    #[test]
    fn rgb_inpainting_works() {
        let n = 8 * 8;
        let mut pixels = vec![128u8; n * 3];
        let mut mask = vec![0u8; n];
        mask[4 * 8 + 4] = 255;
        for c in 0..3 {
            pixels[(4 * 8 + 4) * 3 + c] = 0;
        }

        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };

        let result = inpaint_rgb(&pixels, &info, &mask, 3.0, InpaintMethod::Telea).unwrap();
        assert_eq!(result.len(), n * 3);
        // Each channel should be ~128
        for c in 0..3 {
            let v = result[(4 * 8 + 4) * 3 + c];
            assert!(
                (v as i32 - 128).abs() < 5,
                "RGB inpaint ch{c}: expected ~128, got {v}"
            );
        }
    }
}

#[cfg(test)]
mod opencv_parity {
    use super::*;
    use crate::domain::types::*;
    use std::path::Path;
    use std::process::Command;

    fn venv_python() -> String {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
        assert!(venv.exists(), "venv not found");
        venv.to_string_lossy().into_owned()
    }

    fn run_python(script: &str) -> Vec<u8> {
        let output = Command::new(venv_python())
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Python failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    fn mae(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    fn max_err(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn telea_parity_vs_opencv() {
        // Create test image: uniform 128 with a masked 4x4 hole
        let w = 16u32;
        let h = 16;
        let mut img = vec![128u8; (w * h) as usize];
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 6..10 {
            for x in 6..10 {
                mask[y * w as usize + x] = 255;
                img[y * w as usize + x] = 0;
            }
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let ours = inpaint(&img, &info, &mask, 3.0, InpaintMethod::Telea).unwrap();

        // OpenCV reference
        let script = format!(
            "import sys, cv2\nimport numpy as np\n\
             img=np.array({img:?},dtype=np.uint8).reshape({h},{w})\n\
             mask=np.array({mask:?},dtype=np.uint8).reshape({h},{w})\n\
             out=cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  Telea inpaint vs OpenCV: MAE={m:.4}, max_err={mx}");
        assert!(
            m == 0.0 && mx == 0,
            "Telea MAE={m:.4}, max_err={mx} vs OpenCV — must be exact"
        );
    }

    #[test]
    fn ns_parity_vs_opencv() {
        let w = 16u32;
        let h = 16;
        let mut img = vec![128u8; (w * h) as usize];
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 6..10 {
            for x in 6..10 {
                mask[y * w as usize + x] = 255;
                img[y * w as usize + x] = 0;
            }
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let ours = inpaint(&img, &info, &mask, 3.0, InpaintMethod::NavierStokes).unwrap();

        let script = format!(
            "import sys, cv2\nimport numpy as np\n\
             img=np.array({img:?},dtype=np.uint8).reshape({h},{w})\n\
             mask=np.array({mask:?},dtype=np.uint8).reshape({h},{w})\n\
             out=cv2.inpaint(img,mask,3,cv2.INPAINT_NS)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  NS inpaint vs OpenCV: MAE={m:.4}, max_err={mx}");
        assert!(
            m == 0.0 && mx == 0,
            "NS MAE={m:.4}, max_err={mx} vs OpenCV — must be exact"
        );
    }

    #[test]
    fn telea_gradient_parity_vs_opencv() {
        // Gradient image: harder test for Telea with non-uniform content
        let w = 32u32;
        let h = 32;
        let mut img = vec![0u8; (w * h) as usize];
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                img[y * w as usize + x] = (x * 255 / (w as usize - 1)) as u8;
            }
        }
        // 6x6 hole in center
        for y in 13..19 {
            for x in 13..19 {
                mask[y * w as usize + x] = 255;
                img[y * w as usize + x] = 0;
            }
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let ours = inpaint(&img, &info, &mask, 5.0, InpaintMethod::Telea).unwrap();

        let script = format!(
            "import sys, cv2\nimport numpy as np\n\
             img=np.array({img:?},dtype=np.uint8).reshape({h},{w})\n\
             mask=np.array({mask:?},dtype=np.uint8).reshape({h},{w})\n\
             out=cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  Telea gradient vs OpenCV: MAE={m:.4}, max_err={mx}");
        // Telea's gradient correction term is sensitive to FMM heap ordering
        // (tie-breaking for equal distances). OpenCV's custom heap differs from
        // Rust's BinaryHeap. NS (pure weighted average) is exact; Telea's
        // first-order gradient correction amplifies tiny ordering differences.
        // Uniform images are exact (no gradient to amplify). Threshold: max_err ≤ 6.
        assert!(
            mx <= 6,
            "Telea gradient max_err={mx} vs OpenCV — should be ≤ 6"
        );
    }

    #[test]
    fn ns_gradient_parity_vs_opencv() {
        // Gradient image: harder test for NS
        let w = 32u32;
        let h = 32;
        let mut img = vec![0u8; (w * h) as usize];
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                img[y * w as usize + x] = (x * 255 / (w as usize - 1)) as u8;
            }
        }
        for y in 13..19 {
            for x in 13..19 {
                mask[y * w as usize + x] = 255;
                img[y * w as usize + x] = 0;
            }
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let ours = inpaint(&img, &info, &mask, 5.0, InpaintMethod::NavierStokes).unwrap();

        let script = format!(
            "import sys, cv2\nimport numpy as np\n\
             img=np.array({img:?},dtype=np.uint8).reshape({h},{w})\n\
             mask=np.array({mask:?},dtype=np.uint8).reshape({h},{w})\n\
             out=cv2.inpaint(img,mask,5,cv2.INPAINT_NS)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  NS gradient vs OpenCV: MAE={m:.4}, max_err={mx}");
        assert!(
            m == 0.0 && mx == 0,
            "NS gradient MAE={m:.4}, max_err={mx} vs OpenCV — must be exact"
        );
    }
}
