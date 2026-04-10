use super::super::error::ImageError;
use super::{BAND, FmmEntry, INSIDE, KNOWN, fmm_init_padded, neighbors_4_opencv, solve_eikonal};

pub(crate) fn navier_stokes_inpaint(
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
            if kx <= 0 || ky <= 0 || kx >= w as i32 - 1 || ky >= h as i32 - 1 {
                continue;
            }
            let kx = kx as usize;
            let ky = ky as usize;
            let kidx = ky * w + kx;

            if flags[kidx] == INSIDE {
                continue;
            }

            let rx = (x as f64) - (kx as f64);
            let ry = (y as f64) - (ky as f64);
            let r2 = rx * rx + ry * ry;
            if r2 == 0.0 || r2 > (r as f64 * r as f64) {
                continue;
            }

            let dst = 1.0 / (r2 * r2 + 1.0);

            let mut grad_ix = 0.0f64;
            let mut grad_iy = 0.0f64;

            let down_ok = ky + 1 < h && flags[(ky + 1) * w + kx] != INSIDE;
            let up_ok = ky > 0 && flags[(ky - 1) * w + kx] != INSIDE;
            if down_ok {
                if up_ok {
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

            grad_ix = -grad_ix;

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
