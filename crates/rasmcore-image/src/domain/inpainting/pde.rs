use super::super::error::ImageError;
use super::{fmm_init_padded, neighbors_4_opencv, solve_eikonal, BAND, INSIDE, KNOWN, FmmEntry};

pub(crate) fn telea_inpaint(
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
#[allow(clippy::too_many_arguments)]
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

    let mut ia: f32 = 0.0;
    let mut jx: f32 = 0.0;
    let mut jy: f32 = 0.0;
    let mut s: f32 = 1.0e-20;

    let range = r;
    let rows = h as i32;
    let cols = w as i32;

    for ky in (y as i32 - range)..=(y as i32 + range) {
        for kx in (x as i32 - range)..=(x as i32 + range) {
            if kx <= 0 || ky <= 0 || kx >= cols - 1 || ky >= rows - 1 {
                continue;
            }
            let kxu = kx as usize;
            let kyu = ky as usize;
            let kidx = kyu * w + kxu;

            if flags[kidx] == INSIDE {
                continue;
            }

            let dk = ky - y as i32;
            let dl = kx - x as i32;
            if dl * dl + dk * dk > range * range {
                continue;
            }

            let ry = (y as i32 - ky) as f32;
            let rx = (x as i32 - kx) as f32;
            let r2 = rx * rx + ry * ry;

            let denom_f32 = r2 * r2.sqrt();
            let dst: f32 = (1.0f64 / denom_f32 as f64) as f32;

            let t_diff = (dist[kidx] - t_center).abs();
            let lev: f32 = (1.0f64 / (1.0 + t_diff as f64)) as f32;

            let mut dir: f32 = rx * grad_t_x + ry * grad_t_y;
            if dir.abs() <= 0.01 {
                dir = 0.000001;
            }
            let weight: f32 = (dst * lev * dir).abs();

            let mut grad_ix: f32 = 0.0;
            let r_ok = kxu + 1 < w && flags[kyu * w + kxu + 1] != INSIDE;
            let l_ok = kxu > 0 && flags[kyu * w + kxu - 1] != INSIDE;
            if r_ok {
                if l_ok {
                    grad_ix =
                        (pixels[kyu * w + kxu + 1] as f32 - pixels[kyu * w + kxu - 1] as f32) * 2.0;
                } else {
                    grad_ix = pixels[kyu * w + kxu + 1] as f32 - pixels[kidx] as f32;
                }
            } else if l_ok {
                grad_ix = pixels[kidx] as f32 - pixels[kyu * w + kxu - 1] as f32;
            }

            let mut grad_iy: f32 = 0.0;
            let d_ok = kyu + 1 < h && flags[(kyu + 1) * w + kxu] != INSIDE;
            let u_ok = kyu > 0 && flags[(kyu - 1) * w + kxu] != INSIDE;
            if d_ok {
                if u_ok {
                    grad_iy = (pixels[(kyu + 1) * w + kxu] as f32
                        - pixels[(kyu - 1) * w + kxu] as f32)
                        * 2.0;
                } else {
                    grad_iy = pixels[(kyu + 1) * w + kxu] as f32 - pixels[kidx] as f32;
                }
            } else if u_ok {
                grad_iy = pixels[kidx] as f32 - pixels[(kyu - 1) * w + kxu] as f32;
            }

            ia += weight * pixels[kidx] as f32;
            jx -= weight * (grad_ix * rx);
            jy -= weight * (grad_iy * ry);
            s += weight;
        }
    }

    let sat: f32 = ia / s + (jx + jy) / ((jx * jx + jy * jy).sqrt() + 1.0e-20);
    sat.clamp(0.0, 255.0).round() as u8
}
