//! Image inpainting — fill masked regions using surrounding pixel information.
//!
//! Two algorithms, both FMM-based, matching OpenCV's `cv2.inpaint()`:
//! - **Telea (FMM):** Weighted interpolation using distance, level-set, and time-gradient
//!   alignment, plus a first-order gradient correction term (Telea 2004).
//! - **Navier-Stokes (FMM):** Weighted interpolation using distance^4 and image-gradient
//!   alignment. Pure weighted average without gradient correction.

mod exemplar;
mod pde;

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
        InpaintMethod::Telea => pde::telea_inpaint(pixels, w, h, mask, radius),
        InpaintMethod::NavierStokes => exemplar::navier_stokes_inpaint(pixels, w, h, mask, radius),
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

pub(crate) const KNOWN: u8 = 0;
pub(crate) const BAND: u8 = 1;
pub(crate) const INSIDE: u8 = 2;

#[derive(Clone, Copy)]
pub(crate) struct FmmEntry {
    pub dist: f32,
    pub seq: u32,
    pub x: usize,
    pub y: usize,
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

pub(crate) fn solve_eikonal(dist: &[f32], flags: &[u8], w: usize, h: usize, x: usize, y: usize) -> f32 {
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

    let s1 = fm_solve_pair(up.0, up.1, left.0, left.1);
    let s2 = fm_solve_pair(down.0, down.1, left.0, left.1);
    let s3 = fm_solve_pair(up.0, up.1, right.0, right.1);
    let s4 = fm_solve_pair(down.0, down.1, right.0, right.1);

    s1.min(s2).min(s3).min(s4)
}

pub(crate) fn fmm_init_padded(
    mask: &[u8],
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<f32>, BinaryHeap<FmmEntry>, usize, usize, u32) {
    let pw = w + 2;
    let ph = h + 2;
    let pn = pw * ph;
    let mut flags = vec![KNOWN; pn];
    let mut dist = vec![0.0f32; pn];

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

pub(crate) fn neighbors_4(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
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
pub(crate) fn neighbors_4_opencv(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    if y >= 2 {
        n.push((x, y - 1));
    }
    if x >= 2 {
        n.push((x - 1, y));
    }
    if y + 1 < h {
        n.push((x, y + 1));
    }
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
        mask[4 * 8 + 4] = 255;
        pixels[4 * 8 + 4] = 0;

        let result = inpaint(&pixels, &gray_info(8, 8), &mask, 3.0, InpaintMethod::Telea).unwrap();

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

        let info = ImageInfo { width: w, height: h, format: PixelFormat::Gray8, color_space: ColorSpace::Srgb };
        let ours = inpaint(&img, &info, &mask, 3.0, InpaintMethod::Telea).unwrap();

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
        assert!(m == 0.0 && mx == 0, "Telea MAE={m:.4}, max_err={mx} vs OpenCV — must be exact");
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

        let info = ImageInfo { width: w, height: h, format: PixelFormat::Gray8, color_space: ColorSpace::Srgb };
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
        assert!(m == 0.0 && mx == 0, "NS MAE={m:.4}, max_err={mx} vs OpenCV — must be exact");
    }

    #[test]
    fn telea_gradient_parity_vs_opencv() {
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

        let info = ImageInfo { width: w, height: h, format: PixelFormat::Gray8, color_space: ColorSpace::Srgb };
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
        assert!(mx <= 6, "Telea gradient max_err={mx} vs OpenCV — should be <= 6");
    }

    #[test]
    fn ns_gradient_parity_vs_opencv() {
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

        let info = ImageInfo { width: w, height: h, format: PixelFormat::Gray8, color_space: ColorSpace::Srgb };
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
        assert!(m == 0.0 && mx == 0, "NS gradient MAE={m:.4}, max_err={mx} vs OpenCV — must be exact");
    }
}
