//! Canny filter — proper implementation with Gaussian blur, NMS, and hysteresis.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::convolve3x3;
use super::{SOBEL_X, SOBEL_Y};

/// Canny edge detection — Gaussian blur + Sobel gradient + non-maximum
/// suppression + double threshold with hysteresis.
///
/// Matches OpenCV's cv2.Canny algorithm:
/// 1. Gaussian pre-blur (sigma=1.4)
/// 2. Sobel gradient magnitude and direction
/// 3. Non-maximum suppression (thin edges to 1px)
/// 4. Double threshold: strong edges (>= high), weak edges (>= low)
/// 5. Hysteresis: promote weak edges connected to strong edges
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "canny", category = "spatial", cost = "O(n)")]
pub struct Canny {
    /// Low threshold (edges below this are suppressed)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.1)]
    pub low: f32,
    /// High threshold (edges above this are strong edges)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.3)]
    pub high: f32,
}

impl Filter for Canny {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Step 1: Gaussian pre-blur (sigma=1.4, radius=ceil(3*1.4)=5, kernel 11x11)
        let blurred = gaussian_blur_rgba(input, w, h, 1.4);

        // Step 2: Sobel gradient on blurred image (on luminance)
        let mut mag = vec![0.0f32; n];
        let mut dir = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(&blurred, w, h, x as i32, y as i32, &SOBEL_X);
                let gy = convolve3x3(&blurred, w, h, x as i32, y as i32, &SOBEL_Y);
                let idx = y * w + x;
                mag[idx] = (gx * gx + gy * gy).sqrt();
                dir[idx] = gy.atan2(gx);
            }
        }

        // Step 3: Non-maximum suppression
        let mut nms = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let angle = dir[idx].to_degrees().rem_euclid(180.0);

                let (dx, dy): (i32, i32) = if angle < 22.5 || angle >= 157.5 {
                    (1, 0)
                } else if angle < 67.5 {
                    (1, 1)
                } else if angle < 112.5 {
                    (0, 1)
                } else {
                    (-1, 1)
                };

                let x1 = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let y1 = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                let x2 = (x as i32 - dx).clamp(0, w as i32 - 1) as usize;
                let y2 = (y as i32 - dy).clamp(0, h as i32 - 1) as usize;

                let m = mag[idx];
                if m >= mag[y1 * w + x1] && m >= mag[y2 * w + x2] {
                    nms[idx] = m;
                }
            }
        }

        // Step 4: Double threshold + hysteresis
        let mut edge_type = vec![0u8; n]; // 0=none, 1=weak, 2=strong
        for i in 0..n {
            if nms[i] >= self.high {
                edge_type[i] = 2;
            } else if nms[i] >= self.low {
                edge_type[i] = 1;
            }
        }

        // Hysteresis: promote weak edges connected to strong edges
        let mut changed = true;
        while changed {
            changed = false;
            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    if edge_type[idx] != 1 {
                        continue;
                    }
                    'neighbors: for ny in -1..=1i32 {
                        for nx in -1..=1i32 {
                            if nx == 0 && ny == 0 { continue; }
                            let sx = x as i32 + nx;
                            let sy = y as i32 + ny;
                            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                                if edge_type[sy as usize * w + sx as usize] == 2 {
                                    edge_type[idx] = 2;
                                    changed = true;
                                    break 'neighbors;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 5: Build output (strong edges = 1.0, else 0.0)
        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let v = if edge_type[idx] == 2 { 1.0 } else { 0.0 };
                let pidx = idx * 4;
                out[pidx] = v;
                out[pidx + 1] = v;
                out[pidx + 2] = v;
                out[pidx + 3] = input[pidx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        5 // Gaussian blur (r=5) + Sobel (r=1) + NMS (r=1) + hysteresis propagation
    }
}

/// Gaussian blur on RGBA (processes only luminance-affecting channels).
/// Used as Canny preprocessing step.
fn gaussian_blur_rgba(input: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as i32;
    let size = (2 * r + 1) as usize;

    // Build 1D kernel
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for i in 0..size {
        let d = (i as i32 - r) as f32;
        let v = (-d * d / (2.0 * sigma * sigma)).exp();
        kernel[i] = v;
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Horizontal pass
    let mut temp = input.to_vec();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for ki in 0..size {
                let sx = (x as i32 + ki as i32 - r).clamp(0, w as i32 - 1) as usize;
                let idx = (y * w + sx) * 4;
                acc[0] += input[idx] * kernel[ki];
                acc[1] += input[idx + 1] * kernel[ki];
                acc[2] += input[idx + 2] * kernel[ki];
            }
            let oidx = (y * w + x) * 4;
            temp[oidx] = acc[0];
            temp[oidx + 1] = acc[1];
            temp[oidx + 2] = acc[2];
        }
    }

    // Vertical pass
    let mut out = temp.clone();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for ki in 0..size {
                let sy = (y as i32 + ki as i32 - r).clamp(0, h as i32 - 1) as usize;
                let idx = (sy * w + x) * 4;
                acc[0] += temp[idx] * kernel[ki];
                acc[1] += temp[idx + 1] * kernel[ki];
                acc[2] += temp[idx + 2] * kernel[ki];
            }
            let oidx = (y * w + x) * 4;
            out[oidx] = acc[0];
            out[oidx + 1] = acc[1];
            out[oidx + 2] = acc[2];
        }
    }
    out
}

// GPU shader kept as simplified version for now — proper GPU Canny
// requires multiple passes (blur, gradient, NMS, hysteresis).
// TODO: implement multi-pass GPU Canny
