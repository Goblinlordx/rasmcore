//! Filter: kuwahara (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "kuwahara",
    category = "spatial",
    reference = "Kuwahara 1976 edge-preserving smoothing"
)]
pub struct KuwaharaParams {
    pub radius: u32,
}

impl CpuFilter for KuwaharaParams {
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
        let pixels = pixels.as_slice();
        let radius = self.radius;

        if radius == 0 {
            return Ok(pixels.to_vec());
        }
        validate_format(info.format)?;

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);
        let width = (radius + 1) as i32; // quadrant side length

        // Step 1: Pre-blur with Gaussian in f32 precision (matches IM's Q16-HDRI blur)
        // IM default sigma = radius - 0.5, kernel half-width = radius
        let sigma = (radius as f64 - 0.5).max(0.5);
        let blurred_f32 = gaussian_blur_f32(pixels, w, h, ch, radius as usize, sigma);

        // Convert blurred Q16 data to [0,1] samples for output
        // gaussian_blur_f32 returns values in Q16 scale (0-65535)
        let q16_to_unit = 1.0 / 65535.0;

        let mut out_samples = vec![0.0f32; w * h * ch];
        let wi = w as i32;
        let hi = h as i32;

        for y in 0..h {
            for x in 0..w {
                // IM quadrants: 4 non-overlapping regions of size width*width
                let quadrants: [(i32, i32); 4] = [
                    (x as i32 - width + 1, y as i32 - width + 1), // Q0: top-left
                    (x as i32, y as i32 - width + 1),             // Q1: top-right
                    (x as i32 - width + 1, y as i32),             // Q2: bottom-left
                    (x as i32, y as i32),                         // Q3: bottom-right
                ];

                let mut min_var = f64::MAX;
                let mut best_cx = x as f64;
                let mut best_cy = y as f64;

                for &(qx, qy) in &quadrants {
                    let n = (width * width) as f64;
                    let mut mean_ch = [0.0f64; 4];
                    for ky in 0..width {
                        let sy = (qy + ky).clamp(0, hi - 1) as usize;
                        for kx in 0..width {
                            let sx = (qx + kx).clamp(0, wi - 1) as usize;
                            let off = (sy * w + sx) * ch;
                            for c in 0..ch {
                                mean_ch[c] += blurred_f32[off + c] as f64;
                            }
                        }
                    }
                    for val in mean_ch.iter_mut().take(ch) {
                        *val /= n;
                    }
                    let mean_luma = if ch >= 3 {
                        0.212656 * mean_ch[0] + 0.715158 * mean_ch[1] + 0.072186 * mean_ch[2]
                    } else {
                        mean_ch[0]
                    };

                    let mut variance = 0.0f64;
                    for ky in 0..width {
                        let sy = (qy + ky).clamp(0, hi - 1) as usize;
                        for kx in 0..width {
                            let sx = (qx + kx).clamp(0, wi - 1) as usize;
                            let off = (sy * w + sx) * ch;
                            let pixel_luma = if ch >= 3 {
                                0.212656 * blurred_f32[off] as f64
                                    + 0.715158 * blurred_f32[off + 1] as f64
                                    + 0.072186 * blurred_f32[off + 2] as f64
                            } else {
                                blurred_f32[off] as f64
                            };
                            let d = pixel_luma - mean_luma;
                            variance += d * d;
                        }
                    }

                    if variance < min_var {
                        min_var = variance;
                        best_cx = qx as f64 + width as f64 / 2.0;
                        best_cy = qy as f64 + width as f64 / 2.0;
                    }
                }

                // Bilinear interpolation at sub-pixel center from f32 blurred data
                let sx = best_cx.clamp(0.0, (w - 1) as f64);
                let sy = best_cy.clamp(0.0, (h - 1) as f64);
                let x0i = sx.floor() as usize;
                let y0i = sy.floor() as usize;
                let x1i = (x0i + 1).min(w - 1);
                let y1i = (y0i + 1).min(h - 1);
                let fx = (sx - x0i as f64) as f32;
                let fy = (sy - y0i as f64) as f32;

                let out_off = (y * w + x) * ch;
                for c in 0..ch {
                    let p00 = blurred_f32[(y0i * w + x0i) * ch + c];
                    let p10 = blurred_f32[(y0i * w + x1i) * ch + c];
                    let p01 = blurred_f32[(y1i * w + x0i) * ch + c];
                    let p11 = blurred_f32[(y1i * w + x1i) * ch + c];
                    let v = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                    // Scale from Q16 (0-65535) to [0,1]
                    out_samples[out_off + c] = (v * q16_to_unit).clamp(0.0, 1.0);
                }
            }
        }

        Ok(f32_samples_to_pixels(&out_samples, info.format))
    }
}
