use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{median_cut_palette, nearest_color};

/// Floyd-Steinberg error diffusion dithering.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dither_floyd_steinberg", category = "color", cost = "O(n)")]
pub struct DitherFloydSteinberg {
    /// Max palette size (2-256).
    #[param(min = 2, max = 256, default = 16)]
    pub max_colors: u32,
}

impl Filter for DitherFloydSteinberg {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let w = width as usize;
        let h = height as usize;
        let mut buf = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let (r, g, b) = (buf[idx], buf[idx + 1], buf[idx + 2]);
                let (nr, ng, nb) = nearest_color(&palette, r, g, b);
                let er = r - nr;
                let eg = g - ng;
                let eb = b - nb;
                buf[idx] = nr;
                buf[idx + 1] = ng;
                buf[idx + 2] = nb;
                // Diffuse error
                let diffuse = |buf: &mut [f32], bx: usize, by: usize, weight: f32| {
                    let i = (by * w + bx) * 4;
                    buf[i] += er * weight;
                    buf[i + 1] += eg * weight;
                    buf[i + 2] += eb * weight;
                };
                if x + 1 < w {
                    diffuse(&mut buf, x + 1, y, 7.0 / 16.0);
                }
                if y + 1 < h {
                    if x > 0 {
                        diffuse(&mut buf, x - 1, y + 1, 3.0 / 16.0);
                    }
                    diffuse(&mut buf, x, y + 1, 5.0 / 16.0);
                    if x + 1 < w {
                        diffuse(&mut buf, x + 1, y + 1, 1.0 / 16.0);
                    }
                }
            }
        }
        Ok(buf)
    }
}

impl GpuFilter for DitherFloydSteinberg {
    fn shader_body(&self) -> &str {
        // GPU uses ordered Bayer dither (parallel) instead of sequential Floyd-Steinberg
        include_str!("../../shaders/ordered_dither.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let levels = 1.0 / self.max_colors.max(2) as f32;
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&levels.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dither_floyd_steinberg_reduces_colors() {
        let mut input = Vec::new();
        for i in 0..64 {
            let v = i as f32 / 63.0;
            input.extend_from_slice(&[v, v, v, 1.0]);
        }
        let f = DitherFloydSteinberg { max_colors: 2 };
        let out = f.compute(&input, 8, 8).unwrap();
        let mut unique = std::collections::HashSet::new();
        for pixel in out.chunks_exact(4) {
            let key = (
                (pixel[0] * 1000.0) as i32,
                (pixel[1] * 1000.0) as i32,
                (pixel[2] * 1000.0) as i32,
            );
            unique.insert(key);
        }
        assert!(unique.len() <= 2, "Should have <=2 colors, got {}", unique.len());
    }
}
