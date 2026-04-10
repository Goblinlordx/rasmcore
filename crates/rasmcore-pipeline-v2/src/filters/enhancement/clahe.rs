use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use crate::filters::helpers::luminance;

/// CLAHE — local adaptive histogram equalization on luminance.
///
/// Operates on luminance channel with bilinear interpolation between tiles.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clahe", category = "enhancement", cost = "O(n)")]
pub struct Clahe {
    #[param(min = 1, max = 32, default = 8)]
    pub tile_grid: u32,
    #[param(min = 0.0, max = 100.0, default = 2.0)]
    pub clip_limit: f32,
}

impl Filter for Clahe {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let grid = self.tile_grid as usize;
        if grid == 0 {
            return Ok(input.to_vec());
        }

        let tile_w = w / grid;
        let tile_h = h / grid;
        if tile_w == 0 || tile_h == 0 {
            return Ok(input.to_vec());
        }

        let npixels_per_tile = tile_w * tile_h;
        let clip = (self.clip_limit * npixels_per_tile as f32 / 256.0).max(1.0) as u32;

        // Extract luminance
        let luma: Vec<f32> = input
            .chunks_exact(4)
            .map(|p| luminance(p[0], p[1], p[2]).clamp(0.0, 1.0))
            .collect();

        // Build per-tile LUTs
        let mut tile_luts = vec![[0.0f32; 256]; grid * grid];
        for ty in 0..grid {
            for tx in 0..grid {
                let mut hist = [0u32; 256];
                let y0 = ty * tile_h;
                let x0 = tx * tile_w;
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let py = (y0 + dy).min(h - 1);
                        let px = (x0 + dx).min(w - 1);
                        let bin = (luma[py * w + px] * 255.0) as usize;
                        hist[bin.min(255)] += 1;
                    }
                }

                // Clip histogram and redistribute
                let mut excess = 0u32;
                for h in &mut hist {
                    if *h > clip {
                        excess += *h - clip;
                        *h = clip;
                    }
                }
                let per_bin = excess / 256;
                let remainder = excess % 256;
                for (i, h) in hist.iter_mut().enumerate() {
                    *h += per_bin + if (i as u32) < remainder { 1 } else { 0 };
                }

                // Build CDF -> LUT
                let mut cdf = [0u32; 256];
                cdf[0] = hist[0];
                for i in 1..256 {
                    cdf[i] = cdf[i - 1] + hist[i];
                }
                let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
                let denom = (npixels_per_tile as u32).saturating_sub(cdf_min).max(1);

                let lut = &mut tile_luts[ty * grid + tx];
                for i in 0..256 {
                    lut[i] = (cdf[i] - cdf_min) as f32 / denom as f32;
                }
            }
        }

        // Apply with bilinear interpolation between tiles
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let tx_f = (x as f32 / tile_w as f32 - 0.5).clamp(0.0, (grid - 1) as f32);
                let ty_f = (y as f32 / tile_h as f32 - 0.5).clamp(0.0, (grid - 1) as f32);
                let tx0 = tx_f as usize;
                let ty0 = ty_f as usize;
                let tx1 = (tx0 + 1).min(grid - 1);
                let ty1 = (ty0 + 1).min(grid - 1);
                let fx = tx_f - tx0 as f32;
                let fy = ty_f - ty0 as f32;

                let bin = (luma[y * w + x] * 255.0) as usize;
                let bin = bin.min(255);

                let v00 = tile_luts[ty0 * grid + tx0][bin];
                let v10 = tile_luts[ty0 * grid + tx1][bin];
                let v01 = tile_luts[ty1 * grid + tx0][bin];
                let v11 = tile_luts[ty1 * grid + tx1][bin];

                let new_luma = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                let old_luma = luma[y * w + x].max(1e-10);
                let ratio = new_luma / old_luma;

                let idx = (y * w + x) * 4;
                out[idx] *= ratio;
                out[idx + 1] *= ratio;
                out[idx + 2] *= ratio;
            }
        }

        Ok(out)
    }
}

// ── Clahe GPU (single-pass with pre-computed tile LUTs) ─────────────────

use crate::gpu_shaders::enhancement as enh_shaders;

impl GpuFilter for Clahe {
    fn shader_body(&self) -> &str {
        enh_shaders::CLAHE_APPLY
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let grid = self.tile_grid;
        let tile_w = w / grid.max(1);
        let tile_h = h / grid.max(1);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&grid.to_le_bytes());
        buf.extend_from_slice(&tile_w.to_le_bytes());
        buf.extend_from_slice(&tile_h.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Tile LUTs will be computed on CPU and passed here at dispatch time.
        // For now return empty — the executor handles CPU->GPU LUT upload.
        vec![]
    }
}
