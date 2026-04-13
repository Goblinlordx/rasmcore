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
        // OpenCV LUT scale: (histSize - 1) / tileSizeTotal = 255 / total
        let lut_scale = 255.0 / npixels_per_tile as f32;

        // Extract luminance, quantize to u8 (round-to-nearest, matching OpenCV)
        let luma: Vec<f32> = input
            .chunks_exact(4)
            .map(|p| luminance(p[0], p[1], p[2]).clamp(0.0, 1.0))
            .collect();
        let luma_u8: Vec<u8> = luma.iter()
            .map(|&v| (v * 255.0 + 0.5).min(255.0) as u8)
            .collect();

        // Build per-tile LUTs as u8 (matching OpenCV's saturate_cast<u8>)
        let mut tile_luts = vec![[0u8; 256]; grid * grid];
        for ty in 0..grid {
            for tx in 0..grid {
                let mut hist = [0u32; 256];
                let y0 = ty * tile_h;
                let x0 = tx * tile_w;
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let py = (y0 + dy).min(h - 1);
                        let px = (x0 + dx).min(w - 1);
                        hist[luma_u8[py * w + px] as usize] += 1;
                    }
                }

                // Clip histogram and redistribute (OpenCV single-pass)
                let mut excess = 0u32;
                for h in &mut hist {
                    if *h > clip {
                        excess += *h - clip;
                        *h = clip;
                    }
                }
                let redist_batch = excess / 256;
                let residual = excess - redist_batch * 256;
                for h in hist.iter_mut() {
                    *h += redist_batch;
                }
                // Distribute residual with stride (OpenCV: stride = histSize / residual)
                if residual > 0 {
                    let step = (256 / residual).max(1) as usize;
                    let mut remaining = residual;
                    let mut i = 0;
                    while i < 256 && remaining > 0 {
                        hist[i] += 1;
                        remaining -= 1;
                        i += step;
                    }
                }

                // Build CDF -> LUT as u8 (OpenCV: round(CDF * 255 / total))
                let mut cdf_sum = 0u32;
                let lut = &mut tile_luts[ty * grid + tx];
                for i in 0..256 {
                    cdf_sum += hist[i];
                    // saturate_cast<u8>(sum * lutScale) — round-to-nearest
                    lut[i] = (cdf_sum as f32 * lut_scale + 0.5).min(255.0) as u8;
                }
            }
        }

        // Bilinear interpolation (matching OpenCV: txf = x / tileWidth - 0.5)
        let inv_tw = 1.0 / tile_w as f32;
        let inv_th = 1.0 / tile_h as f32;

        let mut out = input.to_vec();
        for y in 0..h {
            let tyf = y as f32 * inv_th - 0.5;
            let ty1 = (tyf.floor() as i32).max(0).min(grid as i32 - 1) as usize;
            let ty2 = (ty1 + 1).min(grid - 1);
            let ya = tyf - ty1 as f32;
            let ya = ya.clamp(0.0, 1.0);
            let ya1 = 1.0 - ya;

            for x in 0..w {
                let txf = x as f32 * inv_tw - 0.5;
                let tx1 = (txf.floor() as i32).max(0).min(grid as i32 - 1) as usize;
                let tx2 = (tx1 + 1).min(grid - 1);
                let xa = txf - tx1 as f32;
                let xa = xa.clamp(0.0, 1.0);
                let xa1 = 1.0 - xa;

                let bin = luma_u8[y * w + x] as usize;

                // Interpolate between u8 LUT values (matching OpenCV)
                let v_top = tile_luts[ty1 * grid + tx1][bin] as f32 * xa1
                    + tile_luts[ty1 * grid + tx2][bin] as f32 * xa;
                let v_bot = tile_luts[ty2 * grid + tx1][bin] as f32 * xa1
                    + tile_luts[ty2 * grid + tx2][bin] as f32 * xa;
                let new_luma_u8 = v_top * ya1 + v_bot * ya;

                // Convert back to [0,1] and apply as ratio
                let new_luma = new_luma_u8 / 255.0;
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
