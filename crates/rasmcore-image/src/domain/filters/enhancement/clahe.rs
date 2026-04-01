//! Filter: clahe (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply CLAHE — local adaptive contrast enhancement.
///
/// Divides the image into `tile_grid` x `tile_grid` tiles, equalizes each
/// tile's histogram with a clip limit, then bilinear interpolates between
/// tiles for smooth transitions. Grayscale only (convert first for color).
///
/// - `clip_limit`: contrast amplification limit (2.0-4.0 typical, higher = more contrast)
/// - `tile_grid`: number of tiles per dimension (8 = 8x8 grid, OpenCV default)
#[rasmcore_macros::register_filter(
    name = "clahe",
    category = "enhancement",
    reference = "Zuiderveld 1994 contrast-limited adaptive histogram equalization"
)]
pub fn clahe(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ClaheParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let clip_limit = config.clip_limit;
    let tile_grid = config.tile_grid;

    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "CLAHE requires Gray8 input".into(),
        ));
    }
    if tile_grid == 0 || clip_limit < 1.0 {
        return Err(ImageError::InvalidParameters(
            "tile_grid must be > 0, clip_limit must be >= 1.0".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let grid = tile_grid as usize;
    let tile_w = w.div_ceil(grid);
    let tile_h = h.div_ceil(grid);

    // Build per-tile CDF lookup tables
    let mut tile_luts = vec![[0u8; 256]; grid * grid];

    for ty in 0..grid {
        for tx in 0..grid {
            let x0 = tx * tile_w;
            let y0 = ty * tile_h;
            let x1 = (x0 + tile_w).min(w);
            let y1 = (y0 + tile_h).min(h);
            let tile_pixels = (x1 - x0) * (y1 - y0);
            if tile_pixels == 0 {
                continue;
            }

            // Histogram
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[pixels[y * w + x] as usize] += 1;
                }
            }

            // Clip histogram and redistribute (matching OpenCV exactly)
            // No special case for single-value tiles — OpenCV processes all tiles uniformly.
            let clip = ((clip_limit * tile_pixels as f32) / 256.0) as u32;
            let clip = clip.max(1);
            let mut clipped = 0u32;
            for h in hist.iter_mut() {
                if *h > clip {
                    clipped += *h - clip;
                    *h = clip;
                }
            }
            // Redistribute: uniform batch + stepped residual (OpenCV algorithm)
            let redist_batch = clipped / 256;
            let residual = clipped - redist_batch * 256;
            for h in hist.iter_mut() {
                *h += redist_batch;
            }
            if residual > 0 {
                let step = (256 / residual as usize).max(1);
                let mut remaining = residual as usize;
                let mut i = 0;
                while i < 256 && remaining > 0 {
                    hist[i] += 1;
                    remaining -= 1;
                    i += step;
                }
            }

            // Build CDF → LUT (OpenCV formula: lut[i] = saturate(sum * lutScale))
            let lut_scale = 255.0f32 / tile_pixels as f32;
            let lut = &mut tile_luts[ty * grid + tx];
            let mut sum = 0u32;
            for i in 0..256 {
                sum += hist[i];
                let v = (sum as f32 * lut_scale).round();
                lut[i] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Apply with bilinear interpolation (matching OpenCV exactly)
    let inv_tw = 1.0f32 / tile_w as f32;
    let inv_th = 1.0f32 / tile_h as f32;
    let mut result = vec![0u8; pixels.len()];
    for y in 0..h {
        let fy = y as f32 * inv_th - 0.5;
        let ty1i = fy.floor() as isize;
        let ty2i = ty1i + 1;
        let ya = fy - ty1i as f32;
        let ya1 = 1.0 - ya;
        let ty1 = ty1i.clamp(0, grid as isize - 1) as usize;
        let ty2 = (ty2i as usize).min(grid - 1);

        for x in 0..w {
            let fx = x as f32 * inv_tw - 0.5;
            let tx1i = fx.floor() as isize;
            let tx2i = tx1i + 1;
            let xa = fx - tx1i as f32;
            let xa1 = 1.0 - xa;
            let tx1 = tx1i.clamp(0, grid as isize - 1) as usize;
            let tx2 = (tx2i as usize).min(grid - 1);

            let val = pixels[y * w + x] as usize;

            // Bilinear interpolation of 4 tile LUTs (OpenCV order)
            let v = (tile_luts[ty1 * grid + tx1][val] as f32 * xa1
                + tile_luts[ty1 * grid + tx2][val] as f32 * xa)
                * ya1
                + (tile_luts[ty2 * grid + tx1][val] as f32 * xa1
                    + tile_luts[ty2 * grid + tx2][val] as f32 * xa)
                    * ya;

            result[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}
