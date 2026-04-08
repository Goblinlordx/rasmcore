//! LUT parsing — .cube format (1D and 3D) and Hald CLUT images.

use crate::fusion::Clut3D;
use crate::node::PipelineError;

/// Parse a .cube format LUT (1D or 3D) from text content into a Clut3D.
///
/// Supports both `LUT_1D_SIZE` and `LUT_3D_SIZE` directives plus
/// TITLE, DOMAIN_MIN, DOMAIN_MAX.
///
/// 1D LUTs are converted to 3D CLUTs by applying the per-channel transfer
/// functions independently: `out(r,g,b) = (lut_r(r), lut_g(g), lut_b(b))`.
/// The 3D grid size is clamped to a reasonable maximum (65) for 1D->3D
/// conversion.
pub fn parse_cube_lut(content: &str) -> Result<Clut3D, PipelineError> {
    let mut grid_size_3d: Option<u32> = None;
    let mut grid_size_1d: Option<u32> = None;
    let mut data: Vec<f32> = Vec::new();
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            grid_size_3d = Some(
                rest.trim()
                    .parse::<u32>()
                    .map_err(|_| PipelineError::InvalidParams("invalid LUT_3D_SIZE".into()))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_1D_SIZE") {
            grid_size_1d = Some(
                rest.trim()
                    .parse::<u32>()
                    .map_err(|_| PipelineError::InvalidParams("invalid LUT_1D_SIZE".into()))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
            let vals: Vec<f32> = rest
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() == 3 {
                domain_min = [vals[0], vals[1], vals[2]];
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
            let vals: Vec<f32> = rest
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() == 3 {
                domain_max = [vals[0], vals[1], vals[2]];
            }
            continue;
        }
        // Data line: three floats
        let vals: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() == 3 {
            // Normalize to [0,1] from domain
            for i in 0..3 {
                let range = (domain_max[i] - domain_min[i]).max(1e-6);
                data.push((vals[i] - domain_min[i]) / range);
            }
        }
    }

    if let Some(n) = grid_size_3d {
        // Standard 3D LUT
        let expected = (n * n * n * 3) as usize;
        if data.len() != expected {
            return Err(PipelineError::InvalidParams(format!(
                "expected {expected} values for {n}^3 LUT, got {}",
                data.len()
            )));
        }
        Ok(Clut3D { grid_size: n, data })
    } else if let Some(n) = grid_size_1d {
        // 1D LUT — N entries of (r, g, b) transfer values
        let expected = (n * 3) as usize;
        if data.len() != expected {
            return Err(PipelineError::InvalidParams(format!(
                "expected {expected} values for 1D LUT size {n}, got {}",
                data.len()
            )));
        }
        Ok(lut_1d_to_clut3d(&data, n))
    } else {
        Err(PipelineError::InvalidParams(
            "missing LUT_3D_SIZE or LUT_1D_SIZE in .cube file".into(),
        ))
    }
}

/// Convert a 1D LUT (N entries x 3 channels) to a 3D CLUT.
///
/// The 1D transfer function is applied independently per channel:
/// `out(r,g,b) = (lut_r(r), lut_g(g), lut_b(b))`.
///
/// Grid size for the 3D CLUT is min(n, 65) to keep memory reasonable.
fn lut_1d_to_clut3d(data: &[f32], n: u32) -> Clut3D {
    let n = n as usize;

    // Separate into per-channel arrays for interpolation
    let mut lut_r = Vec::with_capacity(n);
    let mut lut_g = Vec::with_capacity(n);
    let mut lut_b = Vec::with_capacity(n);
    for i in 0..n {
        lut_r.push(data[i * 3]);
        lut_g.push(data[i * 3 + 1]);
        lut_b.push(data[i * 3 + 2]);
    }

    // Sample 1D LUT with linear interpolation
    let sample_1d = |lut: &[f32], t: f32| -> f32 {
        let max = (lut.len() - 1) as f32;
        let idx = (t * max).clamp(0.0, max);
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(lut.len() - 1);
        let frac = idx - lo as f32;
        lut[lo] + frac * (lut[hi] - lut[lo])
    };

    // Build 3D CLUT — grid size capped at 65 for memory
    let grid = (n as u32).min(65);
    Clut3D::from_fn(grid, |r, g, b| {
        (sample_1d(&lut_r, r), sample_1d(&lut_g, g), sample_1d(&lut_b, b))
    })
}

/// Decode a Hald CLUT image into a Clut3D.
///
/// A Hald CLUT of level L is an image of dimensions L^3 x L^3 containing
/// an L^2 x L^2 x L^2 color lookup table encoded as an identity-structure
/// image. Each pixel's position maps to an (r, g, b) input coordinate,
/// and the pixel's color is the output.
///
/// The input `pixels` must be f32 RGBA (4 channels per pixel).
/// `width` and `height` must both equal L^3 for some integer L >= 2.
pub fn parse_hald_lut(pixels: &[f32], width: u32, height: u32) -> Result<Clut3D, PipelineError> {
    if width != height {
        return Err(PipelineError::InvalidParams(
            "Hald CLUT image must be square".into(),
        ));
    }

    // Find level L such that L^3 == width
    let dim = width as usize;
    let level = (dim as f64).cbrt().round() as usize;
    if level * level * level != dim {
        return Err(PipelineError::InvalidParams(format!(
            "Hald image dimension {dim} is not a perfect cube (L^3)"
        )));
    }
    if level < 2 {
        return Err(PipelineError::InvalidParams(
            "Hald level must be at least 2".into(),
        ));
    }

    let grid_size = level * level; // L^2
    let total = grid_size * grid_size * grid_size;

    let pixel_count = (width as usize) * (height as usize);
    if pixels.len() < pixel_count * 4 {
        return Err(PipelineError::InvalidParams(format!(
            "Hald image needs {} pixels ({} f32s), got {} f32s",
            pixel_count,
            pixel_count * 4,
            pixels.len()
        )));
    }
    if total > pixel_count {
        return Err(PipelineError::InvalidParams(format!(
            "Hald level {level} needs {total} entries but image has only {pixel_count} pixels"
        )));
    }

    // Extract RGB from each pixel in scanline order.
    // Hald layout: pixel index i maps to:
    //   r = i % grid_size
    //   g = (i / grid_size) % grid_size
    //   b = i / (grid_size * grid_size)
    let mut data = Vec::with_capacity(total * 3);
    for i in 0..total {
        let base = i * 4;
        data.push(pixels[base]);     // R
        data.push(pixels[base + 1]); // G
        data.push(pixels[base + 2]); // B
    }

    Ok(Clut3D {
        grid_size: grid_size as u32,
        data,
    })
}
