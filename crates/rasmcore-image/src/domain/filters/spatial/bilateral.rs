//! Filter: bilateral (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Edge-preserving bilateral filter — pixel-exact match with OpenCV 4.13.
///
/// Uses circular kernel mask, f32 accumulation, BORDER_REFLECT_101 padding,
/// pre-computed spatial/color weight LUTs, and L1 color norm for RGB.
///
/// - `diameter`: filter size (use 0 for auto from sigma_space; typical 5-9)
/// - `sigma_color`: filter sigma in the color/intensity space (10-150 typical)
/// - `sigma_space`: filter sigma in coordinate space (10-150 typical)

/// Parameters for bilateral filter.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BilateralParams {
    /// Filter size (0 for auto from sigma_space; typical 5-9)
    #[param(min = 0, max = 31, step = 2, default = 5, hint = "rc.log_slider")]
    pub diameter: u32,
    /// Filter sigma in color/intensity space (10-150 typical)
    #[param(
        min = 1.0,
        max = 300.0,
        step = 1.0,
        default = 75.0,
        hint = "rc.log_slider"
    )]
    pub sigma_color: f32,
    /// Filter sigma in coordinate space (10-150 typical)
    #[param(
        min = 1.0,
        max = 300.0,
        step = 1.0,
        default = 75.0,
        hint = "rc.log_slider"
    )]
    pub sigma_space: f32,
}

#[rasmcore_macros::register_filter(
    name = "bilateral", gpu = "true",
    category = "spatial",
    group = "denoise",
    variant = "bilateral",
    reference = "Tomasi & Manduchi 1998"
)]
pub fn bilateral(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BilateralParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.diameter / 2 + 1;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let diameter = config.diameter;
    let sigma_color = config.sigma_color;
    let sigma_space = config.sigma_space;

    if info.format != PixelFormat::Gray8 && info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "bilateral filter requires Gray8 or Rgb8".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let channels = if info.format == PixelFormat::Gray8 {
        1
    } else {
        3
    };
    let radius = if diameter > 0 {
        (diameter as usize | 1) / 2
    } else {
        (sigma_space * 1.5).round() as usize
    };

    // Pre-compute spatial weight LUT + offsets (CIRCULAR mask, matching OpenCV)
    let gauss_space_coeff: f32 = -0.5 / (sigma_space * sigma_space);
    let mut space_weight: Vec<f32> = Vec::new();
    let mut space_ofs: Vec<(isize, isize)> = Vec::new();
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let r = ((dy * dy + dx * dx) as f64).sqrt();
            if r > radius as f64 {
                continue; // Circular mask — skip corners
            }
            let r2 = (dy * dy + dx * dx) as f32;
            space_weight.push((r2 * gauss_space_coeff).exp());
            space_ofs.push((dy, dx));
        }
    }
    let maxk = space_weight.len();

    // Pre-compute color weight LUT (indexed by |diff|, 0..255*channels)
    let gauss_color_coeff: f32 = -0.5 / (sigma_color * sigma_color);
    let color_lut_size = 256 * channels;
    let mut color_weight = vec![0.0f32; color_lut_size];
    for (i, cw) in color_weight.iter_mut().enumerate().take(color_lut_size) {
        let fi = i as f32;
        *cw = (fi * fi * gauss_color_coeff).exp();
    }

    // Pad image with BORDER_REFLECT_101
    let pw = w + 2 * radius;
    let ph = h + 2 * radius;
    let mut padded = vec![0u8; pw * ph * channels];
    for py in 0..ph {
        let sy = reflect101(py as isize - radius as isize, h as isize) as usize;
        for px in 0..pw {
            let sx = reflect101(px as isize - radius as isize, w as isize) as usize;
            for c in 0..channels {
                padded[(py * pw + px) * channels + c] = pixels[(sy * w + sx) * channels + c];
            }
        }
    }

    let mut result = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            let py = y + radius;
            let px = x + radius;

            if channels == 1 {
                let val0 = padded[py * pw + px] as i32;
                let mut wsum: f32 = 0.0;
                let mut vsum: f32 = 0.0;
                for k in 0..maxk {
                    let (dy, dx) = space_ofs[k];
                    let n_off = (py as isize + dy) as usize * pw + (px as isize + dx) as usize;
                    let val = padded[n_off] as i32;
                    let w = space_weight[k] * color_weight[(val - val0).unsigned_abs() as usize];
                    wsum += w;
                    vsum += val as f32 * w;
                }
                result[y * w + x] = (vsum / wsum).round().clamp(0.0, 255.0) as u8;
            } else {
                let center_off = (py * pw + px) * channels;
                let b0 = padded[center_off] as i32;
                let g0 = padded[center_off + 1] as i32;
                let r0 = padded[center_off + 2] as i32;
                let mut wsum: f32 = 0.0;
                let mut bsum: f32 = 0.0;
                let mut gsum: f32 = 0.0;
                let mut rsum: f32 = 0.0;
                for k in 0..maxk {
                    let (dy, dx) = space_ofs[k];
                    let n_off =
                        ((py as isize + dy) as usize * pw + (px as isize + dx) as usize) * channels;
                    let b = padded[n_off] as i32;
                    let g = padded[n_off + 1] as i32;
                    let r = padded[n_off + 2] as i32;
                    let color_diff =
                        (b - b0).unsigned_abs() + (g - g0).unsigned_abs() + (r - r0).unsigned_abs();
                    let w = space_weight[k]
                        * color_weight[(color_diff as usize).min(color_lut_size - 1)];
                    wsum += w;
                    bsum += b as f32 * w;
                    gsum += g as f32 * w;
                    rsum += r as f32 * w;
                }
                let out_off = (y * w + x) * channels;
                result[out_off] = (bsum / wsum).round().clamp(0.0, 255.0) as u8;
                result[out_off + 1] = (gsum / wsum).round().clamp(0.0, 255.0) as u8;
                result[out_off + 2] = (rsum / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(crop_to_request(&result, expanded, request, info.format))
}
