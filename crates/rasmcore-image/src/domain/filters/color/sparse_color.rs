//! Filter: sparse_color (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Generate an image by interpolating colors from sparse control points
/// using Shepard's inverse-distance-weighted method.
///
/// Each pixel color is a weighted average of all control points, where
/// weight = 1 / distance^power. IM equivalent: -sparse-color Shepard "..."
/// Sparse color parameters (control points as string).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SparseColorParams {
    /// Control points as "x,y:RRGGBB" entries separated by semicolons.
    #[param(default = "")]
    pub points: String,
    /// Inverse distance power (default 2.0). Higher = sharper falloff.
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 2.0)]
    pub power: f32,
}

#[rasmcore_macros::register_filter(
    name = "sparse_color",
    category = "color",
    reference = "Shepard interpolation from sparse points"
)]
pub fn sparse_color(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    points: String,
    config: &SparseColorParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let power = config.power;

    validate_format(info.format)?;
    let ctrl = parse_sparse_points(&points)?;
    let power = if power <= 0.0 { 2.0 } else { power };

    let bpp = match info.format {
        PixelFormat::Rgba8 => 4,
        PixelFormat::Rgb8 => 3,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "sparse_color requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mut result = pixels.to_vec();
    for y in 0..info.height {
        for x in 0..info.width {
            let px = x as f32;
            let py = y as f32;
            let idx = (y * info.width + x) as usize * bpp;

            let mut sum_r = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut sum_b = 0.0f64;
            let mut sum_w = 0.0f64;
            let mut exact_match = None;

            for &(cx, cy, color) in &ctrl {
                let dx = (px - cx) as f64;
                let dy = (py - cy) as f64;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 0.001 {
                    exact_match = Some(color);
                    break;
                }
                let w = 1.0 / dist_sq.powf(power as f64 / 2.0);
                sum_r += w * color[0] as f64;
                sum_g += w * color[1] as f64;
                sum_b += w * color[2] as f64;
                sum_w += w;
            }

            if let Some(color) = exact_match {
                result[idx] = color[0];
                result[idx + 1] = color[1];
                result[idx + 2] = color[2];
            } else if sum_w > 0.0 {
                result[idx] = (sum_r / sum_w).round().clamp(0.0, 255.0) as u8;
                result[idx + 1] = (sum_g / sum_w).round().clamp(0.0, 255.0) as u8;
                result[idx + 2] = (sum_b / sum_w).round().clamp(0.0, 255.0) as u8;
            }
            // Alpha preserved (if RGBA)
        }
    }
    Ok(result)
}
