//! Template matching — normalized cross-correlation (NCC).
//!
//! Slides a template across the image and computes NCC at each position.
//! Returns the location of the best match.
//!
//! NCC = sum((I - mean_I) * (T - mean_T)) / sqrt(sum((I - mean_I)^2) * sum((T - mean_T)^2))
//!
//! Reference: OpenCV matchTemplate TM_CCOEFF_NORMED

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Result of template matching — best match location and score.
#[derive(Debug, Clone, Copy)]
pub struct MatchResult {
    pub x: u32,
    pub y: u32,
    pub score: f32,
}

/// Find the best match of a template within an image using normalized cross-correlation.
///
/// Both `image` and `template` must be Gray8 pixel buffers.
/// Returns the top-left position and NCC score (0..1) of the best match.
pub fn template_match(
    image: &[u8],
    image_w: u32,
    image_h: u32,
    template: &[u8],
    template_w: u32,
    template_h: u32,
) -> Result<MatchResult, ImageError> {
    let iw = image_w as usize;
    let ih = image_h as usize;
    let tw = template_w as usize;
    let th = template_h as usize;

    if tw > iw || th > ih {
        return Err(ImageError::InvalidParameters(
            "Template larger than image".to_string(),
        ));
    }
    if image.len() != iw * ih || template.len() != tw * th {
        return Err(ImageError::InvalidParameters(
            "Buffer size mismatch".to_string(),
        ));
    }

    // Pre-compute template mean and norm
    let t_sum: f64 = template.iter().map(|&v| v as f64).sum();
    let t_mean = t_sum / (tw * th) as f64;
    let t_norm_sq: f64 = template
        .iter()
        .map(|&v| {
            let d = v as f64 - t_mean;
            d * d
        })
        .sum();
    let t_norm = t_norm_sq.sqrt();

    if t_norm < 1e-10 {
        // Template is constant — any position is equally good
        return Ok(MatchResult { x: 0, y: 0, score: 0.0 });
    }

    let result_h = ih - th + 1;
    let result_w = iw - tw + 1;

    let mut best = MatchResult { x: 0, y: 0, score: f32::NEG_INFINITY };

    for ry in 0..result_h {
        for rx in 0..result_w {
            // Compute NCC at position (rx, ry)
            let mut i_sum = 0.0f64;
            for ty in 0..th {
                for tx in 0..tw {
                    i_sum += image[(ry + ty) * iw + (rx + tx)] as f64;
                }
            }
            let i_mean = i_sum / (tw * th) as f64;

            let mut numer = 0.0f64;
            let mut i_norm_sq = 0.0f64;
            for ty in 0..th {
                for tx in 0..tw {
                    let iv = image[(ry + ty) * iw + (rx + tx)] as f64 - i_mean;
                    let tv = template[ty * tw + tx] as f64 - t_mean;
                    numer += iv * tv;
                    i_norm_sq += iv * iv;
                }
            }

            let denom = (i_norm_sq.sqrt()) * t_norm;
            let score = if denom > 1e-10 { (numer / denom) as f32 } else { 0.0 };

            if score > best.score {
                best = MatchResult {
                    x: rx as u32,
                    y: ry as u32,
                    score,
                };
            }
        }
    }

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exact_sub_image() {
        // Create a 32x32 image with a distinct 8x8 pattern at (10, 12)
        let mut image = vec![128u8; 32 * 32];
        let mut template = vec![0u8; 8 * 8];

        // Place a checkerboard pattern
        for y in 0..8u32 {
            for x in 0..8u32 {
                let val = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                template[(y * 8 + x) as usize] = val;
                image[((12 + y) * 32 + (10 + x)) as usize] = val;
            }
        }

        let result = template_match(&image, 32, 32, &template, 8, 8).unwrap();
        assert_eq!(result.x, 10, "Expected x=10, got {}", result.x);
        assert_eq!(result.y, 12, "Expected y=12, got {}", result.y);
        assert!(result.score > 0.99, "Expected score ~1.0, got {}", result.score);
        eprintln!("  template_match exact: pos=({},{}) score={:.4}", result.x, result.y, result.score);
    }

    #[test]
    fn constant_template_returns_zero_score() {
        let image = vec![128u8; 16 * 16];
        let template = vec![128u8; 4 * 4];
        let result = template_match(&image, 16, 16, &template, 4, 4).unwrap();
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn template_larger_than_image_errors() {
        let image = vec![0u8; 8 * 8];
        let template = vec![0u8; 16 * 16];
        assert!(template_match(&image, 8, 8, &template, 16, 16).is_err());
    }
}
