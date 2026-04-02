//! Compositor: match_color (category: color)
//!
//! Reinhard 2001 color transfer — match color statistics from a reference
//! image onto a target image in CIE LAB space.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::color_spaces::{rgb_to_lab, lab_to_rgb};

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Match color — transfer color statistics from reference to target (Reinhard 2001).
pub struct MatchColorParams {
    /// Blend intensity (0 = no change, 1 = full transfer)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub intensity: f32,
}

#[rasmcore_macros::register_compositor(
    name = "match_color",
    category = "color",
    group = "color",
    variant = "match_color",
    reference = "Reinhard 2001 LAB statistics transfer"
)]
pub fn match_color(
    target_pixels: &[u8],
    target_info: &ImageInfo,
    ref_pixels: &[u8],
    ref_info: &ImageInfo,
    intensity: f32,
) -> Result<Vec<u8>, ImageError> {
    let target_ch = match target_info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "match_color requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let ref_ch = match ref_info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "match_color reference requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let target_n = (target_info.width as usize) * (target_info.height as usize);
    let ref_n = (ref_info.width as usize) * (ref_info.height as usize);
    let has_alpha = target_ch == 4;
    let intensity = intensity.clamp(0.0, 1.0) as f64;

    // Compute LAB statistics for target
    let (t_mean_l, t_std_l, t_mean_a, t_std_a, t_mean_b, t_std_b) =
        lab_stats(target_pixels, target_n, target_ch);

    // Compute LAB statistics for reference
    let (r_mean_l, r_std_l, r_mean_a, r_std_a, r_mean_b, r_std_b) =
        lab_stats(ref_pixels, ref_n, ref_ch);

    // Compute scale factors (guard against zero std)
    let scale_l = if t_std_l > 1e-10 { r_std_l / t_std_l } else { 1.0 };
    let scale_a = if t_std_a > 1e-10 { r_std_a / t_std_a } else { 1.0 };
    let scale_b = if t_std_b > 1e-10 { r_std_b / t_std_b } else { 1.0 };

    // Apply Reinhard transfer to each target pixel
    let mut result = vec![0u8; target_pixels.len()];
    for i in 0..target_n {
        let pi = i * target_ch;
        let r = target_pixels[pi] as f64 / 255.0;
        let g = target_pixels[pi + 1] as f64 / 255.0;
        let b = target_pixels[pi + 2] as f64 / 255.0;

        let (l, a, bv) = rgb_to_lab(r, g, b);

        // Reinhard transfer: shift and scale in LAB
        let new_l = (l - t_mean_l) * scale_l + r_mean_l;
        let new_a = (a - t_mean_a) * scale_a + r_mean_a;
        let new_b = (bv - t_mean_b) * scale_b + r_mean_b;

        // Blend with original by intensity
        let out_l = l + (new_l - l) * intensity;
        let out_a = a + (new_a - a) * intensity;
        let out_b = bv + (new_b - bv) * intensity;

        let (ro, go, bo) = lab_to_rgb(out_l, out_a, out_b);
        result[pi] = (ro.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 1] = (go.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 2] = (bo.clamp(0.0, 1.0) * 255.0).round() as u8;
        if has_alpha {
            result[pi + 3] = target_pixels[pi + 3];
        }
    }

    Ok(result)
}

/// Compute mean and standard deviation of L, a, b channels for an RGB8/RGBA8 image.
fn lab_stats(pixels: &[u8], n: usize, ch: usize) -> (f64, f64, f64, f64, f64, f64) {
    let mut sum_l = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_l2 = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b = pixels[pi + 2] as f64 / 255.0;
        let (l, a, bv) = rgb_to_lab(r, g, b);
        sum_l += l;
        sum_a += a;
        sum_b += bv;
        sum_l2 += l * l;
        sum_a2 += a * a;
        sum_b2 += bv * bv;
    }

    let nf = n as f64;
    let mean_l = sum_l / nf;
    let mean_a = sum_a / nf;
    let mean_b = sum_b / nf;
    // Population std (not sample std) — matches numpy default
    let std_l = (sum_l2 / nf - mean_l * mean_l).max(0.0).sqrt();
    let std_a = (sum_a2 / nf - mean_a * mean_a).max(0.0).sqrt();
    let std_b = (sum_b2 / nf - mean_b * mean_b).max(0.0).sqrt();

    (mean_l, std_l, mean_a, std_a, mean_b, std_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn info_rgb8(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        (0..(w * h)).flat_map(|_| [r, g, b]).collect()
    }

    #[test]
    fn match_color_identity_same_image() {
        // Matching an image to itself should be near-identity
        let pixels = vec![
            200, 100, 50, 100, 150, 200, 50, 200, 100, 180, 80, 140,
        ];
        let info = info_rgb8(2, 2);
        let result = match_color(&pixels, &info, &pixels, &info, 1.0).unwrap();
        for (i, (&orig, &res)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i32 - res as i32).unsigned_abs() <= 1,
                "pixel byte {i}: orig={orig}, result={res}"
            );
        }
    }

    #[test]
    fn match_color_zero_intensity_is_identity() {
        // intensity=0 means no transfer
        let target = vec![200, 100, 50, 100, 150, 200];
        let reference = solid_rgb(1, 2, 0, 0, 255);
        let info = info_rgb8(1, 2);
        let result = match_color(&target, &info, &reference, &info, 0.0).unwrap();
        for (i, (&orig, &res)) in target.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i32 - res as i32).unsigned_abs() <= 1,
                "pixel byte {i}: orig={orig}, result={res}"
            );
        }
    }

    #[test]
    fn match_color_transfers_tone() {
        // Transfer from a warm image to a cool image — result should shift warmer
        let cool = vec![50, 50, 200, 60, 60, 180]; // bluish
        let warm = vec![200, 100, 50, 180, 120, 60]; // warm/orange
        let cool_info = info_rgb8(1, 2);
        let warm_info = info_rgb8(1, 2);
        let result = match_color(&cool, &cool_info, &warm, &warm_info, 1.0).unwrap();
        // Result should have more red and less blue than original cool image
        let avg_r_orig = (cool[0] as u32 + cool[3] as u32) / 2;
        let avg_b_orig = (cool[2] as u32 + cool[5] as u32) / 2;
        let avg_r_result = (result[0] as u32 + result[3] as u32) / 2;
        let avg_b_result = (result[2] as u32 + result[5] as u32) / 2;
        assert!(
            avg_r_result > avg_r_orig,
            "expected red to increase: orig={avg_r_orig}, result={avg_r_result}"
        );
        assert!(
            avg_b_result < avg_b_orig,
            "expected blue to decrease: orig={avg_b_orig}, result={avg_b_result}"
        );
    }

    #[test]
    fn match_color_solid_to_solid() {
        // Matching a solid color target to a solid color reference with full intensity
        // should produce the reference color (since all std=0, transfer uses scale=1)
        let target = solid_rgb(4, 4, 100, 100, 100);
        let reference = solid_rgb(4, 4, 200, 50, 50);
        let info = info_rgb8(4, 4);
        let result = match_color(&target, &info, &reference, &info, 1.0).unwrap();
        // With solid images, std=0 for both, so scale=1.
        // Result: (L - mean_src) * 1 + mean_ref = mean_ref (since all L are same = mean_src)
        // So result should be the reference color
        for i in (0..result.len()).step_by(3) {
            assert!(
                (result[i] as i32 - 200).unsigned_abs() <= 2,
                "R: expected ~200, got {}",
                result[i]
            );
            assert!(
                (result[i + 1] as i32 - 50).unsigned_abs() <= 2,
                "G: expected ~50, got {}",
                result[i + 1]
            );
            assert!(
                (result[i + 2] as i32 - 50).unsigned_abs() <= 2,
                "B: expected ~50, got {}",
                result[i + 2]
            );
        }
    }

    #[test]
    fn match_color_different_sizes() {
        // Target and reference can be different sizes
        let target = solid_rgb(2, 2, 100, 150, 200);
        let reference = solid_rgb(8, 8, 200, 100, 50);
        let t_info = info_rgb8(2, 2);
        let r_info = info_rgb8(8, 8);
        let result = match_color(&target, &t_info, &reference, &r_info, 1.0).unwrap();
        assert_eq!(result.len(), target.len());
    }

    #[test]
    fn match_color_rgba_preserves_alpha() {
        let target = vec![200, 100, 50, 128, 100, 150, 200, 64];
        let reference = vec![50, 50, 200, 255, 60, 60, 180, 255];
        let info = ImageInfo {
            width: 1,
            height: 2,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = match_color(&target, &info, &reference, &info, 1.0).unwrap();
        // Alpha channels should be preserved
        assert_eq!(result[3], 128);
        assert_eq!(result[7], 64);
    }
}
