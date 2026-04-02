//! Filter: replace_color (category: color)
//!
//! Select pixels by HSL range (center hue, hue range, saturation range,
//! lightness range) and shift their hue/saturation/lightness with smooth
//! falloff at boundaries.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::color_grading::{rgb_to_hsl, hsl_to_rgb};

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Replace color — select by HSL range and shift H/S/L.
pub struct ReplaceColorParams {
    /// Center hue to target (degrees 0-360)
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0, hint = "rc.angle_deg")]
    pub center_hue: f32,
    /// Hue range width (degrees, total spread around center)
    #[param(min = 1.0, max = 180.0, step = 1.0, default = 30.0)]
    pub hue_range: f32,
    /// Minimum saturation for selection (0-1)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub sat_min: f32,
    /// Maximum saturation for selection (0-1)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub sat_max: f32,
    /// Minimum lightness for selection (0-1)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub lum_min: f32,
    /// Maximum lightness for selection (0-1)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub lum_max: f32,
    /// Hue shift in degrees
    #[param(min = -180.0, max = 180.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub hue_shift: f32,
    /// Saturation shift (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub sat_shift: f32,
    /// Lightness shift (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub lum_shift: f32,
}

#[rasmcore_macros::register_filter(
    name = "replace_color",
    category = "color",
    reference = "HSL range selection with hue/saturation/lightness shift"
)]
pub fn replace_color(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ReplaceColorParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();

    let (ch, has_alpha) = match info.format {
        PixelFormat::Rgb8 => (3usize, false),
        PixelFormat::Rgba8 => (4usize, true),
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "replace_color requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let n = (info.width as usize) * (info.height as usize);
    let half_hue = config.hue_range / 2.0;
    let mut result = pixels.to_vec();

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f32 / 255.0;
        let g = pixels[pi + 1] as f32 / 255.0;
        let b = pixels[pi + 2] as f32 / 255.0;

        let (h, s, l) = rgb_to_hsl(r, g, b);

        // Hue distance (wrapping around 360)
        let hue_diff = ((h - config.center_hue + 180.0).rem_euclid(360.0)) - 180.0;
        if hue_diff.abs() > half_hue {
            continue;
        }

        // Saturation range check
        if s < config.sat_min || s > config.sat_max {
            continue;
        }

        // Lightness range check
        if l < config.lum_min || l > config.lum_max {
            continue;
        }

        // Compute smooth weight from hue distance (cosine taper at range edges)
        let weight = if half_hue > 0.0 {
            let t = hue_diff.abs() / half_hue;
            0.5 * (1.0 + (t * std::f32::consts::PI).cos())
        } else {
            1.0
        };

        let new_h = (h + config.hue_shift * weight).rem_euclid(360.0);
        let new_s = (s + config.sat_shift * weight).clamp(0.0, 1.0);
        let new_l = (l + config.lum_shift * weight).clamp(0.0, 1.0);

        let (nr, ng, nb) = hsl_to_rgb(new_h, new_s, new_l);
        result[pi] = (nr * 255.0).round().clamp(0.0, 255.0) as u8;
        result[pi + 1] = (ng * 255.0).round().clamp(0.0, 255.0) as u8;
        result[pi + 2] = (nb * 255.0).round().clamp(0.0, 255.0) as u8;
        // Alpha preserved (already copied from pixels)
    }

    Ok(result)
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

    fn make_request(info: &ImageInfo) -> Rect {
        Rect::new(0, 0, info.width, info.height)
    }

    #[test]
    fn replace_color_zero_shift_is_identity() {
        let pixels = vec![200, 50, 50, 50, 50, 200, 100, 200, 100];
        let info = info_rgb8(3, 1);
        let config = ReplaceColorParams {
            center_hue: 0.0,
            hue_range: 180.0,
            sat_min: 0.0,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 0.0,
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let result = replace_color(
            make_request(&info),
            &mut |_| Ok(pixels.clone()),
            &info,
            &config,
        )
        .unwrap();
        // Zero shifts = near identity (within HSL roundtrip tolerance)
        for (i, (&orig, &res)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i32 - res as i32).unsigned_abs() <= 1,
                "pixel byte {i}: orig={orig}, result={res}"
            );
        }
    }

    #[test]
    fn replace_color_targets_red_only() {
        // Red pixel (hue ~0), green pixel (hue ~120), blue pixel (hue ~240)
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let info = info_rgb8(3, 1);
        let config = ReplaceColorParams {
            center_hue: 0.0,
            hue_range: 60.0, // only target reds
            sat_min: 0.0,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 120.0, // shift red toward green
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let result = replace_color(
            make_request(&info),
            &mut |_| Ok(pixels.clone()),
            &info,
            &config,
        )
        .unwrap();

        // Red pixel should have shifted (more green, less red)
        assert!(
            result[1] > result[0],
            "red pixel should shift green: R={}, G={}",
            result[0],
            result[1]
        );
        // Green pixel should be unchanged
        assert_eq!(result[3], 0);
        assert_eq!(result[4], 255);
        assert_eq!(result[5], 0);
        // Blue pixel should be unchanged
        assert_eq!(result[6], 0);
        assert_eq!(result[7], 0);
        assert_eq!(result[8], 255);
    }

    #[test]
    fn replace_color_sat_range_excludes_gray() {
        // Gray pixel (s=0), saturated red (s=1)
        let pixels = vec![128, 128, 128, 255, 0, 0];
        let info = info_rgb8(2, 1);
        let config = ReplaceColorParams {
            center_hue: 0.0,
            hue_range: 180.0,
            sat_min: 0.5,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 120.0,
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let result = replace_color(
            make_request(&info),
            &mut |_| Ok(pixels.clone()),
            &info,
            &config,
        )
        .unwrap();
        // Gray pixel should be unchanged (sat=0, below sat_min=0.5)
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 128);
        assert_eq!(result[2], 128);
        // Red pixel should have shifted
        assert!(
            result[4] > result[3],
            "red should shift green: R={}, G={}",
            result[3],
            result[4]
        );
    }

    #[test]
    fn replace_color_lightness_shift() {
        // Pure red pixel
        let pixels = vec![255, 0, 0];
        let info = info_rgb8(1, 1);
        let config = ReplaceColorParams {
            center_hue: 0.0,
            hue_range: 60.0,
            sat_min: 0.0,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 0.0,
            sat_shift: 0.0,
            lum_shift: -0.2, // darken
        };
        let result = replace_color(
            make_request(&info),
            &mut |_| Ok(pixels.clone()),
            &info,
            &config,
        )
        .unwrap();
        // Should be darker red
        assert!(
            result[0] < 255,
            "red should darken: got {}",
            result[0]
        );
    }

    #[test]
    fn replace_color_rgba_preserves_alpha() {
        let pixels = vec![255, 0, 0, 128]; // red with 50% alpha
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let config = ReplaceColorParams {
            center_hue: 0.0,
            hue_range: 60.0,
            sat_min: 0.0,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 120.0,
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let result = replace_color(
            Rect::new(0, 0, 1, 1),
            &mut |_| Ok(pixels.clone()),
            &info,
            &config,
        )
        .unwrap();
        assert_eq!(result[3], 128, "alpha should be preserved");
    }
}
