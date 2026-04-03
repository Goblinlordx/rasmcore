//! LAB channel operation filters — extract L/a/b, sharpen L, adjust a/b.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;
use crate::domain::color_spaces::{rgb_to_lab, lab_to_rgb};

// ── Extract L channel ────────────────────────────────────────────────────

/// Extract CIE LAB L (lightness) channel as grayscale.
///
/// L range [0, 100] is mapped to [0, 255].
#[rasmcore_macros::register_mapper(
    name = "lab_extract_l",
    category = "color",
    reference = "CIE LAB L channel extraction",
    output_format = "Gray8"
)]
pub fn lab_extract_l(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (ch, has_alpha) = match info.format {
        PixelFormat::Rgb8 => (3usize, false),
        PixelFormat::Rgba8 => (4usize, true),
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "lab_extract_l requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b = pixels[pi + 2] as f64 / 255.0;
        let (l, _a, _b) = rgb_to_lab(r, g, b);
        // L is [0, 100] -> map to [0, 255]
        out.push((l / 100.0 * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    let _ = has_alpha; // alpha is discarded for grayscale output
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((out, out_info))
}

// ── Extract a channel ────────────────────────────────────────────────────

/// Extract CIE LAB a (green-red) channel as grayscale.
///
/// a range [-128, 127] is mapped to [0, 255] (128 = neutral).
#[rasmcore_macros::register_mapper(
    name = "lab_extract_a",
    category = "color",
    reference = "CIE LAB a channel extraction",
    output_format = "Gray8"
)]
pub fn lab_extract_a(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "lab_extract_a requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b = pixels[pi + 2] as f64 / 255.0;
        let (_l, a, _b) = rgb_to_lab(r, g, b);
        // a is roughly [-128, 127] -> map to [0, 255] with 128 as neutral
        out.push((a + 128.0).round().clamp(0.0, 255.0) as u8);
    }
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((out, out_info))
}

// ── Extract b channel ────────────────────────────────────────────────────

/// Extract CIE LAB b (blue-yellow) channel as grayscale.
///
/// b range [-128, 127] is mapped to [0, 255] (128 = neutral).
#[rasmcore_macros::register_mapper(
    name = "lab_extract_b",
    category = "color",
    reference = "CIE LAB b channel extraction",
    output_format = "Gray8"
)]
pub fn lab_extract_b(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3usize,
        PixelFormat::Rgba8 => 4usize,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "lab_extract_b requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b_val = pixels[pi + 2] as f64 / 255.0;
        let (_l, _a, bv) = rgb_to_lab(r, g, b_val);
        // b is roughly [-128, 127] -> map to [0, 255] with 128 as neutral
        out.push((bv + 128.0).round().clamp(0.0, 255.0) as u8);
    }
    let out_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    Ok((out, out_info))
}

// ── LAB Sharpen (luminosity only) ────────────────────────────────────────

/// Parameters for lab_sharpen.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "lab_sharpen", category = "color", reference = "luminosity-only unsharp mask in CIE LAB")]
pub struct LabSharpenParams {
    /// Sharpening strength (0 = none, 1 = standard, higher = aggressive)
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 1.0)]
    pub amount: f32,
    /// Blur radius for unsharp mask (controls detail scale)
    #[param(min = 0.1, max = 20.0, step = 0.1, default = 1.0)]
    pub radius: f32,
}

/// Unsharp-mask sharpening applied only to LAB L channel, preserving color.
///
/// Converts to LAB, applies unsharp mask to L only (L + amount * (L - blurred_L)),
/// then converts back. This avoids color fringing artifacts from RGB-domain sharpening.


// ── LAB Adjust (shift a/b channels) ─────────────────────────────────────

/// Parameters for lab_adjust.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "lab_adjust", category = "color", reference = "CIE LAB a/b channel offset", color_op = "true")]
pub struct LabAdjustParams {
    /// Offset for a channel (green-red axis). Positive shifts toward red.
    #[param(min = -128.0, max = 127.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub a_offset: f32,
    /// Offset for b channel (blue-yellow axis). Positive shifts toward yellow.
    #[param(min = -128.0, max = 127.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub b_offset: f32,
}

impl LabAdjustParams {
    /// Bake the LAB adjust into a 3D CLUT for GPU/fusion support.
    pub fn build_clut(&self) -> crate::domain::color_lut::ColorLut3D {
        let a_off = self.a_offset as f64;
        let b_off = self.b_offset as f64;
        crate::domain::color_lut::ColorLut3D::from_fn(33, |r, g, b| {
            let (l, a, bv) = rgb_to_lab(r as f64, g as f64, b as f64);
            let new_a = (a + a_off).clamp(-128.0, 127.0);
            let new_b = (bv + b_off).clamp(-128.0, 127.0);
            let (or, og, ob) = lab_to_rgb(l, new_a, new_b);
            (or as f32, og as f32, ob as f32)
        })
    }
}

impl CpuFilter for LabAdjustParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;
        let info = &ImageInfo {
            width: request.width,
            height: request.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let a_off = self.a_offset as f64;
        let b_off = self.b_offset as f64;

    let (ch, has_alpha) = match info.format {
        PixelFormat::Rgb8 => (3usize, false),
        PixelFormat::Rgba8 => (4usize, true),
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "lab_adjust requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let mut result = vec![0u8; pixels.len()];

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b = pixels[pi + 2] as f64 / 255.0;
        let (l, a, bv) = rgb_to_lab(r, g, b);
        let (ro, go, bo) = lab_to_rgb(l, a + a_off, bv + b_off);
        result[pi] = (ro.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 1] = (go.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 2] = (bo.clamp(0.0, 1.0) * 255.0).round() as u8;
        if has_alpha {
            result[pi + 3] = pixels[pi + 3];
        }
    }

        Ok(result)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

impl CpuFilter for LabSharpenParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = self.amount;
    let radius = self.radius;

    let (ch, has_alpha) = match info.format {
        PixelFormat::Rgb8 => (3usize, false),
        PixelFormat::Rgba8 => (4usize, true),
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "lab_sharpen requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);

    // Convert to LAB
    let mut l_channel = vec![0.0f64; n];
    let mut a_channel = vec![0.0f64; n];
    let mut b_channel = vec![0.0f64; n];

    for i in 0..n {
        let pi = i * ch;
        let r = pixels[pi] as f64 / 255.0;
        let g = pixels[pi + 1] as f64 / 255.0;
        let b = pixels[pi + 2] as f64 / 255.0;
        let (l, a, bv) = rgb_to_lab(r, g, b);
        l_channel[i] = l;
        a_channel[i] = a;
        b_channel[i] = bv;
    }

    // Create a grayscale image from L channel for blurring
    // L is [0, 100] -> scale to [0, 255] for blur_impl
    let l_as_u8: Vec<u8> = l_channel
        .iter()
        .map(|l| (l / 100.0 * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    let blurred_l_u8 = blur_impl(&l_as_u8, &gray_info, &BlurParams { radius })?;

    // Apply unsharp mask in LAB L space
    for i in 0..n {
        let l_orig = l_channel[i];
        let l_blur = blurred_l_u8[i] as f64 / 255.0 * 100.0;
        let detail = l_orig - l_blur;
        l_channel[i] = (l_orig + (amount as f64) * detail).clamp(0.0, 100.0);
    }

    // Convert back to RGB
    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let (r, g, b) = lab_to_rgb(l_channel[i], a_channel[i], b_channel[i]);
        let pi = i * ch;
        result[pi] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
        result[pi + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
        if has_alpha {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}
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
    fn lab_extract_l_white_is_max() {
        let pixels = solid_rgb(2, 2, 255, 255, 255);
        let info = info_rgb8(2, 2);
        let (out, out_info) = lab_extract_l(&pixels, &info).unwrap();
        assert_eq!(out_info.format, PixelFormat::Gray8);
        // White has L=100, mapped to 255
        for &v in &out {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn lab_extract_l_black_is_min() {
        let pixels = solid_rgb(2, 2, 0, 0, 0);
        let info = info_rgb8(2, 2);
        let (out, _) = lab_extract_l(&pixels, &info).unwrap();
        for &v in &out {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn lab_extract_l_gray_is_uniform() {
        let pixels = solid_rgb(4, 4, 128, 128, 128);
        let info = info_rgb8(4, 4);
        let (out, _) = lab_extract_l(&pixels, &info).unwrap();
        // All pixels should be the same value
        let first = out[0];
        for &v in &out {
            assert_eq!(v, first);
        }
        // Gray 128 has L ~53.6, so mapped value should be around 137
        assert!(first > 100 && first < 180, "expected ~137, got {first}");
    }

    #[test]
    fn lab_extract_a_neutral_gray() {
        let pixels = solid_rgb(2, 2, 128, 128, 128);
        let info = info_rgb8(2, 2);
        let (out, _) = lab_extract_a(&pixels, &info).unwrap();
        // Neutral gray has a=0, mapped to 128
        for &v in &out {
            assert!((v as i32 - 128).unsigned_abs() <= 1, "expected ~128, got {v}");
        }
    }

    #[test]
    fn lab_extract_b_neutral_gray() {
        let pixels = solid_rgb(2, 2, 128, 128, 128);
        let info = info_rgb8(2, 2);
        let (out, _) = lab_extract_b(&pixels, &info).unwrap();
        // Neutral gray has b=0, mapped to 128
        for &v in &out {
            assert!((v as i32 - 128).unsigned_abs() <= 1, "expected ~128, got {v}");
        }
    }

    #[test]
    fn lab_adjust_zero_is_identity() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let result = LabAdjustParams {
                a_offset: 0.0,
                b_offset: 0.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.clone()),
            &info,
        )
        .unwrap();
        // Roundtrip through LAB may lose 1 level due to floating-point
        for (i, (&orig, &res)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i32 - res as i32).unsigned_abs() <= 1,
                "pixel {i}: orig={orig}, result={res}"
            );
        }
    }

    #[test]
    fn lab_sharpen_preserves_color_on_solid() {
        // A solid color image has no detail, so sharpen should be near-identity
        let pixels = solid_rgb(8, 8, 100, 150, 200);
        let info = info_rgb8(8, 8);
        let result = LabSharpenParams {
                amount: 2.0,
                radius: 1.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.clone()),
            &info,
        )
        .unwrap();
        // Solid image: no edges to sharpen, should be near-identity
        for (i, (&orig, &res)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i32 - res as i32).unsigned_abs() <= 2,
                "pixel {i}: orig={orig}, result={res}"
            );
        }
    }

    #[test]
    fn lab_adjust_clut_matches_per_pixel() {
        // Validate that the 3D CLUT output matches the per-pixel CPU output
        // This ensures GPU path (via CLUT) matches CPU path
        let config = LabAdjustParams { a_offset: 30.0, b_offset: -20.0 };
        let clut = config.build_clut();
        let info = info_rgb8(8, 8);

        // Create a gradient test image with varied colors
        let mut pixels = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8u32 {
            for x in 0..8u32 {
                pixels.push((x * 32).min(255) as u8);
                pixels.push((y * 32).min(255) as u8);
                pixels.push(128u8);
            }
        }

        // Per-pixel CPU path
        let cpu_result = config.compute(
            Rect::new(0, 0, 8, 8),
            &mut |_| Ok(pixels.clone()),
            &info,
        ).unwrap();

        // CLUT path
        let clut_result = clut.apply(&pixels, &info).unwrap();

        // Compare — CLUT uses trilinear interpolation so allow ±2 per channel
        let mut max_diff = 0i32;
        for (i, (&cpu, &gpu)) in cpu_result.iter().zip(clut_result.iter()).enumerate() {
            let diff = (cpu as i32 - gpu as i32).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff <= 2,
                "pixel byte {i}: cpu={cpu}, clut={gpu}, diff={diff}"
            );
        }
        eprintln!("lab_adjust CLUT vs CPU max diff: {max_diff}/255");
    }
}
