use super::super::error::ImageError;
use super::super::types::{ImageInfo, PixelFormat};
use super::icc::icc_to_srgb;

// ─── CMYK <-> RGB Conversion ──────────────────────────────────────────────────

/// Convert a CMYK pixel buffer to RGB8 using ICC profile if available,
/// falling back to the naive formula.
pub fn cmyk_to_rgb_icc(
    cmyk_pixels: &[u8],
    info: &ImageInfo,
    icc_profile: Option<&[u8]>,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if let Some(profile) = icc_profile {
        let (icc_pixels, icc_info) = if info.format == PixelFormat::Cmyk8 {
            let padded: Vec<u8> = cmyk_pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2], c[3], 255])
                .collect();
            let padded_info = ImageInfo {
                format: PixelFormat::Cmyka8,
                ..*info
            };
            (padded, padded_info)
        } else {
            (cmyk_pixels.to_vec(), info.clone())
        };

        let rgba = icc_to_srgb(&icc_pixels, &icc_info, profile)?;
        let out_info = ImageInfo {
            format: PixelFormat::Rgba8,
            ..*info
        };
        return Ok((rgba, out_info));
    }
    cmyk_to_rgb(cmyk_pixels, info)
}

/// Convert a CMYK pixel buffer to RGB8 using the naive formula.
pub fn cmyk_to_rgb(
    cmyk_pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let n = (info.width as usize) * (info.height as usize);
    let (src_bpp, has_alpha) = match info.format {
        PixelFormat::Cmyk8 => (4, false),
        PixelFormat::Cmyka8 => (5, true),
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "cmyk_to_rgb requires Cmyk8 or Cmyka8, got {other:?}"
            )));
        }
    };

    if cmyk_pixels.len() != n * src_bpp {
        return Err(ImageError::InvalidInput(
            "pixel buffer size mismatch".into(),
        ));
    }

    let dst_bpp = if has_alpha { 4 } else { 3 };
    let mut rgb = vec![0u8; n * dst_bpp];

    for i in 0..n {
        let si = i * src_bpp;
        let di = i * dst_bpp;
        let c = cmyk_pixels[si] as f32 / 255.0;
        let m = cmyk_pixels[si + 1] as f32 / 255.0;
        let y = cmyk_pixels[si + 2] as f32 / 255.0;
        let k = cmyk_pixels[si + 3] as f32 / 255.0;

        rgb[di] = (255.0 * (1.0 - c) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
        rgb[di + 1] = (255.0 * (1.0 - m) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
        rgb[di + 2] = (255.0 * (1.0 - y) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
        if has_alpha {
            rgb[di + 3] = cmyk_pixels[si + 4];
        }
    }

    let out_format = if has_alpha {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    Ok((
        rgb,
        ImageInfo {
            format: out_format,
            ..*info
        },
    ))
}

/// Convert an RGB8 pixel buffer to CMYK.
pub fn rgb_to_cmyk(
    rgb_pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let n = (info.width as usize) * (info.height as usize);
    let (src_bpp, has_alpha) = match info.format {
        PixelFormat::Rgb8 => (3, false),
        PixelFormat::Rgba8 => (4, true),
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "rgb_to_cmyk requires Rgb8 or Rgba8, got {other:?}"
            )));
        }
    };

    if rgb_pixels.len() != n * src_bpp {
        return Err(ImageError::InvalidInput(
            "pixel buffer size mismatch".into(),
        ));
    }

    let dst_bpp = if has_alpha { 5 } else { 4 };
    let mut cmyk = vec![0u8; n * dst_bpp];

    for i in 0..n {
        let si = i * src_bpp;
        let di = i * dst_bpp;
        let r = rgb_pixels[si] as f32 / 255.0;
        let g = rgb_pixels[si + 1] as f32 / 255.0;
        let b = rgb_pixels[si + 2] as f32 / 255.0;

        let k = 1.0 - r.max(g).max(b);
        if k >= 1.0 {
            cmyk[di] = 0;
            cmyk[di + 1] = 0;
            cmyk[di + 2] = 0;
            cmyk[di + 3] = 255;
        } else {
            let inv_k = 1.0 / (1.0 - k);
            cmyk[di] = ((1.0 - r - k) * inv_k * 255.0).round().clamp(0.0, 255.0) as u8;
            cmyk[di + 1] = ((1.0 - g - k) * inv_k * 255.0).round().clamp(0.0, 255.0) as u8;
            cmyk[di + 2] = ((1.0 - b - k) * inv_k * 255.0).round().clamp(0.0, 255.0) as u8;
            cmyk[di + 3] = (k * 255.0).round() as u8;
        }
        if has_alpha {
            cmyk[di + 4] = rgb_pixels[si + 3];
        }
    }

    let out_format = if has_alpha {
        PixelFormat::Cmyka8
    } else {
        PixelFormat::Cmyk8
    };
    Ok((
        cmyk,
        ImageInfo {
            format: out_format,
            ..*info
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    #[test]
    fn cmyk_rgb_round_trip() {
        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];

        let (cmyk, cmyk_info) = rgb_to_cmyk(&rgb, &info).unwrap();
        assert_eq!(cmyk_info.format, PixelFormat::Cmyk8);
        assert_eq!(cmyk.len(), 16);

        let (rgb2, rgb2_info) = cmyk_to_rgb(&cmyk, &cmyk_info).unwrap();
        assert_eq!(rgb2_info.format, PixelFormat::Rgb8);
        assert_eq!(rgb2.len(), 12);

        for i in 0..12 {
            let diff = (rgb[i] as i32 - rgb2[i] as i32).abs();
            assert!(
                diff <= 1,
                "Round-trip error at byte {i}: {orig} -> {rt} (diff {diff})",
                orig = rgb[i],
                rt = rgb2[i]
            );
        }
    }

    #[test]
    fn cmyk_rgb_to_cmyk_parity_vs_imagemagick() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available for CMYK parity test");
            return;
        }

        let (w, h) = (32u32, 32u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize * 3;
                rgb[i] = (x * 255 / w) as u8;
                rgb[i + 1] = (y * 255 / h) as u8;
                rgb[i + 2] = 128;
            }
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let png_data = crate::domain::encoder::encode(&rgb, &info, "png", None).unwrap();
        let png_path = std::env::temp_dir().join("cmyk_parity_input.png");
        std::fs::write(&png_path, &png_data).unwrap();

        let (our_cmyk, _) = rgb_to_cmyk(&rgb, &info).unwrap();

        let im_raw = std::env::temp_dir().join("cmyk_parity_im.raw");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-colorspace",
                "CMYK",
                "-depth",
                "8",
                &format!("cmyk:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();
        assert!(result.status.success(), "magick RGB->CMYK failed");

        let im_cmyk = std::fs::read(&im_raw).unwrap();
        assert_eq!(our_cmyk.len(), im_cmyk.len());

        let n = our_cmyk.len();
        let mae: f64 = our_cmyk
            .iter()
            .zip(im_cmyk.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / n as f64;

        assert!(
            mae < 1.5,
            "CMYK rgb_to_cmyk vs ImageMagick MAE = {mae:.3} (must be < 1.5)"
        );
        eprintln!("CMYK rgb_to_cmyk vs ImageMagick: MAE = {mae:.3}");

        let cmyk_info = ImageInfo {
            format: PixelFormat::Cmyk8,
            ..info
        };
        let (our_rt, _) = cmyk_to_rgb(&our_cmyk, &cmyk_info).unwrap();

        let im_rt_raw = std::env::temp_dir().join("cmyk_parity_rt.raw");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-colorspace",
                "CMYK",
                "-colorspace",
                "sRGB",
                "-depth",
                "8",
                &format!("rgb:{}", im_rt_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();
        assert!(result.status.success(), "magick round-trip failed");

        let im_rt = std::fs::read(&im_rt_raw).unwrap();
        let rt_mae: f64 = our_rt
            .iter()
            .zip(im_rt.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / our_rt.len() as f64;

        assert!(
            rt_mae < 1.5,
            "CMYK round-trip vs ImageMagick MAE = {rt_mae:.3} (must be < 1.5)"
        );
        eprintln!("CMYK round-trip vs ImageMagick: MAE = {rt_mae:.3}");

        let _ = std::fs::remove_file(&png_path);
        let _ = std::fs::remove_file(&im_raw);
        let _ = std::fs::remove_file(&im_rt_raw);
    }

    #[test]
    fn cmyk_to_rgb_pure_black() {
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Cmyk8,
            color_space: ColorSpace::Srgb,
        };
        let cmyk = vec![0, 0, 0, 255];
        let (rgb, _) = cmyk_to_rgb(&cmyk, &info).unwrap();
        assert_eq!(rgb, vec![0, 0, 0]);
    }

    #[test]
    fn cmyk_to_rgb_pure_white() {
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Cmyk8,
            color_space: ColorSpace::Srgb,
        };
        let cmyk = vec![0, 0, 0, 0];
        let (rgb, _) = cmyk_to_rgb(&cmyk, &info).unwrap();
        assert_eq!(rgb, vec![255, 255, 255]);
    }
}
