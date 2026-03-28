use image::DynamicImage;

use super::error::ImageError;
use super::metadata::ExifOrientation;
use super::types::{
    ColorSpace, DecodedImage, FlipDirection, ImageInfo, PixelFormat, ResizeFilter, Rotation,
};

/// Resize an image to new dimensions using SIMD-optimized fast_image_resize.
///
/// Uses fast_image_resize with native SIMD (SSE4.1/AVX2 on x86, NEON on ARM,
/// SIMD128 on WASM). 15-57x faster than the image crate on large images.
pub fn resize(
    pixels: &[u8],
    info: &ImageInfo,
    width: u32,
    height: u32,
    filter: ResizeFilter,
) -> Result<DecodedImage, ImageError> {
    use fast_image_resize as fir;

    if info.width == 0 || info.height == 0 || width == 0 || height == 0 {
        return Err(ImageError::InvalidParameters(
            "resize dimensions must be > 0".into(),
        ));
    }

    let pixel_type = match info.format {
        PixelFormat::Rgb8 => fir::PixelType::U8x3,
        PixelFormat::Rgba8 => fir::PixelType::U8x4,
        PixelFormat::Gray8 => fir::PixelType::U8,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "resize from {other:?} not supported by SIMD backend"
            )));
        }
    };

    let fir_filter = match filter {
        ResizeFilter::Nearest => fir::ResizeAlg::Nearest,
        ResizeFilter::Bilinear => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        ResizeFilter::Bicubic => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        ResizeFilter::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    };

    let src_image =
        fir::images::Image::from_vec_u8(info.width, info.height, pixels.to_vec(), pixel_type)
            .map_err(|e| ImageError::InvalidInput(format!("pixel data mismatch: {e}")))?;

    let mut dst_image = fir::images::Image::new(width, height, pixel_type);

    let options = fir::ResizeOptions::new().resize_alg(fir_filter);
    let mut resizer = fir::Resizer::new();
    #[cfg(target_arch = "wasm32")]
    unsafe {
        resizer.set_cpu_extensions(fir::CpuExtensions::Simd128);
    }
    resizer
        .resize(&src_image, &mut dst_image, &options)
        .map_err(|e| ImageError::ProcessingFailed(format!("resize failed: {e}")))?;

    let result_pixels = dst_image.into_vec();

    Ok(DecodedImage {
        pixels: result_pixels,
        info: ImageInfo {
            width,
            height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Crop a region from an image
pub fn crop(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, ImageError> {
    if x + width > info.width || y + height > info.height {
        return Err(ImageError::InvalidParameters(format!(
            "crop region ({x},{y},{width},{height}) exceeds image bounds ({},{})",
            info.width, info.height
        )));
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidParameters(
            "crop dimensions must be > 0".into(),
        ));
    }
    let img = pixels_to_image(pixels, info)?;
    let cropped = img.crop_imm(x, y, width, height);
    image_to_decoded(cropped, info.format, info.color_space)
}

/// Rotate an image
pub fn rotate(
    pixels: &[u8],
    info: &ImageInfo,
    degrees: Rotation,
) -> Result<DecodedImage, ImageError> {
    let img = pixels_to_image(pixels, info)?;
    let rotated = match degrees {
        Rotation::R90 => img.rotate90(),
        Rotation::R180 => img.rotate180(),
        Rotation::R270 => img.rotate270(),
    };
    image_to_decoded(rotated, info.format, info.color_space)
}

/// Flip an image
pub fn flip(
    pixels: &[u8],
    info: &ImageInfo,
    direction: FlipDirection,
) -> Result<DecodedImage, ImageError> {
    let img = pixels_to_image(pixels, info)?;
    let flipped = match direction {
        FlipDirection::Horizontal => img.fliph(),
        FlipDirection::Vertical => img.flipv(),
    };
    image_to_decoded(flipped, info.format, info.color_space)
}

/// Convert pixel format
pub fn convert_format(
    pixels: &[u8],
    info: &ImageInfo,
    target: PixelFormat,
) -> Result<DecodedImage, ImageError> {
    let img = pixels_to_image(pixels, info)?;
    let (new_pixels, new_format) = match target {
        PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
        PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
        PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "conversion to {other:?} not supported"
            )));
        }
    };
    Ok(DecodedImage {
        pixels: new_pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format: new_format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Auto-orient an image by applying the EXIF orientation transform.
///
/// Maps EXIF orientation values (1-8) to the correct sequence of
/// rotation and flip operations.
pub fn auto_orient(
    pixels: &[u8],
    info: &ImageInfo,
    orientation: ExifOrientation,
) -> Result<DecodedImage, ImageError> {
    match orientation {
        ExifOrientation::Normal => Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        }),
        ExifOrientation::FlipHorizontal => flip(pixels, info, FlipDirection::Horizontal),
        ExifOrientation::Rotate180 => rotate(pixels, info, Rotation::R180),
        ExifOrientation::FlipVertical => flip(pixels, info, FlipDirection::Vertical),
        ExifOrientation::Transpose => {
            let rotated = rotate(pixels, info, Rotation::R270)?;
            flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate90 => rotate(pixels, info, Rotation::R90),
        ExifOrientation::Transverse => {
            let rotated = rotate(pixels, info, Rotation::R90)?;
            flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate270 => rotate(pixels, info, Rotation::R270),
    }
}

/// Auto-orient using EXIF metadata from encoded data.
///
/// Reads EXIF orientation from the encoded data and applies the correct
/// transform to the pixel data. If no EXIF is found or orientation is
/// normal, returns the pixels unchanged.
pub fn auto_orient_from_exif(
    pixels: &[u8],
    info: &ImageInfo,
    encoded_data: &[u8],
) -> Result<DecodedImage, ImageError> {
    let orientation = match super::metadata::read_exif(encoded_data) {
        Ok(meta) => meta.orientation.unwrap_or(ExifOrientation::Normal),
        Err(_) => ExifOrientation::Normal,
    };
    auto_orient(pixels, info, orientation)
}

fn pixels_to_image(pixels: &[u8], info: &ImageInfo) -> Result<DynamicImage, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => image::RgbImage::from_raw(info.width, info.height, pixels.to_vec())
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| ImageError::InvalidInput("pixel data size mismatch".into())),
        PixelFormat::Rgba8 => image::RgbaImage::from_raw(info.width, info.height, pixels.to_vec())
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| ImageError::InvalidInput("pixel data size mismatch".into())),
        PixelFormat::Gray8 => image::GrayImage::from_raw(info.width, info.height, pixels.to_vec())
            .map(DynamicImage::ImageLuma8)
            .ok_or_else(|| ImageError::InvalidInput("pixel data size mismatch".into())),
        other => Err(ImageError::UnsupportedFormat(format!(
            "transform from {other:?} not supported"
        ))),
    }
}

fn image_to_decoded(
    img: DynamicImage,
    original_format: PixelFormat,
    color_space: ColorSpace,
) -> Result<DecodedImage, ImageError> {
    let (pixels, format) = match original_format {
        PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
        PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
        PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
        _ => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
    };
    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format,
            color_space,
        },
        icc_profile: None,
    })
}

// ─── Extended Geometry ──────────────────────────────────────────────────────

/// Rotate an image by an arbitrary angle (degrees) with bilinear interpolation.
///
/// The output dimensions are the bounding box of the rotated corners.
/// Uncovered regions are filled with `bg_color` (RGB or RGBA depending on format).
pub fn rotate_arbitrary(
    pixels: &[u8],
    info: &ImageInfo,
    degrees: f64,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let rad = degrees.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Compute bounding box of rotated corners
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let corners = [(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)];
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for &(x, y) in &corners {
        let rx = x * cos_a - y * sin_a;
        let ry = x * sin_a + y * cos_a;
        min_x = min_x.min(rx);
        max_x = max_x.max(rx);
        min_y = min_y.min(ry);
        max_y = max_y.max(ry);
    }

    let out_w = (max_x - min_x).ceil() as usize;
    let out_h = (max_y - min_y).ceil() as usize;
    if out_w == 0 || out_h == 0 {
        return Err(ImageError::InvalidParameters(
            "rotation produced zero-size output".into(),
        ));
    }

    let out_cx = out_w as f64 / 2.0;
    let out_cy = out_h as f64 / 2.0;

    let mut output = vec![0u8; out_w * out_h * bpp];

    // For each output pixel, inverse-map to source and sample
    for oy in 0..out_h {
        for ox in 0..out_w {
            let dx = ox as f64 - out_cx;
            let dy = oy as f64 - out_cy;
            // Inverse rotation: rotate by -angle
            let sx = dx * cos_a + dy * sin_a + cx;
            let sy = -dx * sin_a + dy * cos_a + cy;

            let out_idx = (oy * out_w + ox) * bpp;

            if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                // Bilinear interpolation
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                for c in 0..bpp {
                    let p00 = pixels[y0 * w * bpp + x0 * bpp + c] as f64;
                    let p10 = pixels[y0 * w * bpp + x1 * bpp + c] as f64;
                    let p01 = pixels[y1 * w * bpp + x0 * bpp + c] as f64;
                    let p11 = pixels[y1 * w * bpp + x1 * bpp + c] as f64;

                    let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                    output[out_idx + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            } else {
                // Background fill
                for c in 0..bpp {
                    output[out_idx + c] = bg_color.get(c).copied().unwrap_or(0);
                }
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_w as u32,
            height: out_h as u32,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Extend the canvas by adding padding around the image.
///
/// `fill_color` should match the pixel format (3 bytes for RGB8, 4 for RGBA8, etc.).
pub fn pad(
    pixels: &[u8],
    info: &ImageInfo,
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    fill_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = w + left as usize + right as usize;
    let out_h = h + top as usize + bottom as usize;

    // Fill output with background color
    let mut output = Vec::with_capacity(out_w * out_h * bpp);
    for _ in 0..out_w * out_h {
        for c in 0..bpp {
            output.push(fill_color.get(c).copied().unwrap_or(0));
        }
    }

    // Blit original image at (left, top)
    for y in 0..h {
        let src_start = y * w * bpp;
        let dst_start = ((top as usize + y) * out_w + left as usize) * bpp;
        output[dst_start..dst_start + w * bpp]
            .copy_from_slice(&pixels[src_start..src_start + w * bpp]);
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_w as u32,
            height: out_h as u32,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Trim uniform borders from an image.
///
/// Scans inward from each edge, comparing pixels against the top-left corner pixel.
/// Pixels within `threshold` (per-channel absolute difference) are considered border.
pub fn trim(pixels: &[u8], info: &ImageInfo, threshold: u8) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    if w == 0 || h == 0 {
        return Err(ImageError::InvalidParameters("empty image".into()));
    }

    // Reference color: top-left pixel
    let ref_color = &pixels[0..bpp];

    let pixel_matches = |x: usize, y: usize| -> bool {
        let idx = (y * w + x) * bpp;
        for c in 0..bpp {
            if (pixels[idx + c] as i16 - ref_color[c] as i16).unsigned_abs() > threshold as u16 {
                return false;
            }
        }
        true
    };

    // Scan from each edge
    let mut top = 0;
    'top: for y in 0..h {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'top;
            }
        }
        top = y + 1;
    }

    let mut bottom = h;
    'bottom: for y in (top..h).rev() {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'bottom;
            }
        }
        bottom = y;
    }

    let mut left = 0;
    'left: for x in 0..w {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'left;
            }
        }
        left = x + 1;
    }

    let mut right = w;
    'right: for x in (left..w).rev() {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'right;
            }
        }
        right = x;
    }

    if left >= right || top >= bottom {
        // Entire image is uniform border — return 1x1
        return Ok(DecodedImage {
            pixels: ref_color.to_vec(),
            info: ImageInfo {
                width: 1,
                height: 1,
                format: info.format,
                color_space: info.color_space,
            },
            icc_profile: None,
        });
    }

    crop(
        pixels,
        info,
        left as u32,
        top as u32,
        (right - left) as u32,
        (bottom - top) as u32,
    )
}

/// Apply a general 2D affine transform.
///
/// `matrix` is [a, b, tx, c, d, ty] representing:
///   x' = a*x + b*y + tx
///   y' = c*x + d*y + ty
///
/// Output dimensions must be specified. Uses bilinear interpolation.
pub fn affine(
    pixels: &[u8],
    info: &ImageInfo,
    matrix: &[f64; 6],
    out_width: u32,
    out_height: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = out_width as usize;
    let out_h = out_height as usize;

    // Compute inverse matrix for inverse mapping
    let [a, b, tx, c, d, ty] = *matrix;
    let det = a * d - b * c;
    if det.abs() < 1e-10 {
        return Err(ImageError::InvalidParameters(
            "singular affine matrix".into(),
        ));
    }
    let inv_det = 1.0 / det;
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);

    let mut output = vec![0u8; out_w * out_h * bpp];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let sx = ia * ox as f64 + ib * oy as f64 + itx;
            let sy = ic * ox as f64 + id * oy as f64 + ity;

            let out_idx = (oy * out_w + ox) * bpp;

            if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                for ch in 0..bpp {
                    let p00 = pixels[y0 * w * bpp + x0 * bpp + ch] as f64;
                    let p10 = pixels[y0 * w * bpp + x1 * bpp + ch] as f64;
                    let p01 = pixels[y1 * w * bpp + x0 * bpp + ch] as f64;
                    let p11 = pixels[y1 * w * bpp + x1 * bpp + ch] as f64;

                    let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                    output[out_idx + ch] = val.round().clamp(0.0, 255.0) as u8;
                }
            } else {
                for ch in 0..bpp {
                    output[out_idx + ch] = bg_color.get(ch).copied().unwrap_or(0);
                }
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_width,
            height: out_height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Get bytes per pixel for supported formats.
fn bytes_per_pixel(format: PixelFormat) -> Result<usize, ImageError> {
    match format {
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        PixelFormat::Gray8 => Ok(1),
        _ => Err(ImageError::UnsupportedFormat(format!(
            "{format:?} not supported for geometric transforms"
        ))),
    }
}

/// Validate pixel buffer size matches dimensions and format.
fn validate_pixel_buffer(pixels: &[u8], info: &ImageInfo, bpp: usize) -> Result<(), ImageError> {
    let expected = info.width as usize * info.height as usize * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidInput(format!(
            "pixel buffer too small: need {expected}, got {}",
            pixels.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn resize_changes_dimensions() {
        let (px, info) = make_image(64, 64);
        let result = resize(&px, &info, 32, 16, ResizeFilter::Bilinear).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn resize_preserves_format() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 8, 8, ResizeFilter::Nearest).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn resize_pixel_data_length_correct() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 32, 24, ResizeFilter::Lanczos3).unwrap();
        assert_eq!(result.pixels.len(), 32 * 24 * 3);
    }

    #[test]
    fn resize_all_filters_work() {
        let (px, info) = make_image(16, 16);
        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Bilinear,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3,
        ] {
            let result = resize(&px, &info, 8, 8, filter);
            assert!(result.is_ok(), "filter {filter:?} failed");
        }
    }

    #[test]
    fn crop_returns_correct_region() {
        let (px, info) = make_image(32, 32);
        let result = crop(&px, &info, 4, 4, 16, 16).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
        assert_eq!(result.pixels.len(), 16 * 16 * 3);
    }

    #[test]
    fn crop_out_of_bounds_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 10, 10, 10, 10);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::InvalidParameters(_) => {}
            other => panic!("expected InvalidParameters, got {other:?}"),
        }
    }

    #[test]
    fn crop_zero_dimension_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 0, 0, 0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn rotate_90_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn rotate_180_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn rotate_270_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn flip_horizontal_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Horizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
        assert_eq!(result.pixels.len(), px.len());
    }

    #[test]
    fn flip_vertical_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Vertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn convert_rgb8_to_rgba8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn convert_rgb8_to_gray8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Gray8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 8 * 8 * 1);
    }

    #[test]
    fn convert_unsupported_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Nv12);
        assert!(result.is_err());
    }

    #[test]
    fn auto_orient_normal_is_identity() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Normal).unwrap();
        assert_eq!(result.pixels, px);
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate90_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_rotate180_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate270_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_flip_horizontal_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipHorizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_flip_vertical_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipVertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_transpose_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transpose).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_transverse_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transverse).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_all_8_orientations_work() {
        let (px, info) = make_image(16, 8);
        for tag in 1..=8 {
            let orient = ExifOrientation::from_tag(tag);
            let result = auto_orient(&px, &info, orient);
            assert!(result.is_ok(), "orientation {tag} failed");
        }
    }

    #[test]
    fn auto_orient_from_exif_with_no_exif_is_identity() {
        let (px, info) = make_image(16, 8);
        // Non-JPEG data — no EXIF, should return unchanged
        let result = auto_orient_from_exif(&px, &info, &[0x89, 0x50]).unwrap();
        assert_eq!(result.pixels, px);
    }

    // ─── Extended Geometry Tests ────────────────────────────────────────

    #[test]
    fn rotate_arbitrary_0_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 0.0, &bg).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 32);
    }

    #[test]
    fn rotate_arbitrary_90_matches_dimensions() {
        let (px, info) = make_image(32, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 90.0, &bg).unwrap();
        // 90 degrees: output is roughly height x width
        assert!(result.info.width >= 15 && result.info.width <= 17);
        assert!(result.info.height >= 31 && result.info.height <= 33);
    }

    #[test]
    fn rotate_arbitrary_45_expands_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [255, 255, 255];
        let result = rotate_arbitrary(&px, &info, 45.0, &bg).unwrap();
        // 45 degrees expands: side * sqrt(2) ≈ 45
        assert!(result.info.width > 40);
        assert!(result.info.height > 40);
    }

    #[test]
    fn rotate_arbitrary_180_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 180.0, &bg).unwrap();
        // Floating-point bounding box may be ±1 of original
        assert!((result.info.width as i32 - 32).abs() <= 1);
        assert!((result.info.height as i32 - 32).abs() <= 1);
    }

    #[test]
    fn rotate_arbitrary_preserves_format() {
        let (px, info) = make_image(16, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 37.0, &bg).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn pad_symmetric() {
        let (px, info) = make_image(16, 16);
        let result = pad(&px, &info, 4, 4, 4, 4, &[128, 128, 128]).unwrap();
        assert_eq!(result.info.width, 24);
        assert_eq!(result.info.height, 24);
        assert_eq!(result.pixels.len(), 24 * 24 * 3);
    }

    #[test]
    fn pad_asymmetric() {
        let (px, info) = make_image(8, 8);
        let result = pad(&px, &info, 2, 4, 6, 8, &[255, 0, 0]).unwrap();
        assert_eq!(result.info.width, 8 + 4 + 8);
        assert_eq!(result.info.height, 8 + 2 + 6);
    }

    #[test]
    fn pad_preserves_center_pixels() {
        let (px, info) = make_image(4, 4);
        let result = pad(&px, &info, 1, 1, 1, 1, &[0, 0, 0]).unwrap();
        // Check center pixel (1,1) in output = (0,0) in original
        let bpp = 3;
        let out_w = 6;
        let idx = (1 * out_w + 1) * bpp;
        assert_eq!(result.pixels[idx..idx + 3], px[0..3]);
    }

    #[test]
    fn pad_fill_color_correct() {
        let px = vec![128u8; 4 * 4 * 3]; // 4x4 gray
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = pad(&px, &info, 1, 1, 1, 1, &[255, 0, 0]).unwrap();
        // Top-left corner (0,0) should be fill color (red)
        assert_eq!(result.pixels[0], 255); // R
        assert_eq!(result.pixels[1], 0); // G
        assert_eq!(result.pixels[2], 0); // B
    }

    #[test]
    fn trim_removes_uniform_border() {
        // Create 8x8 image with 2-pixel red border around green center
        let mut px = vec![255u8; 8 * 8 * 3]; // all red
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 0; // R
                px[idx + 1] = 255; // G
                px[idx + 2] = 0; // B
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
        assert_eq!(result.info.height, 4);
    }

    #[test]
    fn trim_with_threshold() {
        // Create image with near-uniform border (within threshold)
        let mut px = vec![100u8; 8 * 8 * 3]; // all 100
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 200; // significantly different
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Threshold 0: all 100-valued border trimmed
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
    }

    #[test]
    fn trim_all_uniform_returns_1x1() {
        let px = vec![128u8; 8 * 8 * 3];
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 1);
        assert_eq!(result.info.height, 1);
    }

    #[test]
    fn affine_identity() {
        // Use a larger image so edge bg-fill is a small fraction
        let (px, info) = make_image(64, 64);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = affine(&px, &info, &identity, 64, 64, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 64);
        // Interior pixels should match; edge row/col may differ (bg fill)
        // With 64x64, edge pixels are ~3% of total → low MAE
        let mae: f64 = px
            .iter()
            .zip(result.pixels.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 5.0,
            "identity affine MAE should be < 5.0, got {mae:.2}"
        );
    }

    #[test]
    fn affine_scale_2x() {
        let (px, info) = make_image(8, 8);
        let scale2 = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let result = affine(&px, &info, &scale2, 16, 16, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn affine_singular_matrix_rejected() {
        let (px, info) = make_image(8, 8);
        let singular = [1.0, 2.0, 0.0, 2.0, 4.0, 0.0]; // det = 1*4 - 2*2 = 0
        let result = affine(&px, &info, &singular, 8, 8, &[0, 0, 0]);
        assert!(result.is_err());
    }
}
