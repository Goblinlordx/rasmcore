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
}
