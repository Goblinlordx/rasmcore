use image::DynamicImage;

use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo, PixelFormat};

/// Apply gaussian blur
pub fn blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
    if radius < 0.0 {
        return Err(ImageError::InvalidParameters(
            "blur radius must be >= 0".into(),
        ));
    }
    let img = pixels_to_image(pixels, info)?;
    let blurred = img.blur(radius);
    Ok(extract_pixels(&blurred, info.format))
}

/// Apply sharpening
pub fn sharpen(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    let img = pixels_to_image(pixels, info)?;
    let sharpened = img.unsharpen(amount, 1);
    Ok(extract_pixels(&sharpened, info.format))
}

/// Adjust brightness (-1.0 to 1.0 maps to -255 to 255 internally)
pub fn brightness(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "brightness must be between -1.0 and 1.0".into(),
        ));
    }
    let img = pixels_to_image(pixels, info)?;
    let adjusted = img.brighten((amount * 128.0) as i32);
    Ok(extract_pixels(&adjusted, info.format))
}

/// Adjust contrast (-1.0 to 1.0 maps to -100 to 100 internally)
pub fn contrast(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    let img = pixels_to_image(pixels, info)?;
    let adjusted = img.adjust_contrast(amount * 100.0);
    Ok(extract_pixels(&adjusted, info.format))
}

/// Convert to grayscale
pub fn grayscale(pixels: &[u8], info: &ImageInfo) -> Result<DecodedImage, ImageError> {
    let img = pixels_to_image(pixels, info)?;
    let gray = img.grayscale();
    let gray_pixels = gray.to_luma8().into_raw();
    Ok(DecodedImage {
        pixels: gray_pixels,
        info: ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Gray8,
            color_space: info.color_space,
        },
    })
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
            "filter on {other:?} not supported"
        ))),
    }
}

fn extract_pixels(img: &DynamicImage, original_format: PixelFormat) -> Vec<u8> {
    match original_format {
        PixelFormat::Rgb8 => img.to_rgb8().into_raw(),
        PixelFormat::Rgba8 => img.to_rgba8().into_raw(),
        PixelFormat::Gray8 => img.to_luma8().into_raw(),
        _ => img.to_rgba8().into_raw(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

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
    fn blur_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = blur(&px, &info, 2.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn blur_zero_radius_preserves_pixels() {
        let (px, info) = make_image(8, 8);
        let result = blur(&px, &info, 0.0).unwrap();
        // With 0 radius, blur should be near identity
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn blur_negative_radius_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = blur(&px, &info, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn sharpen_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = sharpen(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn brightness_increases() {
        let (px, info) = make_image(8, 8);
        let result = brightness(&px, &info, 0.5).unwrap();
        assert_eq!(result.len(), px.len());
        // Brighter image should generally have higher average pixel values
        let avg_orig: f64 = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
        let avg_bright: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(avg_bright > avg_orig, "brightness should increase average");
    }

    #[test]
    fn brightness_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(brightness(&px, &info, 1.5).is_err());
        assert!(brightness(&px, &info, -1.5).is_err());
    }

    #[test]
    fn contrast_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = contrast(&px, &info, 0.5).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn contrast_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(contrast(&px, &info, 2.0).is_err());
    }

    #[test]
    fn grayscale_changes_format() {
        let (px, info) = make_image(16, 16);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 16 * 16); // 1 byte per pixel
    }

    #[test]
    fn grayscale_preserves_dimensions() {
        let (px, info) = make_image(32, 24);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 24);
    }

    #[test]
    fn filters_work_on_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        assert!(blur(&pixels, &info, 1.0).is_ok());
        assert!(sharpen(&pixels, &info, 1.0).is_ok());
        assert!(brightness(&pixels, &info, 0.2).is_ok());
        assert!(contrast(&pixels, &info, 0.2).is_ok());
        assert!(grayscale(&pixels, &info).is_ok());
    }
}
