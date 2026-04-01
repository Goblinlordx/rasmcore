use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo, PixelFormat, ResizeFilter};

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
        PixelFormat::Rgb16 => fir::PixelType::U16x3,
        PixelFormat::Rgba16 => fir::PixelType::U16x4,
        PixelFormat::Gray16 => fir::PixelType::U16,
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
