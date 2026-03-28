use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// JPEG encode configuration.
#[derive(Debug, Clone)]
pub struct JpegEncodeConfig {
    /// Quality level 1-100 (default: 85).
    pub quality: u8,
    /// Emit progressive JPEG (default: false).
    pub progressive: bool,
}

impl Default for JpegEncodeConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            progressive: false,
        }
    }
}

/// Map our pixel format to rasmcore-jpeg's PixelFormat.
fn to_jpeg_format(format: PixelFormat) -> Result<rasmcore_jpeg::PixelFormat, ImageError> {
    match format {
        PixelFormat::Rgb8 => Ok(rasmcore_jpeg::PixelFormat::Rgb8),
        PixelFormat::Rgba8 => Ok(rasmcore_jpeg::PixelFormat::Rgba8),
        PixelFormat::Gray8 => Ok(rasmcore_jpeg::PixelFormat::Gray8),
        other => Err(ImageError::UnsupportedFormat(format!(
            "JPEG encoding from {other:?} pixel format not supported"
        ))),
    }
}

/// Encode raw pixel data to JPEG with the given configuration.
///
/// Uses rasmcore-jpeg (pure Rust, MIT/Apache-2.0) as the encoding backend.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    config: &JpegEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let jpeg_format = to_jpeg_format(info.format)?;

    let jpeg_config = rasmcore_jpeg::EncodeConfig {
        quality: config.quality,
        progressive: config.progressive,
        ..Default::default()
    };

    rasmcore_jpeg::encode(pixels, info.width, info.height, jpeg_format, &jpeg_config)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))
}

/// Encode a DynamicImage to JPEG (convenience wrapper for pipeline sink compatibility).
pub fn encode(
    img: &image::DynamicImage,
    info: &ImageInfo,
    config: &JpegEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let (pixels, pixel_format) = match img {
        image::DynamicImage::ImageRgb8(buf) => (buf.as_raw().as_slice(), PixelFormat::Rgb8),
        image::DynamicImage::ImageRgba8(buf) => (buf.as_raw().as_slice(), PixelFormat::Rgba8),
        image::DynamicImage::ImageLuma8(buf) => (buf.as_raw().as_slice(), PixelFormat::Gray8),
        _ => {
            let rgb = img.to_rgb8();
            let adjusted_info = ImageInfo {
                format: PixelFormat::Rgb8,
                ..*info
            };
            return encode_pixels(rgb.as_raw(), &adjusted_info, config);
        }
    };

    let adjusted_info = ImageInfo {
        format: pixel_format,
        ..*info
    };
    encode_pixels(pixels, &adjusted_info, config)
}

/// Embed an ICC profile into already-encoded JPEG data.
///
/// Inserts APP2 markers with "ICC_PROFILE\0" signature after SOI.
/// Large profiles are split into 64KB chunks per JPEG spec.
pub fn embed_icc_profile(jpeg_data: &[u8], icc_profile: &[u8]) -> Result<Vec<u8>, ImageError> {
    if jpeg_data.len() < 2 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
        return Err(ImageError::InvalidInput("not a valid JPEG".into()));
    }

    const MAX_CHUNK: usize = 65519;
    let chunks: Vec<&[u8]> = icc_profile.chunks(MAX_CHUNK).collect();
    let num_chunks = chunks.len().min(255) as u8;

    let mut result = Vec::with_capacity(jpeg_data.len() + icc_profile.len() + chunks.len() * 18);
    result.extend_from_slice(&jpeg_data[..2]);

    for (i, chunk) in chunks.iter().enumerate() {
        let seg_len = (2 + 14 + chunk.len()) as u16;
        result.push(0xFF);
        result.push(0xE2);
        result.extend_from_slice(&seg_len.to_be_bytes());
        result.extend_from_slice(b"ICC_PROFILE\0");
        result.push((i + 1) as u8);
        result.push(num_chunks);
        result.extend_from_slice(chunk);
    }

    result.extend_from_slice(&jpeg_data[2..]);
    Ok(result)
}

/// Embed EXIF data into already-encoded JPEG data.
pub fn embed_exif(jpeg_data: &[u8], exif_data: &[u8]) -> Result<Vec<u8>, ImageError> {
    if jpeg_data.len() < 2 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
        return Err(ImageError::InvalidInput("not a valid JPEG".into()));
    }

    let seg_len = (exif_data.len() + 2) as u16;
    let mut result = Vec::with_capacity(jpeg_data.len() + exif_data.len() + 4);
    result.extend_from_slice(&jpeg_data[..2]);
    result.push(0xFF);
    result.push(0xE1);
    result.extend_from_slice(&seg_len.to_be_bytes());
    result.extend_from_slice(exif_data);
    result.extend_from_slice(&jpeg_data[2..]);
    Ok(result)
}

/// Embed XMP data into already-encoded JPEG data.
pub fn embed_xmp(jpeg_data: &[u8], xmp_data: &[u8]) -> Result<Vec<u8>, ImageError> {
    if jpeg_data.len() < 2 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
        return Err(ImageError::InvalidInput("not a valid JPEG".into()));
    }

    let prefix = b"http://ns.adobe.com/xap/1.0/\x00";
    let payload_len = prefix.len() + xmp_data.len();
    let seg_len = (payload_len + 2) as u16;

    let mut result = Vec::with_capacity(jpeg_data.len() + payload_len + 4);
    result.extend_from_slice(&jpeg_data[..2]);
    result.push(0xFF);
    result.push(0xE1);
    result.extend_from_slice(&seg_len.to_be_bytes());
    result.extend_from_slice(prefix);
    result.extend_from_slice(xmp_data);
    result.extend_from_slice(&jpeg_data[2..]);
    Ok(result)
}

/// Embed IPTC data into already-encoded JPEG data.
pub fn embed_iptc(jpeg_data: &[u8], iptc_data: &[u8]) -> Result<Vec<u8>, ImageError> {
    if jpeg_data.len() < 2 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
        return Err(ImageError::InvalidInput("not a valid JPEG".into()));
    }

    let prefix = b"Photoshop 3.0\x00";
    let payload_len = prefix.len() + iptc_data.len();
    let seg_len = (payload_len + 2) as u16;

    let mut result = Vec::with_capacity(jpeg_data.len() + payload_len + 4);
    result.extend_from_slice(&jpeg_data[..2]);
    result.push(0xFF);
    result.push(0xED);
    result.extend_from_slice(&seg_len.to_be_bytes());
    result.extend_from_slice(prefix);
    result.extend_from_slice(iptc_data);
    result.extend_from_slice(&jpeg_data[2..]);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_test_pixels() -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_produces_valid_jpeg() {
        let (pixels, info) = make_test_pixels();
        let result = encode_pixels(&pixels, &info, &JpegEncodeConfig::default()).unwrap();
        assert_eq!(&result[..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    #[test]
    fn encode_with_custom_quality() {
        let (pixels, info) = make_test_pixels();
        let result = encode_pixels(
            &pixels,
            &info,
            &JpegEncodeConfig {
                quality: 50,
                progressive: false,
            },
        );
        assert!(result.is_ok());
    }

    #[test]
    fn encode_gray8() {
        let pixels: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &JpegEncodeConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn encode_rgba8() {
        let pixels: Vec<u8> = (0..16 * 16 * 4).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &JpegEncodeConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn encode_decodable_by_image_crate() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32u8 {
            for x in 0..32u8 {
                pixels.push(x * 8);
                pixels.push(y * 8);
                pixels.push(128);
            }
        }
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let config = JpegEncodeConfig::default();
        let jpeg = encode_pixels(&pixels, &info, &config).unwrap();

        let result = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg);
        assert!(
            result.is_ok(),
            "JPEG should be decodable: {:?}",
            result.err()
        );
        let img = result.unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (32, 32));
    }

    #[test]
    fn default_config_quality_is_85() {
        assert_eq!(JpegEncodeConfig::default().quality, 85);
    }

    #[test]
    fn determinism() {
        let (pixels, info) = make_test_pixels();
        let config = JpegEncodeConfig::default();
        let a = encode_pixels(&pixels, &info, &config).unwrap();
        let b = encode_pixels(&pixels, &info, &config).unwrap();
        assert_eq!(a, b, "JPEG output must be deterministic");
    }
}
