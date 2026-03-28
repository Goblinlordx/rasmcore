use image::ImageFormat;

use super::color;
use super::error::ImageError;
use super::types::{ColorSpace, DecodedImage, ImageInfo, PixelFormat};

/// Supported decode formats
const SUPPORTED_FORMATS: &[&str] = &[
    "png", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "qoi", "ico", "tga", "hdr", "pnm", "exr",
    "dds", "jxl", "jp2", "heic",
];

/// Detect image format from header bytes
pub fn detect_format(header: &[u8]) -> Option<String> {
    // Check formats not supported by image crate first
    if is_jxl(header) {
        return Some("jxl".to_string());
    }
    if is_jp2(header) {
        return Some("jp2".to_string());
    }
    #[cfg(feature = "native-heif")]
    if is_heif(header) {
        return Some("heic".to_string());
    }
    image::guess_format(header)
        .ok()
        .and_then(|fmt| format_to_str(fmt))
        .map(String::from)
}

/// Check if data starts with a HEIF/HEIC/AVIF ftyp box with a recognized brand.
#[cfg(feature = "native-heif")]
fn is_heif(header: &[u8]) -> bool {
    rasmcore_isobmff::detect(header).is_some()
}

/// Check if data starts with JPEG 2000 magic bytes.
/// JP2 container: 0x00 0x00 0x00 0x0C 0x6A 0x50 0x20 0x20 (JP2 signature box)
/// J2K codestream: 0xFF 0x4F (SOC marker)
fn is_jp2(header: &[u8]) -> bool {
    // JP2 container signature
    if header.len() >= 12 && header[..4] == [0x00, 0x00, 0x00, 0x0C] && &header[4..8] == b"jP  " {
        return true;
    }
    // Raw J2K codestream (SOC marker)
    if header.len() >= 2 && header[0] == 0xFF && header[1] == 0x4F {
        return true;
    }
    false
}

/// Check if data starts with JPEG XL magic bytes.
/// Bare codestream: 0xFF 0x0A
/// ISOBMFF container: 0x00 0x00 0x00 0x0C 0x4A 0x58 0x4C 0x20 ("....JXL ")
fn is_jxl(header: &[u8]) -> bool {
    if header.len() >= 2 && header[0] == 0xFF && header[1] == 0x0A {
        return true;
    }
    if header.len() >= 12 && header[..4] == [0x00, 0x00, 0x00, 0x0C] && &header[4..8] == b"JXL " {
        return true;
    }
    false
}

/// Decode an image from raw bytes
pub fn decode(data: &[u8]) -> Result<DecodedImage, ImageError> {
    // Formats not supported by image crate — handle first
    if is_jxl(data) {
        return decode_jxl(data);
    }
    if is_jp2(data) {
        return decode_jp2(data);
    }
    #[cfg(feature = "native-heif")]
    if is_heif(data) {
        return decode_heif(data);
    }

    let img = image::load_from_memory(data).map_err(|e| ImageError::InvalidInput(e.to_string()))?;

    let format = detect_pixel_format(&img);
    let pixels = match format {
        PixelFormat::Rgba8 => img.to_rgba8().into_raw(),
        PixelFormat::Rgb8 => img.to_rgb8().into_raw(),
        PixelFormat::Gray8 => img.to_luma8().into_raw(),
        PixelFormat::Gray16 => {
            let luma16 = img.to_luma16();
            luma16
                .as_raw()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        _ => img.to_rgba8().into_raw(),
    };

    // Extract ICC profile from raw bytes (before the image crate strips metadata)
    let icc_profile = extract_icc_profile(data);

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// Extract ICC profile from raw image bytes based on detected format.
fn extract_icc_profile(data: &[u8]) -> Option<Vec<u8>> {
    match detect_format(data)?.as_str() {
        "jpeg" => color::extract_icc_from_jpeg(data),
        "png" => color::extract_icc_from_png(data),
        _ => None,
    }
}

/// Decode and convert to a specific pixel format
pub fn decode_as(data: &[u8], target_format: PixelFormat) -> Result<DecodedImage, ImageError> {
    // JP2/J2K: decode then convert if needed
    if is_jp2(data) {
        let decoded = decode_jp2(data)?;
        if decoded.info.format == target_format {
            return Ok(decoded);
        }
        let img = crate::domain::encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info)?;
        let (pixels, format) = match target_format {
            PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
            PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
            PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "conversion to {other:?} not supported"
                )));
            }
        };
        return Ok(DecodedImage {
            pixels,
            info: ImageInfo {
                width: decoded.info.width,
                height: decoded.info.height,
                format,
                color_space: decoded.info.color_space,
            },
            icc_profile: decoded.icc_profile,
        });
    }

    // JXL: decode then convert
    if is_jxl(data) {
        let decoded = decode_jxl(data)?;
        // If already in target format, return as-is
        if decoded.info.format == target_format {
            return Ok(decoded);
        }
        // Otherwise, use image crate for format conversion
        let img = crate::domain::encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info)?;
        let (pixels, format) = match target_format {
            PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
            PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
            PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "conversion to {other:?} not supported"
                )));
            }
        };
        return Ok(DecodedImage {
            pixels,
            info: ImageInfo {
                width: decoded.info.width,
                height: decoded.info.height,
                format,
                color_space: decoded.info.color_space,
            },
            icc_profile: decoded.icc_profile,
        });
    }

    let img = image::load_from_memory(data).map_err(|e| ImageError::InvalidInput(e.to_string()))?;

    let (pixels, format) = match target_format {
        PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
        PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
        PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
        PixelFormat::Gray16 => {
            let luma16 = img.to_luma16();
            let bytes = luma16
                .as_raw()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            (bytes, PixelFormat::Gray16)
        }
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "conversion to {other:?} not supported"
            )));
        }
    };

    let icc_profile = extract_icc_profile(data);

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// List supported decode formats
pub fn supported_formats() -> Vec<String> {
    SUPPORTED_FORMATS.iter().map(|s| String::from(*s)).collect()
}

fn detect_pixel_format(img: &image::DynamicImage) -> PixelFormat {
    match img.color() {
        image::ColorType::Rgb8 => PixelFormat::Rgb8,
        image::ColorType::Rgba8 => PixelFormat::Rgba8,
        image::ColorType::L8 => PixelFormat::Gray8,
        image::ColorType::L16 => PixelFormat::Gray16,
        _ => PixelFormat::Rgba8,
    }
}

/// Decode a JPEG 2000 image using justjp2.
fn decode_jp2(data: &[u8]) -> Result<DecodedImage, ImageError> {
    let image =
        justjp2::decode(data).map_err(|e| ImageError::InvalidInput(format!("JP2 decode: {e}")))?;

    let width = image.width;
    let height = image.height;
    let num_components = image.components.len();
    let num_pixels = (width * height) as usize;

    // justjp2 returns separate i32 component planes — interleave to u8
    let (pixels, format) = match num_components {
        1 => {
            // Grayscale
            let comp = &image.components[0];
            let precision = comp.precision;
            let pixels: Vec<u8> = comp
                .data
                .iter()
                .map(|&v| clamp_to_u8(v, precision))
                .collect();
            (pixels, PixelFormat::Gray8)
        }
        3 => {
            // RGB (or YCbCr already converted by justjp2)
            let mut pixels = Vec::with_capacity(num_pixels * 3);
            let precision = image.components[0].precision;
            for i in 0..num_pixels {
                pixels.push(clamp_to_u8(image.components[0].data[i], precision));
                pixels.push(clamp_to_u8(image.components[1].data[i], precision));
                pixels.push(clamp_to_u8(image.components[2].data[i], precision));
            }
            (pixels, PixelFormat::Rgb8)
        }
        4 => {
            // RGBA
            let mut pixels = Vec::with_capacity(num_pixels * 4);
            let precision = image.components[0].precision;
            for i in 0..num_pixels {
                pixels.push(clamp_to_u8(image.components[0].data[i], precision));
                pixels.push(clamp_to_u8(image.components[1].data[i], precision));
                pixels.push(clamp_to_u8(image.components[2].data[i], precision));
                pixels.push(clamp_to_u8(image.components[3].data[i], precision));
            }
            (pixels, PixelFormat::Rgba8)
        }
        _ => {
            return Err(ImageError::UnsupportedFormat(format!(
                "JP2 with {num_components} components not supported"
            )));
        }
    };

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width,
            height,
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile: None,
    })
}

/// Clamp an i32 sample to u8 range, handling different bit precisions.
fn clamp_to_u8(value: i32, precision: u32) -> u8 {
    if precision <= 8 {
        value.clamp(0, 255) as u8
    } else {
        // Scale down from higher precision (e.g., 12-bit → 8-bit)
        let max = (1i32 << precision) - 1;
        (value.clamp(0, max) as u64 * 255 / max as u64) as u8
    }
}

/// Decode a JPEG XL image using jxl-oxide.
fn decode_jxl(data: &[u8]) -> Result<DecodedImage, ImageError> {
    use jxl_oxide::JxlImage;

    let image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .map_err(|e| ImageError::InvalidInput(format!("JXL decode: {e}")))?;

    let width = image.width();
    let height = image.height();
    let render = image
        .render_frame(0)
        .map_err(|e| ImageError::ProcessingFailed(format!("JXL render: {e}")))?;

    // Get interleaved pixel buffer (f32) and convert to u8
    let fb = render.image_all_channels();
    let channels = fb.channels();
    let float_buf = fb.buf();
    let pixels: Vec<u8> = float_buf
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    let format = match channels {
        1 => PixelFormat::Gray8,
        3 => PixelFormat::Rgb8,
        4 => PixelFormat::Rgba8,
        _ => PixelFormat::Rgba8,
    };

    // Extract ICC profile if present
    let icc_profile = image.original_icc().map(|icc| icc.to_vec());

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width,
            height,
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// Decode a HEIF/HEIC file using rasmcore-isobmff container parser.
///
/// Currently parses the container and extracts metadata. Pixel decoding requires
/// the HEVC decoder (nonfree-hevc feature) which is not yet implemented.
/// Returns metadata-only result with dimensions and codec info.
#[cfg(feature = "native-heif")]
fn decode_heif(data: &[u8]) -> Result<DecodedImage, ImageError> {
    let file = rasmcore_isobmff::parse(data)
        .map_err(|e| ImageError::InvalidInput(format!("HEIF parse: {e}")))?;

    let img = &file.primary_image;

    // Attempt pixel decode based on codec type
    match img.codec {
        #[cfg(feature = "nonfree-hevc")]
        rasmcore_isobmff::CodecType::Hevc => decode_heif_hevc(&file),

        rasmcore_isobmff::CodecType::Av1 => {
            // AVIF-in-HEIF: delegate to existing AVIF decode path via image crate
            Err(ImageError::UnsupportedFormat(
                "AVIF-in-HEIF: use native AVIF decode path instead".into(),
            ))
        }

        _ => {
            let codec_name = match img.codec {
                rasmcore_isobmff::CodecType::Hevc => "HEVC",
                rasmcore_isobmff::CodecType::Av1 => "AV1",
                rasmcore_isobmff::CodecType::Jpeg => "JPEG",
                rasmcore_isobmff::CodecType::Unknown(_) => "unknown",
            };

            #[cfg(not(feature = "nonfree-hevc"))]
            if matches!(img.codec, rasmcore_isobmff::CodecType::Hevc) {
                return Err(ImageError::UnsupportedFormat(format!(
                    "HEIF container parsed ({}x{}, codec: HEVC, {} bytes) — \
                     enable the 'nonfree-hevc' feature to decode HEVC content",
                    img.width,
                    img.height,
                    img.bitstream.len()
                )));
            }

            Err(ImageError::UnsupportedFormat(format!(
                "HEIF codec {codec_name} not supported",
            )))
        }
    }
}

/// Decode HEIC file using rasmcore-hevc (nonfree).
#[cfg(all(feature = "native-heif", feature = "nonfree-hevc"))]
fn decode_heif_hevc(file: &rasmcore_isobmff::IsobmffFile) -> Result<DecodedImage, ImageError> {
    let img = &file.primary_image;

    // Handle grid images (multiple tiles composited)
    if let Some(grid) = &img.grid {
        return decode_heif_grid(grid, img);
    }

    // Single image decode
    let frame = rasmcore_hevc::decode_frame(&img.bitstream, None)
        .map_err(|e| ImageError::ProcessingFailed(format!("HEVC decode: {e}")))?;

    let icc_profile = img.icc_profile.clone();

    Ok(DecodedImage {
        pixels: frame.pixels,
        info: ImageInfo {
            width: frame.width,
            height: frame.height,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// Decode a grid HEIC image (multiple tiles composited into one output).
#[cfg(all(feature = "native-heif", feature = "nonfree-hevc"))]
fn decode_heif_grid(
    grid: &rasmcore_isobmff::GridDescriptor,
    img: &rasmcore_isobmff::ImageItem,
) -> Result<DecodedImage, ImageError> {
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    let mut output = vec![0u8; out_w * out_h * 3];

    let cols = grid.cols as usize;

    for (tile_idx, tile) in grid.tiles.iter().enumerate() {
        let tile_row = tile_idx / cols;
        let tile_col = tile_idx % cols;

        // Decode this tile
        let frame = rasmcore_hevc::decode_frame(&tile.bitstream, None).map_err(|e| {
            ImageError::ProcessingFailed(format!("HEVC tile {tile_idx} decode: {e}"))
        })?;

        let tw = frame.width as usize;
        let th = frame.height as usize;
        let tile_x = tile_col * tw;
        let tile_y = tile_row * th;

        // Blit tile into output buffer
        for row in 0..th {
            let out_y = tile_y + row;
            if out_y >= out_h {
                break;
            }
            let src_start = row * tw * 3;
            let src_end = src_start + tw.min(out_w - tile_x) * 3;
            let dst_start = (out_y * out_w + tile_x) * 3;
            let copy_len = src_end - src_start;
            if dst_start + copy_len <= output.len() {
                output[dst_start..dst_start + copy_len]
                    .copy_from_slice(&frame.pixels[src_start..src_end]);
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: grid.output_width,
            height: grid.output_height,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        },
        icc_profile: img.icc_profile.clone(),
    })
}

fn format_to_str(fmt: ImageFormat) -> Option<&'static str> {
    match fmt {
        ImageFormat::Png => Some("png"),
        ImageFormat::Jpeg => Some("jpeg"),
        ImageFormat::Gif => Some("gif"),
        ImageFormat::WebP => Some("webp"),
        ImageFormat::Bmp => Some("bmp"),
        ImageFormat::Tiff => Some("tiff"),
        ImageFormat::Avif => Some("avif"),
        ImageFormat::Qoi => Some("qoi"),
        ImageFormat::Ico => Some("ico"),
        ImageFormat::Tga => Some("tga"),
        ImageFormat::Hdr => Some("hdr"),
        ImageFormat::Pnm => Some("pnm"),
        ImageFormat::OpenExr => Some("exr"),
        ImageFormat::Dds => Some("dds"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageFormat::Png).unwrap();
        buf
    }

    fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageFormat::Jpeg).unwrap();
        buf
    }

    #[test]
    fn detect_format_png() {
        let data = make_png(8, 8);
        assert_eq!(detect_format(&data), Some("png".to_string()));
    }

    #[test]
    fn detect_format_jpeg() {
        let data = make_jpeg(8, 8);
        assert_eq!(detect_format(&data), Some("jpeg".to_string()));
    }

    #[test]
    fn detect_format_empty() {
        assert_eq!(detect_format(&[]), None);
    }

    #[test]
    fn detect_format_garbage() {
        assert_eq!(detect_format(&[0xFF, 0x00, 0x42, 0x99]), None);
    }

    #[test]
    fn decode_png_returns_correct_dimensions() {
        let data = make_png(64, 32);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 32);
    }

    #[test]
    fn decode_png_returns_rgb8_format() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn decode_png_pixel_data_length_matches() {
        let data = make_png(10, 10);
        let result = decode(&data).unwrap();
        let expected_len = 10 * 10 * 3;
        assert_eq!(result.pixels.len(), expected_len);
    }

    #[test]
    fn decode_jpeg_returns_correct_dimensions() {
        let data = make_jpeg(100, 50);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.width, 100);
        assert_eq!(result.info.height, 50);
    }

    #[test]
    fn decode_invalid_data_returns_error() {
        let result = decode(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn decode_as_rgba8() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn decode_as_gray8() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Gray8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 8 * 8 * 1);
    }

    #[test]
    fn decode_as_unsupported_returns_error() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Yuv420p);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::UnsupportedFormat(_) => {}
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn supported_formats_includes_common_formats() {
        let fmts = supported_formats();
        for f in [
            "png", "jpeg", "webp", "gif", "bmp", "tiff", "avif", "qoi", "ico", "tga", "hdr", "pnm",
            "exr", "dds", "jxl", "jp2",
        ] {
            assert!(fmts.contains(&f.to_string()), "missing decode format: {f}");
        }
    }

    #[test]
    fn detect_format_jp2_container() {
        // JP2 signature: 0x00 0x00 0x00 0x0C "jP  " (with trailing spaces)
        let header = [
            0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
        ];
        assert_eq!(detect_format(&header), Some("jp2".to_string()));
    }

    #[test]
    fn detect_format_j2k_codestream() {
        // J2K SOC marker: 0xFF 0x4F
        assert_eq!(
            detect_format(&[0xFF, 0x4F, 0xFF, 0x51]),
            Some("jp2".to_string())
        );
    }

    #[test]
    fn detect_format_jxl_bare_codestream() {
        // JXL bare codestream starts with 0xFF 0x0A
        assert_eq!(
            detect_format(&[0xFF, 0x0A, 0x00, 0x00]),
            Some("jxl".to_string())
        );
    }

    #[test]
    fn detect_format_jxl_container() {
        // JXL ISOBMFF container: 0x00 0x00 0x00 0x0C "JXL " 0x0D 0x0A 0x87 0x0A
        let header = [
            0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
        ];
        assert_eq!(detect_format(&header), Some("jxl".to_string()));
    }

    #[test]
    fn decode_color_space_defaults_to_srgb() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn decode_png_no_icc_profile() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert!(result.icc_profile.is_none());
    }

    #[test]
    fn decode_jpeg_no_icc_profile() {
        let data = make_jpeg(8, 8);
        let result = decode(&data).unwrap();
        assert!(result.icc_profile.is_none());
    }

    #[test]
    fn decode_jpeg_with_icc_extracts_profile() {
        let jpeg = make_jpeg(8, 8);
        // Embed a fake ICC profile into the JPEG
        let fake_icc = vec![42u8; 128];
        let jpeg_with_icc =
            crate::domain::encoder::jpeg::embed_icc_profile(&jpeg, &fake_icc).unwrap();
        let result = decode(&jpeg_with_icc).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }

    #[test]
    fn decode_png_with_icc_extracts_profile() {
        let png = make_png(8, 8);
        // Embed a fake ICC profile into the PNG
        let fake_icc = vec![99u8; 200];
        let png_with_icc = crate::domain::encoder::png::embed_icc_profile(&png, &fake_icc).unwrap();
        let result = decode(&png_with_icc).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }

    #[test]
    fn decode_as_preserves_icc_profile() {
        let jpeg = make_jpeg(8, 8);
        let fake_icc = vec![55u8; 64];
        let jpeg_with_icc =
            crate::domain::encoder::jpeg::embed_icc_profile(&jpeg, &fake_icc).unwrap();
        let result = decode_as(&jpeg_with_icc, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }
}
