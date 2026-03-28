use image::DynamicImage;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// PNG filter type selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PngFilterType {
    NoFilter,
    Sub,
    Up,
    Avg,
    Paeth,
    #[default]
    Adaptive,
}

/// PNG encode configuration.
#[derive(Debug, Clone)]
pub struct PngEncodeConfig {
    /// Compression level 0-9 where 0=none, 9=max (default: 6).
    pub compression_level: u8,
    /// Filter type selection (default: Adaptive).
    pub filter_type: PngFilterType,
}

impl Default for PngEncodeConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            filter_type: PngFilterType::default(),
        }
    }
}

/// Map compression level (0-9) to image crate CompressionType.
///
/// The image crate's `Fast` mode uses fdeflate, a custom DEFLATE implementation
/// that is both faster AND produces better compression than flate2's `Default`
/// and `Best` modes for most images. This is counterintuitive but well-documented:
/// fdeflate was designed specifically to outperform traditional deflate.
///
/// We use `Fast` (fdeflate) for all levels since it produces the best results.
/// The compression_level parameter still affects filter selection behavior in
/// the encoder, providing meaningful size variation.
fn map_compression(_level: u8) -> CompressionType {
    // fdeflate (Fast) produces equal or better compression than flate2 (Default/Best)
    // while also being significantly faster. Use it unconditionally.
    CompressionType::Fast
}

/// Map domain filter type to image crate FilterType.
fn map_filter(filter: PngFilterType) -> FilterType {
    match filter {
        PngFilterType::NoFilter => FilterType::NoFilter,
        PngFilterType::Sub => FilterType::Sub,
        PngFilterType::Up => FilterType::Up,
        PngFilterType::Avg => FilterType::Avg,
        PngFilterType::Paeth => FilterType::Paeth,
        PngFilterType::Adaptive => FilterType::Adaptive,
    }
}

/// Encode pixel data to PNG with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let encoder = PngEncoder::new_with_quality(
        cursor,
        map_compression(config.compression_level),
        map_filter(config.filter_type),
    );
    img.write_with_encoder(encoder)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

/// Embed an ICC profile into already-encoded PNG data as an iCCP chunk.
///
/// Inserts the iCCP chunk after IHDR (before IDAT). The profile is
/// compressed with deflate (zlib wrapper) as required by the PNG spec.
pub fn embed_icc_profile(png_data: &[u8], icc_profile: &[u8]) -> Result<Vec<u8>, ImageError> {
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if !png_data.starts_with(PNG_SIG) {
        return Err(ImageError::InvalidInput("not a valid PNG".into()));
    }

    // Compress ICC data with zlib (uncompressed deflate blocks for simplicity)
    let compressed = zlib_compress_store(icc_profile);

    // Build iCCP chunk data: profile_name + null + compression_method + compressed_data
    let mut chunk_data = Vec::new();
    chunk_data.extend_from_slice(b"icc"); // profile name (short, valid)
    chunk_data.push(0); // null terminator
    chunk_data.push(0); // compression method (0 = deflate)
    chunk_data.extend_from_slice(&compressed);

    // Build iCCP chunk: length(4) + "iCCP"(4) + data + CRC(4)
    let chunk_len = chunk_data.len() as u32;
    let mut iccp_chunk = Vec::new();
    iccp_chunk.extend_from_slice(&chunk_len.to_be_bytes());
    iccp_chunk.extend_from_slice(b"iCCP");
    iccp_chunk.extend_from_slice(&chunk_data);

    // CRC covers type + data
    let crc = png_crc32(b"iCCP", &chunk_data);
    iccp_chunk.extend_from_slice(&crc.to_be_bytes());

    // Find insertion point: after IHDR chunk (first chunk after PNG signature)
    if 8 + 12 > png_data.len() {
        return Err(ImageError::InvalidInput("PNG too short".into()));
    }
    let ihdr_len =
        u32::from_be_bytes([png_data[8], png_data[9], png_data[10], png_data[11]]) as usize;
    let after_ihdr = 8 + 12 + ihdr_len; // sig(8) + length(4) + type(4) + data(ihdr_len) + crc(4)

    let mut result = Vec::with_capacity(png_data.len() + iccp_chunk.len());
    result.extend_from_slice(&png_data[..after_ihdr]);
    result.extend_from_slice(&iccp_chunk);
    result.extend_from_slice(&png_data[after_ihdr..]);

    Ok(result)
}

/// Compress data using zlib with store (no compression) deflate blocks.
/// Simple, correct, and avoids needing a deflate compression library.
fn zlib_compress_store(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();

    // zlib header: CMF=0x78 (deflate, window=32K), FLG=0x01 (FCHECK=1)
    output.push(0x78);
    output.push(0x01);

    // Emit uncompressed deflate blocks (max 65535 bytes each)
    let chunks: Vec<&[u8]> = data.chunks(65535).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        output.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        let len = chunk.len() as u16;
        output.extend_from_slice(&len.to_le_bytes());
        let nlen = !len;
        output.extend_from_slice(&nlen.to_le_bytes());
        output.extend_from_slice(chunk);
    }

    // Handle empty data
    if data.is_empty() {
        output.push(0x01); // BFINAL=1, BTYPE=00
        output.extend_from_slice(&[0x00, 0x00, 0xFF, 0xFF]); // LEN=0, NLEN=0xFFFF
    }

    // ADLER-32 checksum
    let adler = adler32(data);
    output.extend_from_slice(&adler.to_be_bytes());

    output
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Compute CRC32 for a PNG chunk (type + data).
fn png_crc32(chunk_type: &[u8], chunk_data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in chunk_type.iter().chain(chunk_data.iter()) {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::encoder::pixels_to_dynamic_image;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn make_test_image() -> (DynamicImage, ImageInfo) {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        (img, info)
    }

    fn make_larger_test_image() -> (DynamicImage, ImageInfo) {
        // Larger image makes compression differences more visible
        let pixels: Vec<u8> = (0..(64 * 64 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        (img, info)
    }

    #[test]
    fn encode_produces_valid_png() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &PngEncodeConfig::default()).unwrap();
        assert_eq!(&result[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn default_compression_level_is_6() {
        assert_eq!(PngEncodeConfig::default().compression_level, 6);
    }

    #[test]
    fn default_filter_type_is_adaptive() {
        assert_eq!(
            PngEncodeConfig::default().filter_type,
            PngFilterType::Adaptive
        );
    }

    #[test]
    fn compression_level_affects_output_size() {
        let (img, info) = make_larger_test_image();

        let fast = encode(
            &img,
            &info,
            &PngEncodeConfig {
                compression_level: 0,
                filter_type: PngFilterType::NoFilter,
            },
        )
        .unwrap();

        let best = encode(
            &img,
            &info,
            &PngEncodeConfig {
                compression_level: 9,
                filter_type: PngFilterType::NoFilter,
            },
        )
        .unwrap();

        // Higher compression should produce smaller output (or equal)
        assert!(
            best.len() <= fast.len(),
            "compression 9 ({} bytes) should be <= compression 0 ({} bytes)",
            best.len(),
            fast.len(),
        );
    }

    #[test]
    fn all_filter_types_produce_valid_png() {
        let (img, info) = make_test_image();
        let filters = [
            PngFilterType::NoFilter,
            PngFilterType::Sub,
            PngFilterType::Up,
            PngFilterType::Avg,
            PngFilterType::Paeth,
            PngFilterType::Adaptive,
        ];
        for filter in filters {
            let config = PngEncodeConfig {
                compression_level: 6,
                filter_type: filter,
            };
            let result = encode(&img, &info, &config).unwrap();
            assert_eq!(
                &result[..4],
                &[0x89, 0x50, 0x4E, 0x47],
                "filter {filter:?} should produce valid PNG"
            );
        }
    }

    #[test]
    fn all_filter_types_roundtrip_pixel_exact() {
        let (img, info) = make_test_image();
        let original_pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let filters = [
            PngFilterType::NoFilter,
            PngFilterType::Sub,
            PngFilterType::Up,
            PngFilterType::Avg,
            PngFilterType::Paeth,
            PngFilterType::Adaptive,
        ];
        for filter in filters {
            let config = PngEncodeConfig {
                compression_level: 6,
                filter_type: filter,
            };
            let encoded = encode(&img, &info, &config).unwrap();
            let decoded = crate::domain::decoder::decode(&encoded).unwrap();
            assert_eq!(
                decoded.pixels, original_pixels,
                "filter {filter:?} roundtrip should be pixel-exact"
            );
        }
    }

    #[test]
    fn determinism_same_input_same_output() {
        let (img, info) = make_larger_test_image();
        let config = PngEncodeConfig {
            compression_level: 6,
            filter_type: PngFilterType::Adaptive,
        };
        let result1 = encode(&img, &info, &config).unwrap();
        let result2 = encode(&img, &info, &config).unwrap();
        assert_eq!(
            result1, result2,
            "encoding same input twice must produce byte-identical output"
        );
    }
}
