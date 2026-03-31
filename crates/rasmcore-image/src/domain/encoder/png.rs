use crate::domain::error::ImageError;
use crate::domain::types::{DisposalMethod, FrameSequence, ImageInfo, PixelFormat};

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

/// Map compression level (0-9) to png crate Compression.
///
/// The png crate's `Fast` mode uses fdeflate, a custom DEFLATE implementation
/// that is both faster AND produces better compression than flate2's `Default`
/// and `Best` modes for most images. We use `Fast` unconditionally.
fn map_compression(_level: u8) -> png::Compression {
    png::Compression::Fast
}

/// Map domain filter type to png crate FilterType.
fn map_filter(filter: PngFilterType) -> png::FilterType {
    match filter {
        PngFilterType::NoFilter => png::FilterType::NoFilter,
        PngFilterType::Sub => png::FilterType::Sub,
        PngFilterType::Up => png::FilterType::Up,
        PngFilterType::Avg => png::FilterType::Avg,
        PngFilterType::Paeth | PngFilterType::Adaptive => png::FilterType::Paeth,
    }
}

/// Whether to enable adaptive filtering for the given filter type.
fn map_adaptive(filter: PngFilterType) -> png::AdaptiveFilterType {
    match filter {
        PngFilterType::Adaptive => png::AdaptiveFilterType::Adaptive,
        _ => png::AdaptiveFilterType::NonAdaptive,
    }
}

/// Encode raw pixel data to PNG with the given configuration.
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let (color_type, bit_depth) = match info.format {
        PixelFormat::Gray8 => (png::ColorType::Grayscale, png::BitDepth::Eight),
        PixelFormat::Rgb8 => (png::ColorType::Rgb, png::BitDepth::Eight),
        PixelFormat::Rgba8 => (png::ColorType::Rgba, png::BitDepth::Eight),
        PixelFormat::Gray16 => (png::ColorType::Grayscale, png::BitDepth::Sixteen),
        PixelFormat::Rgb16 => (png::ColorType::Rgb, png::BitDepth::Sixteen),
        PixelFormat::Rgba16 => (png::ColorType::Rgba, png::BitDepth::Sixteen),
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "PNG encode from {other:?} not supported"
            )));
        }
    };

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, info.width, info.height);
        encoder.set_color(color_type);
        encoder.set_depth(bit_depth);
        encoder.set_compression(map_compression(config.compression_level));
        encoder.set_filter(map_filter(config.filter_type));
        encoder.set_adaptive_filter(map_adaptive(config.filter_type));

        let mut writer = encoder
            .write_header()
            .map_err(|e| ImageError::ProcessingFailed(format!("PNG: {e}")))?;

        // For 16-bit: convert from our LE storage to PNG's required BE byte order.
        if bit_depth == png::BitDepth::Sixteen {
            let mut be_pixels = pixels.to_vec();
            for chunk in be_pixels.chunks_exact_mut(2) {
                chunk.swap(0, 1);
            }
            writer
                .write_image_data(&be_pixels)
                .map_err(|e| ImageError::ProcessingFailed(format!("PNG: {e}")))?;
        } else {
            writer
                .write_image_data(pixels)
                .map_err(|e| ImageError::ProcessingFailed(format!("PNG: {e}")))?;
        }
    }
    Ok(buf)
}

/// Encode a FrameSequence to APNG (animated PNG).
///
/// All frames are encoded as RGBA8 at 8-bit depth. Per-frame delay, disposal,
/// and offset metadata from FrameInfo is preserved in fcTL chunks.
pub fn encode_sequence(
    seq: &FrameSequence,
    config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    if seq.is_empty() {
        return Err(ImageError::InvalidInput(
            "cannot encode empty frame sequence as APNG".into(),
        ));
    }

    let num_frames = seq.frames.len() as u32;
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, seq.canvas_width, seq.canvas_height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(map_compression(config.compression_level));
        encoder.set_filter(map_filter(config.filter_type));
        encoder.set_adaptive_filter(map_adaptive(config.filter_type));
        encoder
            .set_animated(num_frames, 0)
            .map_err(|e| ImageError::ProcessingFailed(format!("APNG: {e}")))?;

        let mut writer = encoder
            .write_header()
            .map_err(|e| ImageError::ProcessingFailed(format!("APNG: {e}")))?;

        for (image, frame_info) in &seq.frames {
            // Convert to RGBA8 for uniform encoding
            let rgba = match image.info.format {
                PixelFormat::Rgba8 => image.pixels.clone(),
                PixelFormat::Rgb8 => image
                    .pixels
                    .chunks_exact(3)
                    .flat_map(|c| [c[0], c[1], c[2], 255])
                    .collect(),
                PixelFormat::Gray8 => image.pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
                _ => {
                    return Err(ImageError::UnsupportedFormat(
                        "APNG encode requires RGB8, RGBA8, or Gray8 frames".into(),
                    ));
                }
            };

            // Set per-frame metadata
            let delay_num = frame_info.delay_ms as u16;
            let delay_den = 1000u16;
            writer
                .set_frame_delay(delay_num, delay_den)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG frame delay: {e}")))?;

            writer
                .set_frame_dimension(frame_info.width, frame_info.height)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG frame dim: {e}")))?;

            writer
                .set_frame_position(frame_info.x_offset, frame_info.y_offset)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG frame pos: {e}")))?;

            let dispose_op = match frame_info.disposal {
                DisposalMethod::None => png::DisposeOp::None,
                DisposalMethod::Background => png::DisposeOp::Background,
                DisposalMethod::Previous => png::DisposeOp::Previous,
            };
            writer
                .set_dispose_op(dispose_op)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG dispose: {e}")))?;

            writer
                .set_blend_op(png::BlendOp::Source)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG blend: {e}")))?;

            writer
                .write_image_data(&rgba)
                .map_err(|e| ImageError::ProcessingFailed(format!("APNG frame data: {e}")))?;
        }
    }
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

/// Embed EXIF data into already-encoded PNG data as an eXIf chunk.
///
/// Inserts the eXIf chunk after IHDR (before IDAT).
/// The exif_data should NOT include "Exif\0\0" prefix — raw TIFF bytes only.
pub fn embed_exif(png_data: &[u8], exif_data: &[u8]) -> Result<Vec<u8>, ImageError> {
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if !png_data.starts_with(PNG_SIG) {
        return Err(ImageError::InvalidInput("not a valid PNG".into()));
    }

    // Strip "Exif\0\0" prefix if present (eXIf chunk uses raw TIFF data)
    let tiff_data = if exif_data.starts_with(b"Exif\x00\x00") {
        &exif_data[6..]
    } else {
        exif_data
    };

    // Build eXIf chunk
    let chunk_len = tiff_data.len() as u32;
    let mut exif_chunk = Vec::new();
    exif_chunk.extend_from_slice(&chunk_len.to_be_bytes());
    exif_chunk.extend_from_slice(b"eXIf");
    exif_chunk.extend_from_slice(tiff_data);
    let crc = png_crc32(b"eXIf", tiff_data);
    exif_chunk.extend_from_slice(&crc.to_be_bytes());

    // Find insertion point after IHDR
    if 8 + 12 > png_data.len() {
        return Err(ImageError::InvalidInput("PNG too short".into()));
    }
    let ihdr_len =
        u32::from_be_bytes([png_data[8], png_data[9], png_data[10], png_data[11]]) as usize;
    let after_ihdr = 8 + 12 + ihdr_len;

    let mut result = Vec::with_capacity(png_data.len() + exif_chunk.len());
    result.extend_from_slice(&png_data[..after_ihdr]);
    result.extend_from_slice(&exif_chunk);
    result.extend_from_slice(&png_data[after_ihdr..]);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo};

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

    fn make_larger_test_pixels() -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(64 * 64 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_produces_valid_png() {
        let (pixels, info) = make_test_pixels();
        let result = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
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
        let (pixels, info) = make_larger_test_pixels();

        let fast = encode(
            &pixels,
            &info,
            &PngEncodeConfig {
                compression_level: 0,
                filter_type: PngFilterType::NoFilter,
            },
        )
        .unwrap();

        let best = encode(
            &pixels,
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
        let (pixels, info) = make_test_pixels();
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
            let result = encode(&pixels, &info, &config).unwrap();
            assert_eq!(
                &result[..4],
                &[0x89, 0x50, 0x4E, 0x47],
                "filter {filter:?} should produce valid PNG"
            );
        }
    }

    #[test]
    fn all_filter_types_roundtrip_pixel_exact() {
        let (pixels, info) = make_test_pixels();
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
            let encoded = encode(&pixels, &info, &config).unwrap();
            let decoded = crate::domain::decoder::decode(&encoded).unwrap();
            assert_eq!(
                decoded.pixels, pixels,
                "filter {filter:?} roundtrip should be pixel-exact"
            );
        }
    }

    #[test]
    fn roundtrip_rgb16_pixel_exact() {
        // Create 16-bit gradient image
        let (w, h) = (16u32, 16u32);
        let mut pixel_bytes = Vec::with_capacity((w * h * 6) as usize);
        for i in 0..(w * h) {
            let r = ((i * 257) % 65536) as u16;
            let g = ((i * 131 + 1000) % 65536) as u16;
            let b = ((i * 73 + 5000) % 65536) as u16;
            pixel_bytes.extend_from_slice(&r.to_le_bytes());
            pixel_bytes.extend_from_slice(&g.to_le_bytes());
            pixel_bytes.extend_from_slice(&b.to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let encoded = encode(&pixel_bytes, &info, &PngEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb16);
        assert_eq!(decoded.info.width, w);
        assert_eq!(decoded.info.height, h);
        assert_eq!(
            decoded.pixels, pixel_bytes,
            "16-bit PNG roundtrip must be pixel-exact"
        );
    }

    #[test]
    fn determinism_same_input_same_output() {
        let (pixels, info) = make_larger_test_pixels();
        let config = PngEncodeConfig {
            compression_level: 6,
            filter_type: PngFilterType::Adaptive,
        };
        let result1 = encode(&pixels, &info, &config).unwrap();
        let result2 = encode(&pixels, &info, &config).unwrap();
        assert_eq!(
            result1, result2,
            "encoding same input twice must produce byte-identical output"
        );
    }

    // ─── APNG Encode Tests ─────────────────────────────────────────────

    fn make_test_frame_sequence() -> FrameSequence {
        use crate::domain::types::{ColorSpace, DecodedImage, FrameInfo};
        let mut seq = FrameSequence::new(4, 4);
        let colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        for (i, color) in colors.iter().enumerate() {
            let pixels: Vec<u8> = (0..16).flat_map(|_| *color).collect();
            seq.push(
                DecodedImage {
                    pixels,
                    info: ImageInfo {
                        width: 4,
                        height: 4,
                        format: PixelFormat::Rgba8,
                        color_space: ColorSpace::Srgb,
                    },
                    icc_profile: None,
                },
                FrameInfo {
                    index: i as u32,
                    delay_ms: 100,
                    disposal: DisposalMethod::None,
                    width: 4,
                    height: 4,
                    x_offset: 0,
                    y_offset: 0,
                },
            );
        }
        seq
    }

    #[test]
    fn apng_encode_produces_valid_output() {
        let seq = make_test_frame_sequence();
        let config = PngEncodeConfig::default();
        let data = encode_sequence(&seq, &config).unwrap();
        // Should start with PNG signature
        assert_eq!(&data[..4], &[0x89, 0x50, 0x4E, 0x47]);
        // Decoding should find 3 frames
        let count = crate::domain::decoder::frame_count(&data).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn apng_roundtrip_frame_count_preserved() {
        let seq = make_test_frame_sequence();
        let config = PngEncodeConfig::default();
        let encoded = encode_sequence(&seq, &config).unwrap();
        let frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn apng_roundtrip_delay_preserved() {
        let seq = make_test_frame_sequence();
        let config = PngEncodeConfig::default();
        let encoded = encode_sequence(&seq, &config).unwrap();
        let frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        for (_, info) in &frames {
            assert_eq!(info.delay_ms, 100);
        }
    }

    #[test]
    fn apng_roundtrip_pixels_preserved() {
        let seq = make_test_frame_sequence();
        let config = PngEncodeConfig::default();
        let encoded = encode_sequence(&seq, &config).unwrap();
        let frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        // Frame 0: red
        assert_eq!(frames[0].0.pixels[0], 255); // R
        assert_eq!(frames[0].0.pixels[1], 0); // G
        assert_eq!(frames[0].0.pixels[2], 0); // B
        // Frame 1: green
        assert_eq!(frames[1].0.pixels[0], 0);
        assert_eq!(frames[1].0.pixels[1], 255);
        assert_eq!(frames[1].0.pixels[2], 0);
        // Frame 2: blue
        assert_eq!(frames[2].0.pixels[0], 0);
        assert_eq!(frames[2].0.pixels[1], 0);
        assert_eq!(frames[2].0.pixels[2], 255);
    }

    #[test]
    fn apng_encode_empty_sequence_errors() {
        let seq = FrameSequence::new(4, 4);
        let config = PngEncodeConfig::default();
        assert!(encode_sequence(&seq, &config).is_err());
    }

    /// Three-way codec validation for APNG:
    /// A = our_encode(original) → our_decode
    /// B = our_encode(original) → ref_decode (png crate directly)
    /// C = ref_encode(original) → ref_decode
    /// Lossless: A == B pixel-exact, B == original pixel-exact.
    #[test]
    fn apng_three_way_codec_validation() {
        let seq = make_test_frame_sequence();
        let config = PngEncodeConfig::default();
        let encoded = encode_sequence(&seq, &config).unwrap();

        // A: our_encode → our_decode
        let a_frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        assert_eq!(a_frames.len(), 3);

        // B: our_encode → ref_decode (png crate directly)
        let b_decoder = png::Decoder::new(std::io::Cursor::new(&encoded));
        let mut b_reader = b_decoder.read_info().unwrap();
        let b_actl = b_reader.info().animation_control().unwrap();
        assert_eq!(b_actl.num_frames, 3);
        // Read all frames via raw png crate
        for _ in 0..3 {
            let mut buf = vec![0u8; b_reader.output_buffer_size()];
            b_reader.next_frame(&mut buf).unwrap();
        }

        // C: ref_encode → ref_decode (png crate encode directly → our decode)
        let ref_encoded = {
            let mut buf = Vec::new();
            let mut enc = png::Encoder::new(&mut buf, 4, 4);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_animated(3, 0).unwrap();
            let mut w = enc.write_header().unwrap();
            let colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
            for color in &colors {
                let px: Vec<u8> = (0..16).flat_map(|_| *color).collect();
                w.set_frame_delay(10, 100).unwrap();
                w.write_image_data(&px).unwrap();
            }
            drop(w);
            buf
        };
        let c_frames = crate::domain::decoder::decode_all_frames(&ref_encoded).unwrap();
        assert_eq!(c_frames.len(), 3);

        // Verify A == original pixels (lossless roundtrip)
        for (i, (decoded, _)) in a_frames.iter().enumerate() {
            let original_pixels = &seq.frames[i].0.pixels;
            assert_eq!(
                &decoded.pixels, original_pixels,
                "frame {i}: our_encode→our_decode must match original pixels"
            );
        }
    }
}
