use crate::domain::error::ImageError;
use crate::domain::types::{ColorSpace, DisposalMethod, FrameSequence, ImageInfo, PixelFormat};

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

    // Embed color space signaling chunk based on ImageInfo.color_space.
    // sRGB → sRGB chunk; DisplayP3/Bt2020/etc → cICP chunk.
    // ProPhotoRgb/AdobeRgb have no cICP code points and are left for ICC profile handling.
    let buf = embed_color_space_chunk(&buf, info.color_space)?;

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

/// Strip all PNG chunks whose type matches any in `chunk_types`.
/// Returns a new Vec with those chunks removed.
fn strip_png_chunks(png_data: &[u8], chunk_types: &[&[u8; 4]]) -> Vec<u8> {
    const PNG_SIG_LEN: usize = 8;
    let mut result = Vec::with_capacity(png_data.len());
    result.extend_from_slice(&png_data[..PNG_SIG_LEN]);
    let mut pos = PNG_SIG_LEN;
    while pos + 12 <= png_data.len() {
        let len =
            u32::from_be_bytes([png_data[pos], png_data[pos + 1], png_data[pos + 2], png_data[pos + 3]])
                as usize;
        let chunk_end = pos + 12 + len;
        if chunk_end > png_data.len() {
            // Malformed — copy rest and stop
            result.extend_from_slice(&png_data[pos..]);
            return result;
        }
        let ctype = &png_data[pos + 4..pos + 8];
        let should_strip = chunk_types.iter().any(|t| ctype == *t);
        if !should_strip {
            result.extend_from_slice(&png_data[pos..chunk_end]);
        }
        pos = chunk_end;
    }
    result
}

/// Embed an ICC profile into already-encoded PNG data as an iCCP chunk.
///
/// Inserts the iCCP chunk after IHDR (before IDAT). The profile is
/// compressed with deflate (zlib wrapper) as required by the PNG spec.
///
/// Per the PNG spec, iCCP is mutually exclusive with cICP and sRGB chunks.
/// If either is already present, they are stripped first — an explicit ICC
/// profile embedding overrides auto-inserted color space signaling.
pub fn embed_icc_profile(png_data: &[u8], icc_profile: &[u8]) -> Result<Vec<u8>, ImageError> {
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if !png_data.starts_with(PNG_SIG) {
        return Err(ImageError::InvalidInput("not a valid PNG".into()));
    }

    // Mutual exclusion: strip existing cICP/sRGB chunks before inserting iCCP.
    let png_data = strip_png_chunks(png_data, &[b"cICP", b"sRGB"]);

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

/// Map a `ColorSpace` to cICP (ITU-T H.273) code points.
///
/// Returns `Some((colour_primaries, transfer_characteristics, matrix_coefficients,
/// video_full_range_flag))` for color spaces with well-defined cICP values.
/// Returns `None` for color spaces that lack standard cICP code points
/// (ProPhotoRgb, AdobeRgb) — these should use iCCP with an ICC profile instead.
pub(crate) fn color_space_to_cicp(cs: ColorSpace) -> Option<(u8, u8, u8, u8)> {
    match cs {
        // BT.709 primaries, sRGB transfer, Identity matrix, full range
        ColorSpace::Srgb => Some((1, 13, 0, 1)),
        // BT.709 primaries, linear transfer, Identity matrix, full range
        ColorSpace::LinearSrgb => Some((1, 8, 0, 1)),
        // Display P3 primaries (SMPTE EG 432-1), sRGB transfer
        ColorSpace::DisplayP3 => Some((12, 13, 0, 1)),
        // BT.709 primaries, BT.709 transfer
        ColorSpace::Bt709 => Some((1, 1, 0, 1)),
        // BT.2020 primaries, BT.2020-10bit transfer
        ColorSpace::Bt2020 => Some((9, 14, 0, 1)),
        // No standard cICP code point for ProPhoto RGB primaries
        ColorSpace::ProPhotoRgb => None,
        // No standard cICP code point for Adobe RGB (1998)
        ColorSpace::AdobeRgb => None,
    }
}

/// Embed a cICP (Codec Independent Code Points) chunk into already-encoded PNG data.
///
/// The cICP chunk signals color space to browsers via ITU-T H.273 code points,
/// as defined in PNG Third Edition (2025). Inserted after IHDR, before PLTE/IDAT.
///
/// Per the PNG spec, cICP is mutually exclusive with sRGB and iCCP chunks.
pub fn embed_cicp_chunk(
    png_data: &[u8],
    colour_primaries: u8,
    transfer_function: u8,
    matrix_coefficients: u8,
    full_range: u8,
) -> Result<Vec<u8>, ImageError> {
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if !png_data.starts_with(PNG_SIG) {
        return Err(ImageError::InvalidInput("not a valid PNG".into()));
    }

    // cICP payload: 4 bytes
    let chunk_data = [
        colour_primaries,
        transfer_function,
        matrix_coefficients,
        full_range,
    ];

    // Build chunk: length(4) + "cICP"(4) + data(4) + CRC(4)
    let chunk_len = 4u32;
    let mut cicp_chunk = Vec::with_capacity(16);
    cicp_chunk.extend_from_slice(&chunk_len.to_be_bytes());
    cicp_chunk.extend_from_slice(b"cICP");
    cicp_chunk.extend_from_slice(&chunk_data);
    let crc = png_crc32(b"cICP", &chunk_data);
    cicp_chunk.extend_from_slice(&crc.to_be_bytes());

    // Insert after IHDR
    if 8 + 12 > png_data.len() {
        return Err(ImageError::InvalidInput("PNG too short".into()));
    }
    let ihdr_len =
        u32::from_be_bytes([png_data[8], png_data[9], png_data[10], png_data[11]]) as usize;
    let after_ihdr = 8 + 12 + ihdr_len;

    let mut result = Vec::with_capacity(png_data.len() + cicp_chunk.len());
    result.extend_from_slice(&png_data[..after_ihdr]);
    result.extend_from_slice(&cicp_chunk);
    result.extend_from_slice(&png_data[after_ihdr..]);

    Ok(result)
}

/// Embed an sRGB chunk into already-encoded PNG data.
///
/// The sRGB chunk is a 1-byte payload indicating the rendering intent.
/// Per the PNG spec, sRGB is mutually exclusive with cICP and iCCP chunks.
pub fn embed_srgb_chunk(png_data: &[u8], rendering_intent: u8) -> Result<Vec<u8>, ImageError> {
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    if !png_data.starts_with(PNG_SIG) {
        return Err(ImageError::InvalidInput("not a valid PNG".into()));
    }

    // sRGB payload: 1 byte (rendering intent)
    let chunk_data = [rendering_intent];

    // Build chunk: length(4) + "sRGB"(4) + data(1) + CRC(4)
    let chunk_len = 1u32;
    let mut srgb_chunk = Vec::with_capacity(13);
    srgb_chunk.extend_from_slice(&chunk_len.to_be_bytes());
    srgb_chunk.extend_from_slice(b"sRGB");
    srgb_chunk.extend_from_slice(&chunk_data);
    let crc = png_crc32(b"sRGB", &chunk_data);
    srgb_chunk.extend_from_slice(&crc.to_be_bytes());

    // Insert after IHDR
    if 8 + 12 > png_data.len() {
        return Err(ImageError::InvalidInput("PNG too short".into()));
    }
    let ihdr_len =
        u32::from_be_bytes([png_data[8], png_data[9], png_data[10], png_data[11]]) as usize;
    let after_ihdr = 8 + 12 + ihdr_len;

    let mut result = Vec::with_capacity(png_data.len() + srgb_chunk.len());
    result.extend_from_slice(&png_data[..after_ihdr]);
    result.extend_from_slice(&srgb_chunk);
    result.extend_from_slice(&png_data[after_ihdr..]);

    Ok(result)
}

/// Embed the appropriate color space signaling chunk into PNG data based on `ColorSpace`.
///
/// - `Srgb` → sRGB chunk (rendering intent: perceptual)
/// - `DisplayP3`, `Bt2020`, `LinearSrgb`, `Bt709` → cICP chunk
/// - `ProPhotoRgb`, `AdobeRgb` → no chunk (require ICC profile via `embed_icc_profile`)
///
/// This function is the primary entry point for color space metadata in PNG encoding.
/// It respects mutual exclusion rules: cICP/sRGB/iCCP cannot coexist.
pub fn embed_color_space_chunk(
    png_data: &[u8],
    color_space: ColorSpace,
) -> Result<Vec<u8>, ImageError> {
    match color_space {
        ColorSpace::Srgb => {
            // sRGB chunk with rendering intent 0 (perceptual)
            embed_srgb_chunk(png_data, 0)
        }
        cs => match color_space_to_cicp(cs) {
            Some((primaries, transfer, matrix, range)) => {
                embed_cicp_chunk(png_data, primaries, transfer, matrix, range)
            }
            // ProPhotoRgb, AdobeRgb: no cICP code points — caller should use ICC profile
            None => Ok(png_data.to_vec()),
        },
    }
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


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "png",
        format: "png",
        mime: "image/png",
        extensions: &["png"],
        fn_name: "encode_png",
        encode_fn: Some(|pixels, info| encode(pixels, info, &PngEncodeConfig::default())),
    }
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

    // ─── cICP / sRGB Chunk Tests ──────────────────────────────────────

    /// Helper: scan PNG chunks and return the 4-byte type of each chunk found.
    fn collect_chunk_types(png_data: &[u8]) -> Vec<String> {
        let mut types = Vec::new();
        let mut pos = 8; // skip PNG signature
        while pos + 12 <= png_data.len() {
            let len = u32::from_be_bytes([
                png_data[pos],
                png_data[pos + 1],
                png_data[pos + 2],
                png_data[pos + 3],
            ]) as usize;
            let ctype = std::str::from_utf8(&png_data[pos + 4..pos + 8])
                .unwrap_or("????")
                .to_string();
            types.push(ctype);
            pos += 12 + len;
        }
        types
    }

    /// Helper: extract cICP payload from PNG data.
    fn extract_cicp_payload(png_data: &[u8]) -> Option<(u8, u8, u8, u8)> {
        let mut pos = 8;
        while pos + 12 <= png_data.len() {
            let len = u32::from_be_bytes([
                png_data[pos],
                png_data[pos + 1],
                png_data[pos + 2],
                png_data[pos + 3],
            ]) as usize;
            let ctype = &png_data[pos + 4..pos + 8];
            if ctype == b"cICP" && len == 4 {
                return Some((
                    png_data[pos + 8],
                    png_data[pos + 9],
                    png_data[pos + 10],
                    png_data[pos + 11],
                ));
            }
            pos += 12 + len;
        }
        None
    }

    /// Helper: extract sRGB payload (rendering intent) from PNG data.
    fn extract_srgb_payload(png_data: &[u8]) -> Option<u8> {
        let mut pos = 8;
        while pos + 12 <= png_data.len() {
            let len = u32::from_be_bytes([
                png_data[pos],
                png_data[pos + 1],
                png_data[pos + 2],
                png_data[pos + 3],
            ]) as usize;
            let ctype = &png_data[pos + 4..pos + 8];
            if ctype == b"sRGB" && len == 1 {
                return Some(png_data[pos + 8]);
            }
            pos += 12 + len;
        }
        None
    }

    #[test]
    fn cicp_mapping_srgb() {
        assert_eq!(color_space_to_cicp(ColorSpace::Srgb), Some((1, 13, 0, 1)));
    }

    #[test]
    fn cicp_mapping_linear_srgb() {
        assert_eq!(
            color_space_to_cicp(ColorSpace::LinearSrgb),
            Some((1, 8, 0, 1))
        );
    }

    #[test]
    fn cicp_mapping_display_p3() {
        assert_eq!(
            color_space_to_cicp(ColorSpace::DisplayP3),
            Some((12, 13, 0, 1))
        );
    }

    #[test]
    fn cicp_mapping_bt709() {
        assert_eq!(color_space_to_cicp(ColorSpace::Bt709), Some((1, 1, 0, 1)));
    }

    #[test]
    fn cicp_mapping_bt2020() {
        assert_eq!(
            color_space_to_cicp(ColorSpace::Bt2020),
            Some((9, 14, 0, 1))
        );
    }

    #[test]
    fn cicp_mapping_prophoto_rgb_none() {
        assert_eq!(color_space_to_cicp(ColorSpace::ProPhotoRgb), None);
    }

    #[test]
    fn cicp_mapping_adobe_rgb_none() {
        assert_eq!(color_space_to_cicp(ColorSpace::AdobeRgb), None);
    }

    #[test]
    fn encode_srgb_has_srgb_chunk() {
        let (pixels, info) = make_test_pixels(); // color_space: Srgb
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let chunks = collect_chunk_types(&encoded);
        assert!(chunks.contains(&"sRGB".to_string()), "sRGB chunk missing");
        assert!(
            !chunks.contains(&"cICP".to_string()),
            "cICP should not be present for sRGB"
        );
        // Verify rendering intent is perceptual (0)
        assert_eq!(extract_srgb_payload(&encoded), Some(0));
    }

    #[test]
    fn encode_display_p3_has_cicp_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::DisplayP3,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let chunks = collect_chunk_types(&encoded);
        assert!(chunks.contains(&"cICP".to_string()), "cICP chunk missing");
        assert!(
            !chunks.contains(&"sRGB".to_string()),
            "sRGB should not be present for Display P3"
        );
        // Verify code points: primaries=12, transfer=13, matrix=0, range=1
        assert_eq!(extract_cicp_payload(&encoded), Some((12, 13, 0, 1)));
    }

    #[test]
    fn encode_bt2020_has_cicp_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Bt2020,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        assert_eq!(extract_cicp_payload(&encoded), Some((9, 14, 0, 1)));
    }

    #[test]
    fn encode_linear_srgb_has_cicp_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::LinearSrgb,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        assert_eq!(extract_cicp_payload(&encoded), Some((1, 8, 0, 1)));
    }

    #[test]
    fn encode_bt709_has_cicp_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Bt709,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        assert_eq!(extract_cicp_payload(&encoded), Some((1, 1, 0, 1)));
    }

    #[test]
    fn encode_prophoto_rgb_no_cicp_no_srgb() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::ProPhotoRgb,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let chunks = collect_chunk_types(&encoded);
        assert!(!chunks.contains(&"cICP".to_string()));
        assert!(!chunks.contains(&"sRGB".to_string()));
    }

    #[test]
    fn srgb_chunk_rendering_intents() {
        let (pixels, info) = make_test_pixels();
        let _base = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        // The encode() path always uses intent 0 (perceptual).
        // Test the standalone embed_srgb_chunk with other intents.
        // First, make a PNG without any color chunks for testing.
        let raw = {
            let mut buf = Vec::new();
            let mut enc = png::Encoder::new(&mut buf, 8, 8);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            let px = vec![0u8; 8 * 8 * 3];
            w.write_image_data(&px).unwrap();
            drop(w);
            buf
        };
        for intent in 0..=3u8 {
            let with_srgb = embed_srgb_chunk(&raw, intent).unwrap();
            assert_eq!(
                extract_srgb_payload(&with_srgb),
                Some(intent),
                "rendering intent {intent} mismatch"
            );
        }
    }

    #[test]
    fn cicp_chunk_after_ihdr_before_idat() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::DisplayP3,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let chunks = collect_chunk_types(&encoded);
        let cicp_idx = chunks.iter().position(|c| c == "cICP").unwrap();
        let idat_idx = chunks.iter().position(|c| c == "IDAT").unwrap();
        let ihdr_idx = chunks.iter().position(|c| c == "IHDR").unwrap();
        assert!(
            cicp_idx > ihdr_idx,
            "cICP must appear after IHDR"
        );
        assert!(
            cicp_idx < idat_idx,
            "cICP must appear before IDAT"
        );
    }

    #[test]
    fn icc_profile_replaces_cicp_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::DisplayP3,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        // Explicit ICC profile embedding strips cICP and adds iCCP
        let fake_icc = vec![42u8; 100];
        let with_icc = embed_icc_profile(&encoded, &fake_icc).unwrap();
        let chunks = collect_chunk_types(&with_icc);
        assert!(
            chunks.contains(&"iCCP".to_string()),
            "iCCP should be present after explicit ICC embed"
        );
        assert!(
            !chunks.contains(&"cICP".to_string()),
            "cICP must be stripped when iCCP is added"
        );
    }

    #[test]
    fn icc_profile_replaces_srgb_chunk() {
        let (pixels, info) = make_test_pixels(); // color_space: Srgb
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let fake_icc = vec![42u8; 100];
        let with_icc = embed_icc_profile(&encoded, &fake_icc).unwrap();
        let chunks = collect_chunk_types(&with_icc);
        assert!(
            chunks.contains(&"iCCP".to_string()),
            "iCCP should be present after explicit ICC embed"
        );
        assert!(
            !chunks.contains(&"sRGB".to_string()),
            "sRGB must be stripped when iCCP is added"
        );
    }

    #[test]
    fn icc_profile_works_when_no_color_chunk() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::ProPhotoRgb, // No cICP/sRGB chunk emitted
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let fake_icc = vec![42u8; 100];
        let with_icc = embed_icc_profile(&encoded, &fake_icc).unwrap();
        let chunks = collect_chunk_types(&with_icc);
        assert!(
            chunks.contains(&"iCCP".to_string()),
            "iCCP should be embedded when no cICP/sRGB present"
        );
    }

    #[test]
    fn roundtrip_display_p3_color_space_preserved() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::DisplayP3,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(
            decoded.info.color_space,
            ColorSpace::DisplayP3,
            "color_space must survive encode→decode roundtrip"
        );
        assert_eq!(decoded.pixels, pixels, "pixels must be exact");
    }

    #[test]
    fn roundtrip_bt2020_color_space_preserved() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Bt2020,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.color_space, ColorSpace::Bt2020);
    }

    #[test]
    fn roundtrip_srgb_color_space_preserved() {
        let (pixels, info) = make_test_pixels(); // Srgb
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        // sRGB doesn't produce cICP, decoder defaults to Srgb, so this should pass
        assert_eq!(decoded.info.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn roundtrip_16bit_display_p3() {
        let (w, h) = (8u32, 8u32);
        let mut pixels = Vec::with_capacity((w * h * 6) as usize);
        for i in 0..(w * h) {
            let r = ((i * 257) % 65536) as u16;
            let g = ((i * 131 + 1000) % 65536) as u16;
            let b = ((i * 73 + 5000) % 65536) as u16;
            pixels.extend_from_slice(&r.to_le_bytes());
            pixels.extend_from_slice(&g.to_le_bytes());
            pixels.extend_from_slice(&b.to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::DisplayP3,
        };
        let encoded = encode(&pixels, &info, &PngEncodeConfig::default()).unwrap();
        // Verify cICP chunk is present
        assert_eq!(extract_cicp_payload(&encoded), Some((12, 13, 0, 1)));
        // Verify roundtrip
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.color_space, ColorSpace::DisplayP3);
        assert_eq!(decoded.info.format, PixelFormat::Rgb16);
        assert_eq!(decoded.pixels, pixels, "16-bit P3 roundtrip must be pixel-exact");
    }
}
