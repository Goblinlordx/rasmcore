//! V2 decoders — unified f32 RGBA output for all image formats.
//!
//! Each decoder wraps the V1 decoder (in rasmcore-image), converts its output
//! to f32 RGBA, and registers via `OperationRegistration`.
//!
//! sRGB formats (JPEG, PNG, etc.) output f32 in sRGB color space.
//! Linear formats (EXR, HDR, FITS) output f32 in Linear color space.
//! The pipeline's promote/linearize node handles sRGB→Linear conversion.

use crate::convert;
use rasmcore_pipeline_v2::ops::{DecodedImage, Decoder};
use rasmcore_pipeline_v2::registry::{
    OperationCapabilities, OperationKind, OperationRegistration,
};
use rasmcore_pipeline_v2::PipelineError;

// ─── Macro for sRGB decoders (u8 native, converted to f32) ──────────────────

macro_rules! v2_decoder {
    (
        struct_name: $struct:ident,
        format: $format:literal,
        display_name: $display:literal,
        extensions: [$($ext:literal),+],
        detect: |$data:ident| $detect:expr,
        reg_name: $reg_name:ident
    ) => {
        pub struct $struct;

        impl Decoder for $struct {
            fn decode(&self, data: &[u8]) -> Result<DecodedImage, PipelineError> {
                let v1 = rasmcore_image::domain::decoder::decode_with_hint(data, Some($format))
                    .map_err(|e| PipelineError::ComputeError(format!("{}: {e}", $format)))?;
                convert::v1_to_v2(v1)
            }

            fn can_decode(&self, $data: &[u8]) -> bool {
                $detect
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }

        inventory::submit! {
            &OperationRegistration {
                name: concat!($format, "_decode"),
                display_name: concat!($display, " Decoder"),
                category: "codec",
                kind: OperationKind::Decoder,
                params: &[],
                capabilities: OperationCapabilities {
                    gpu: false, analytic: false, affine: false, clut: false,
                },
                doc_path: "",
                cost: "",
            }
        }
    };
}

// ─── Macro for f32-native decoders (EXR, HDR, FITS) ─────────────────────────

macro_rules! v2_f32_decoder {
    (
        struct_name: $struct:ident,
        format: $format:literal,
        display_name: $display:literal,
        extensions: [$($ext:literal),+],
        detect: |$data:ident| $detect:expr,
        reg_name: $reg_name:ident
    ) => {
        pub struct $struct;

        impl Decoder for $struct {
            fn decode(&self, data: &[u8]) -> Result<DecodedImage, PipelineError> {
                let v1 = rasmcore_image::domain::decoder::decode_f32_with_hint(data, Some($format))
                    .map_err(|e| PipelineError::ComputeError(format!("{}: {e}", $format)))?;
                let mut decoded = convert::v1_to_v2(v1)?;
                // Override: f32-native formats are always linear regardless of V1 metadata
                decoded.info.color_space = rasmcore_pipeline_v2::ColorSpace::Linear;
                Ok(decoded)
            }

            fn can_decode(&self, $data: &[u8]) -> bool {
                $detect
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }

        inventory::submit! {
            &OperationRegistration {
                name: concat!($format, "_decode"),
                display_name: concat!($display, " Decoder"),
                category: "codec",
                kind: OperationKind::Decoder,
                params: &[],
                capabilities: OperationCapabilities {
                    gpu: false, analytic: false, affine: false, clut: false,
                },
                doc_path: "",
                cost: "",
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB decoders — u8/u16 native, output f32 in sRGB color space
// ═══════════════════════════════════════════════════════════════════════════════

v2_decoder! {
    struct_name: PngDecoder,
    format: "png",
    display_name: "PNG",
    extensions: ["png"],
    detect: |d| d.len() >= 4 && d[..4] == [0x89, 0x50, 0x4E, 0x47],
    reg_name: PNG_DECODE_REG
}

v2_decoder! {
    struct_name: JpegDecoder,
    format: "jpeg",
    display_name: "JPEG",
    extensions: ["jpg", "jpeg", "jfif"],
    detect: |d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8,
    reg_name: JPEG_DECODE_REG
}

v2_decoder! {
    struct_name: WebpDecoder,
    format: "webp",
    display_name: "WebP",
    extensions: ["webp"],
    detect: |d| d.len() >= 12 && &d[..4] == b"RIFF" && &d[8..12] == b"WEBP",
    reg_name: WEBP_DECODE_REG
}

v2_decoder! {
    struct_name: GifDecoder,
    format: "gif",
    display_name: "GIF",
    extensions: ["gif"],
    detect: |d| d.len() >= 3 && &d[..3] == b"GIF",
    reg_name: GIF_DECODE_REG
}

v2_decoder! {
    struct_name: BmpDecoder,
    format: "bmp",
    display_name: "BMP",
    extensions: ["bmp", "dib"],
    detect: |d| d.len() >= 2 && &d[..2] == b"BM",
    reg_name: BMP_DECODE_REG
}

v2_decoder! {
    struct_name: QoiDecoder,
    format: "qoi",
    display_name: "QOI",
    extensions: ["qoi"],
    detect: |d| d.len() >= 4 && &d[..4] == b"qoif",
    reg_name: QOI_DECODE_REG
}

v2_decoder! {
    struct_name: IcoDecoder,
    format: "ico",
    display_name: "ICO",
    extensions: ["ico", "cur"],
    detect: |d| {
        d.len() >= 6
            && d[0] == 0 && d[1] == 0
            && (d[2] == 1 || d[2] == 2) && d[3] == 0
            && {
                let count = u16::from_le_bytes([d[4], d[5]]);
                count > 0 && count <= 256 && d.len() >= 6 + count as usize * 16
            }
    },
    reg_name: ICO_DECODE_REG
}

v2_decoder! {
    struct_name: TgaDecoder,
    format: "tga",
    display_name: "TGA",
    extensions: ["tga", "targa"],
    detect: |d| {
        // TGA has no magic bytes — use same heuristic as V1
        if d.len() < 18 { return false; }
        let img_type = d[2];
        matches!(img_type, 1 | 2 | 3 | 9 | 10 | 11)
    },
    reg_name: TGA_DECODE_REG
}

v2_decoder! {
    struct_name: TiffDecoder,
    format: "tiff",
    display_name: "TIFF",
    extensions: ["tiff", "tif"],
    detect: |d| {
        d.len() >= 4
            && ((d[0] == b'I' && d[1] == b'I' && d[2] == 42 && d[3] == 0)
                || (d[0] == b'M' && d[1] == b'M' && d[2] == 0 && d[3] == 42))
    },
    reg_name: TIFF_DECODE_REG
}

v2_decoder! {
    struct_name: DdsDecoder,
    format: "dds",
    display_name: "DDS",
    extensions: ["dds"],
    detect: |d| d.len() >= 4 && d[..4] == [0x44, 0x44, 0x53, 0x20],
    reg_name: DDS_DECODE_REG
}

v2_decoder! {
    struct_name: PnmDecoder,
    format: "pnm",
    display_name: "PNM",
    extensions: ["pnm", "ppm", "pgm", "pbm", "pam"],
    detect: |d| d.len() >= 2 && d[0] == b'P' && d[1].is_ascii_digit(),
    reg_name: PNM_DECODE_REG
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear f32-native decoders — output f32 in Linear color space
// ═══════════════════════════════════════════════════════════════════════════════

v2_f32_decoder! {
    struct_name: ExrDecoder,
    format: "exr",
    display_name: "OpenEXR",
    extensions: ["exr"],
    detect: |d| d.len() >= 4 && d[..4] == [0x76, 0x2F, 0x31, 0x01],
    reg_name: EXR_DECODE_REG
}

v2_f32_decoder! {
    struct_name: HdrDecoder,
    format: "hdr",
    display_name: "Radiance HDR",
    extensions: ["hdr", "rgbe"],
    detect: |d| d.len() >= 10 && (d.starts_with(b"#?RADIANCE") || d.starts_with(b"#?RGBE")),
    reg_name: HDR_DECODE_REG
}

v2_f32_decoder! {
    struct_name: FitsDecoder,
    format: "fits",
    display_name: "FITS",
    extensions: ["fits", "fit"],
    detect: |d| d.len() >= 6 && d.starts_with(b"SIMPLE"),
    reg_name: FITS_DECODE_REG
}

// ─── V2 codec dispatch ──────────────────────────────────────────────────────

/// Dispatch helper: decode data via a specific V2 decoder struct.
fn decode_via<D: Decoder>(decoder: &D, data: &[u8]) -> Result<DecodedImage, PipelineError> {
    decoder.decode(data)
}

/// Detect format and decode to V2 f32 RGBA.
///
/// Iterates all registered V2 decoders by priority order (binary magic first,
/// then structured headers, then text heuristics).
pub fn decode(data: &[u8]) -> Result<DecodedImage, PipelineError> {
    // Priority 10 — binary magic (fast, unambiguous)
    if PngDecoder.can_decode(data) { return decode_via(&PngDecoder, data); }
    if JpegDecoder.can_decode(data) { return decode_via(&JpegDecoder, data); }
    if GifDecoder.can_decode(data) { return decode_via(&GifDecoder, data); }
    if WebpDecoder.can_decode(data) { return decode_via(&WebpDecoder, data); }
    if BmpDecoder.can_decode(data) { return decode_via(&BmpDecoder, data); }
    if QoiDecoder.can_decode(data) { return decode_via(&QoiDecoder, data); }
    if DdsDecoder.can_decode(data) { return decode_via(&DdsDecoder, data); }
    if ExrDecoder.can_decode(data) { return decode_via(&ExrDecoder, data); }
    if HdrDecoder.can_decode(data) { return decode_via(&HdrDecoder, data); }
    // Priority 20
    if IcoDecoder.can_decode(data) { return decode_via(&IcoDecoder, data); }
    // Priority 60 — structured
    if FitsDecoder.can_decode(data) { return decode_via(&FitsDecoder, data); }
    // Priority 120 — TIFF-based
    if TiffDecoder.can_decode(data) { return decode_via(&TiffDecoder, data); }
    // Priority 200+ — text/heuristic
    if PnmDecoder.can_decode(data) { return decode_via(&PnmDecoder, data); }
    if TgaDecoder.can_decode(data) { return decode_via(&TgaDecoder, data); }

    Err(PipelineError::ComputeError(
        "no V2 decoder matched the input data".into(),
    ))
}

/// Decode with an explicit format hint.
pub fn decode_with_hint(data: &[u8], format: &str) -> Result<DecodedImage, PipelineError> {
    match format {
        "png" => decode_via(&PngDecoder, data),
        "jpeg" | "jpg" | "jfif" => decode_via(&JpegDecoder, data),
        "webp" => decode_via(&WebpDecoder, data),
        "gif" => decode_via(&GifDecoder, data),
        "bmp" | "dib" => decode_via(&BmpDecoder, data),
        "qoi" => decode_via(&QoiDecoder, data),
        "ico" | "cur" => decode_via(&IcoDecoder, data),
        "tga" | "targa" => decode_via(&TgaDecoder, data),
        "tiff" | "tif" => decode_via(&TiffDecoder, data),
        "dds" => decode_via(&DdsDecoder, data),
        "pnm" | "ppm" | "pgm" | "pbm" | "pam" => decode_via(&PnmDecoder, data),
        "exr" => decode_via(&ExrDecoder, data),
        "hdr" | "rgbe" => decode_via(&HdrDecoder, data),
        "fits" | "fit" => decode_via(&FitsDecoder, data),
        _ => Err(PipelineError::ComputeError(format!(
            "no V2 decoder for format '{format}'"
        ))),
    }
}

/// Detect image format from raw bytes. Returns format name or None.
pub fn detect_format(data: &[u8]) -> Option<&'static str> {
    if PngDecoder.can_decode(data) { return Some("png"); }
    if JpegDecoder.can_decode(data) { return Some("jpg"); }
    if GifDecoder.can_decode(data) { return Some("gif"); }
    if WebpDecoder.can_decode(data) { return Some("webp"); }
    if BmpDecoder.can_decode(data) { return Some("bmp"); }
    if QoiDecoder.can_decode(data) { return Some("qoi"); }
    if DdsDecoder.can_decode(data) { return Some("dds"); }
    if ExrDecoder.can_decode(data) { return Some("exr"); }
    if HdrDecoder.can_decode(data) { return Some("hdr"); }
    if IcoDecoder.can_decode(data) { return Some("ico"); }
    if FitsDecoder.can_decode(data) { return Some("fits"); }
    if TiffDecoder.can_decode(data) { return Some("tiff"); }
    if PnmDecoder.can_decode(data) { return Some("pnm"); }
    if TgaDecoder.can_decode(data) { return Some("tga"); }
    None
}

/// Probe metadata without decoding pixels — fast path for file browsers.
///
/// Parses only headers/metadata from the image data. For formats with
/// optimized metadata paths (JPEG, PNG, WebP), this is 10-100x faster
/// than full decode because it skips pixel decompression.
pub fn probe_metadata(data: &[u8]) -> Result<DecodedImageMetadata, PipelineError> {
    // Try format-specific fast paths first
    if JpegDecoder.can_decode(data) {
        return probe_jpeg_metadata(data);
    }
    if PngDecoder.can_decode(data) {
        return probe_png_metadata(data);
    }
    if WebpDecoder.can_decode(data) {
        return probe_webp_metadata(data);
    }

    // Fallback: full decode, discard pixels (uses default trait impl)
    if let Some(format) = detect_format(data) {
        let decoded = decode_with_hint(data, format)?;
        return Ok(DecodedImageMetadata {
            width: decoded.info.width,
            height: decoded.info.height,
            color_space: decoded.info.color_space,
            metadata: decoded.metadata,
        });
    }

    Err(PipelineError::ComputeError("no decoder matched for metadata probe".into()))
}

// ─── Fast metadata-only parsers ───────────────────────────────────────────

use rasmcore_pipeline_v2::image_metadata::ImageMetadata;
use rasmcore_pipeline_v2::ops::DecodedImageMetadata;

/// JPEG metadata-only: parse APP markers + SOF dimensions without Huffman decode.
fn probe_jpeg_metadata(data: &[u8]) -> Result<DecodedImageMetadata, PipelineError> {
    let mut metadata = ImageMetadata::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut pos = 2; // skip SOI (0xFF 0xD8)

    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        if marker == 0xD9 || marker == 0xDA { break; } // EOI or SOS — stop before pixel data
        if marker == 0x00 || marker == 0xFF {
            pos += 1;
            continue;
        }

        let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        let seg_start = pos + 4;
        let seg_end = (pos + 2 + seg_len).min(data.len());

        match marker {
            // SOF0-SOF15 (except SOF4 = DHT, SOF12 = DAC)
            0xC0..=0xCF if marker != 0xC4 && marker != 0xCC => {
                if seg_end >= seg_start + 5 {
                    height = u16::from_be_bytes([data[seg_start + 1], data[seg_start + 2]]) as u32;
                    width = u16::from_be_bytes([data[seg_start + 3], data[seg_start + 4]]) as u32;
                }
            }
            // APP1 — EXIF or XMP
            0xE1 => {
                if seg_end > seg_start + 6 && &data[seg_start..seg_start + 6] == b"Exif\0\0" {
                    metadata.exif = Some(data[seg_start + 6..seg_end].to_vec());
                } else if seg_end > seg_start + 29 && data[seg_start..].starts_with(b"http://ns.adobe.com/xap/1.0/") {
                    let xmp_start = data[seg_start..seg_end].iter().position(|&b| b == 0).unwrap_or(28) + 1;
                    metadata.xmp = Some(data[seg_start + xmp_start..seg_end].to_vec());
                }
            }
            // APP2 — ICC profile
            0xE2 => {
                if seg_end > seg_start + 14 && &data[seg_start..seg_start + 12] == b"ICC_PROFILE\0" {
                    // Simplified: take first chunk (multi-chunk ICC not handled here)
                    if metadata.icc_profile.is_none() {
                        metadata.icc_profile = Some(data[seg_start + 14..seg_end].to_vec());
                    }
                }
            }
            // APP13 — IPTC
            0xED => {
                metadata.iptc = Some(data[seg_start..seg_end].to_vec());
            }
            _ => {}
        }

        pos = pos + 2 + seg_len;
    }

    if width == 0 || height == 0 {
        return Err(PipelineError::ComputeError("JPEG: no SOF frame found".into()));
    }

    Ok(DecodedImageMetadata {
        width,
        height,
        color_space: rasmcore_pipeline_v2::ColorSpace::Srgb,
        metadata,
    })
}

/// PNG metadata-only: parse chunks before IDAT for dimensions + metadata.
fn probe_png_metadata(data: &[u8]) -> Result<DecodedImageMetadata, PipelineError> {
    let mut metadata = ImageMetadata::new();
    let mut width = 0u32;
    let mut height = 0u32;

    if data.len() < 8 { return Err(PipelineError::ComputeError("PNG too short".into())); }
    let mut pos = 8; // skip PNG signature

    while pos + 12 <= data.len() {
        let chunk_len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        let chunk_data_start = pos + 8;
        let chunk_data_end = (chunk_data_start + chunk_len).min(data.len());

        match chunk_type {
            b"IHDR" => {
                if chunk_len >= 8 {
                    width = u32::from_be_bytes([data[chunk_data_start], data[chunk_data_start + 1],
                                                data[chunk_data_start + 2], data[chunk_data_start + 3]]);
                    height = u32::from_be_bytes([data[chunk_data_start + 4], data[chunk_data_start + 5],
                                                 data[chunk_data_start + 6], data[chunk_data_start + 7]]);
                }
            }
            b"iCCP" => {
                // ICC profile (compressed) — extract raw for now
                if let Some(null_pos) = data[chunk_data_start..chunk_data_end].iter().position(|&b| b == 0) {
                    let profile_start = chunk_data_start + null_pos + 2; // skip name + compression method
                    if profile_start < chunk_data_end {
                        metadata.icc_profile = Some(data[profile_start..chunk_data_end].to_vec());
                    }
                }
            }
            b"eXIf" => {
                metadata.exif = Some(data[chunk_data_start..chunk_data_end].to_vec());
            }
            b"tEXt" | b"iTXt" => {
                if let Ok(text) = std::str::from_utf8(&data[chunk_data_start..chunk_data_end]) {
                    metadata.format_specific.push(rasmcore_pipeline_v2::image_metadata::MetadataChunk {
                        key: "text".into(),
                        value: text.as_bytes().to_vec(),
                    });
                }
            }
            b"IDAT" | b"IEND" => break, // stop before pixel data
            _ => {}
        }

        pos = chunk_data_end + 4; // skip CRC
    }

    if width == 0 || height == 0 {
        return Err(PipelineError::ComputeError("PNG: no IHDR found".into()));
    }

    Ok(DecodedImageMetadata {
        width,
        height,
        color_space: rasmcore_pipeline_v2::ColorSpace::Srgb,
        metadata,
    })
}

/// WebP metadata-only: parse RIFF chunks for dimensions + metadata.
fn probe_webp_metadata(data: &[u8]) -> Result<DecodedImageMetadata, PipelineError> {
    let mut metadata = ImageMetadata::new();
    let mut width = 0u32;
    let mut height = 0u32;

    if data.len() < 12 { return Err(PipelineError::ComputeError("WebP too short".into())); }
    let mut pos = 12; // skip RIFF + size + WEBP

    while pos + 8 <= data.len() {
        let fourcc = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let chunk_start = pos + 8;
        let chunk_end = (chunk_start + chunk_size).min(data.len());

        match fourcc {
            b"VP8 " => {
                // Lossy VP8 header — first 10 bytes contain dimensions
                if chunk_size >= 10 && chunk_end >= chunk_start + 10 {
                    let d = &data[chunk_start..];
                    // VP8 bitstream starts at byte 3 after frame tag
                    if d.len() >= 10 && d[3] == 0x9D && d[4] == 0x01 && d[5] == 0x2A {
                        width = u16::from_le_bytes([d[6], d[7]]) as u32 & 0x3FFF;
                        height = u16::from_le_bytes([d[8], d[9]]) as u32 & 0x3FFF;
                    }
                }
                break; // don't parse pixel data
            }
            b"VP8L" => {
                // Lossless VP8L header
                if chunk_size >= 5 && chunk_end >= chunk_start + 5 {
                    let d = &data[chunk_start..];
                    if d[0] == 0x2F {
                        let bits = u32::from_le_bytes([d[1], d[2], d[3], d[4]]);
                        width = (bits & 0x3FFF) + 1;
                        height = ((bits >> 14) & 0x3FFF) + 1;
                    }
                }
                break;
            }
            b"VP8X" => {
                // Extended WebP header — has canvas dimensions
                if chunk_size >= 10 && chunk_end >= chunk_start + 10 {
                    let d = &data[chunk_start..];
                    width = (u32::from_le_bytes([d[4], d[5], d[6], 0]) & 0xFFFFFF) + 1;
                    height = (u32::from_le_bytes([d[7], d[8], d[9], 0]) & 0xFFFFFF) + 1;
                }
            }
            b"ICCP" => {
                metadata.icc_profile = Some(data[chunk_start..chunk_end].to_vec());
            }
            b"EXIF" => {
                metadata.exif = Some(data[chunk_start..chunk_end].to_vec());
            }
            b"XMP " => {
                metadata.xmp = Some(data[chunk_start..chunk_end].to_vec());
            }
            _ => {}
        }

        // Chunks are padded to even size
        pos = chunk_start + ((chunk_size + 1) & !1);
    }

    if width == 0 || height == 0 {
        return Err(PipelineError::ComputeError("WebP: no dimensions found".into()));
    }

    Ok(DecodedImageMetadata {
        width,
        height,
        color_space: rasmcore_pipeline_v2::ColorSpace::Srgb,
        metadata,
    })
}

/// List all supported V2 decode formats.
pub fn supported_formats() -> Vec<&'static str> {
    vec![
        "png", "jpg", "gif", "webp", "bmp", "qoi", "dds", "exr",
        "hdr", "ico", "fits", "tiff", "pnm", "tga",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_count() {
        let fmts = supported_formats();
        assert_eq!(fmts.len(), 14);
        assert!(fmts.contains(&"png"));
        assert!(fmts.contains(&"exr"));
        assert!(fmts.contains(&"fits"));
    }

    #[test]
    fn detect_png() {
        let header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_format(&header), Some("png"));
    }

    #[test]
    fn detect_jpeg() {
        let header = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_format(&header), Some("jpg"));
    }

    #[test]
    fn detect_exr() {
        let header = [0x76, 0x2F, 0x31, 0x01, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&header), Some("exr"));
    }

    #[test]
    fn unknown_format_returns_error() {
        let garbage = [0x00, 0x01, 0x02, 0x03];
        assert!(decode(&garbage).is_err());
    }

    #[test]
    fn hint_unknown_format_returns_error() {
        assert!(decode_with_hint(&[0], "foobar").is_err());
    }
}
