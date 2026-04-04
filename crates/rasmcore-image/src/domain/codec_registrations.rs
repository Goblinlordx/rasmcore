//! Codec registrations — one `inventory::submit!` per format.
//!
//! This file registers ALL image and LUT codecs in a single place.
//! Adding a new format: add one `inventory::submit!` block here.

use crate::domain::codec::CodecRegistration;

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE CODECS — Binary magic detection (priority 10-20)
// ═══════════════════════════════════════════════════════════════════════════

inventory::submit! { &CodecRegistration {
    format: "png", extensions: &["png"], mime: "image/png",
    detect_fn: Some(|d| d.len() >= 4 && d[..4] == [0x89, 0x50, 0x4E, 0x47]),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_png),
    encode_fn: Some(|px, info, _q| {
        let config = crate::domain::encoder::png::PngEncodeConfig::default();
        crate::domain::encoder::png::encode(px, info, &config)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "jpeg", extensions: &["jpg", "jpeg", "jfif"], mime: "image/jpeg",
    detect_fn: Some(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_jpeg),
    encode_fn: Some(|px, info, q| {
        let config = crate::domain::encoder::jpeg::JpegEncodeConfig {
            quality: q.unwrap_or(85),
            progressive: false,
            turbo: false,
        };
        crate::domain::encoder::jpeg::encode_pixels(px, info, &config)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "gif", extensions: &["gif"], mime: "image/gif",
    detect_fn: Some(|d| d.len() >= 3 && &d[..3] == b"GIF"),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_gif),
    encode_fn: Some(|px, info, _q| {
        let config = crate::domain::encoder::gif::GifEncodeConfig::default();
        crate::domain::encoder::gif::encode_pixels(px, info, &config)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "webp", extensions: &["webp"], mime: "image/webp",
    detect_fn: Some(|d| d.len() >= 12 && &d[..4] == b"RIFF" && &d[8..12] == b"WEBP"),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_webp),
    encode_fn: Some(|px, info, q| {
        let config = crate::domain::encoder::webp::WebpEncodeConfig {
            quality: q.unwrap_or(75),
            lossless: false,
        };
        crate::domain::encoder::webp::encode_pixels(px, info, &config)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "bmp", extensions: &["bmp", "dib"], mime: "image/bmp",
    detect_fn: Some(|d| d.len() >= 2 && &d[..2] == b"BM"),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_bmp),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::native_trivial::encode_bmp(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "qoi", extensions: &["qoi"], mime: "image/x-qoi",
    detect_fn: Some(|d| d.len() >= 4 && &d[..4] == b"qoif"),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_qoi),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::native_trivial::encode_qoi(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "dds", extensions: &["dds"], mime: "image/vnd-ms.dds",
    detect_fn: Some(|d| d.len() >= 4 && d[..4] == [0x44, 0x44, 0x53, 0x20]),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_dds_native),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::dds::encode_dds(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "exr", extensions: &["exr"], mime: "image/x-exr",
    detect_fn: Some(|d| d.len() >= 4 && d[..4] == [0x76, 0x2F, 0x31, 0x01]),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_exr),
    encode_fn: Some(|px, info, _q| {
        crate::domain::encoder::exr::encode_pixels(px, info, &crate::domain::encoder::exr::ExrEncodeConfig)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "hdr", extensions: &["hdr", "rgbe"], mime: "image/vnd.radiance",
    detect_fn: Some(|d| d.len() >= 10 && (d.starts_with(b"#?RADIANCE") || d.starts_with(b"#?RGBE"))),
    detection_priority: 10,
    decode_fn: Some(crate::domain::decoder::decode_native_hdr),
    encode_fn: Some(|px, info, _q| {
        crate::domain::encoder::hdr::encode_pixels(px, info, &crate::domain::encoder::hdr::HdrEncodeConfig)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "ico", extensions: &["ico", "cur"], mime: "image/x-icon",
    detect_fn: Some(|d| {
        d.len() >= 6 && d[0] == 0 && d[1] == 0
            && (d[2] == 1 || d[2] == 2) && d[3] == 0
            && { let c = u16::from_le_bytes([d[4], d[5]]); c > 0 && c <= 256 && d.len() >= 6 + c as usize * 16 }
    }),
    detection_priority: 20,
    decode_fn: Some(crate::domain::decoder::decode_native_ico),
    encode_fn: Some(|px, info, _q| {
        crate::domain::encoder::ico::encode_pixels(px, info, &crate::domain::encoder::ico::IcoEncodeConfig)
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE CODECS — Structured header detection (priority 60-70)
// ═══════════════════════════════════════════════════════════════════════════

inventory::submit! { &CodecRegistration {
    format: "jxl", extensions: &["jxl"], mime: "image/jxl",
    detect_fn: Some(crate::domain::decoder::is_jxl),
    detection_priority: 60,
    decode_fn: Some(crate::domain::decoder::decode_jxl),
    encode_fn: None, // scaffold only
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "jp2", extensions: &["jp2", "j2k", "j2c", "jpeg2000"], mime: "image/jp2",
    detect_fn: Some(crate::domain::decoder::is_jp2),
    detection_priority: 60,
    decode_fn: Some(crate::domain::decoder::decode_jp2),
    encode_fn: Some(|px, info, _q| {
        crate::domain::encoder::jp2::encode(px, info, &crate::domain::encoder::jp2::Jp2EncodeConfig::default())
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "fits", extensions: &["fits", "fit"], mime: "image/fits",
    detect_fn: Some(rasmcore_fits::is_fits),
    detection_priority: 60,
    decode_fn: Some(crate::domain::decoder::decode_fits),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::fits::encode_pixels(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

#[cfg(feature = "nonfree-hevc")]
inventory::submit! { &CodecRegistration {
    format: "heic", extensions: &["heic", "heif"], mime: "image/heic",
    detect_fn: Some(crate::domain::decoder::is_heif),
    detection_priority: 65,
    decode_fn: Some(crate::domain::decoder::decode_heif),
    encode_fn: Some(|px, info, q| {
        crate::domain::encoder::heic::encode(px, info, &crate::domain::encoder::heic::HeicEncodeConfig {
            quality: q.unwrap_or(75),
        })
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

// Avif decode — detected via HEIF ftyp but uses different decoder
// Currently encode returns UnsupportedFormat (rav1e removed)
inventory::submit! { &CodecRegistration {
    format: "avif", extensions: &["avif"], mime: "image/avif",
    detect_fn: None, // detected via heic ftyp
    detection_priority: 66,
    decode_fn: None, // handled by heic decoder when ftyp is avif
    encode_fn: Some(|px, info, q| {
        crate::domain::encoder::avif::encode(px, info, &crate::domain::encoder::avif::AvifEncodeConfig {
            quality: q.unwrap_or(75),
            ..Default::default()
        })
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE CODECS — TIFF-based (priority 110-120, DNG before generic TIFF)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "native-raw")]
inventory::submit! { &CodecRegistration {
    format: "dng", extensions: &["dng"], mime: "image/x-adobe-dng",
    detect_fn: Some(|d| {
        d.len() >= 4
            && ((d[0] == b'I' && d[1] == b'I' && d[2] == 42 && d[3] == 0)
                || (d[0] == b'M' && d[1] == b'M' && d[2] == 0 && d[3] == 42))
            && rasmcore_raw::is_dng(d)
    }),
    detection_priority: 110,
    decode_fn: Some(crate::domain::decoder::decode_native_dng),
    encode_fn: None,
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "tiff", extensions: &["tiff", "tif"], mime: "image/tiff",
    detect_fn: Some(|d| {
        d.len() >= 4
            && ((d[0] == b'I' && d[1] == b'I' && d[2] == 42 && d[3] == 0)
                || (d[0] == b'M' && d[1] == b'M' && d[2] == 0 && d[3] == 42))
    }),
    detection_priority: 120,
    decode_fn: Some(crate::domain::decoder::decode_tiff_native),
    encode_fn: Some(|px, info, _q| {
        crate::domain::encoder::tiff::encode(px, info, &crate::domain::encoder::tiff::TiffEncodeConfig::default())
    }),
    decode_lut_fn: None, encode_lut_fn: None,
}}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE CODECS — Text-based / weak detection (priority 200-300)
// ═══════════════════════════════════════════════════════════════════════════

inventory::submit! { &CodecRegistration {
    format: "svg", extensions: &["svg", "svgz"], mime: "image/svg+xml",
    detect_fn: Some(crate::domain::decoder::is_svg),
    detection_priority: 200,
    decode_fn: Some(crate::domain::decoder::decode_svg),
    encode_fn: None, // SVG is decode-only
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "pnm", extensions: &["pnm", "ppm", "pgm", "pbm", "pam"], mime: "image/x-portable-anymap",
    detect_fn: Some(|d| d.len() >= 2 && d[0] == b'P' && d[1].is_ascii_digit()),
    detection_priority: 200,
    decode_fn: Some(crate::domain::decoder::decode_native_pnm),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::native_trivial::encode_pnm(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

inventory::submit! { &CodecRegistration {
    format: "tga", extensions: &["tga", "targa"], mime: "image/x-tga",
    detect_fn: Some(crate::domain::decoder::detect_tga),
    detection_priority: 300,
    decode_fn: Some(crate::domain::decoder::decode_native_tga),
    encode_fn: Some(|px, info, _q| crate::domain::encoder::native_trivial::encode_tga(px, info)),
    decode_lut_fn: None, encode_lut_fn: None,
}}

// ═══════════════════════════════════════════════════════════════════════════
// LUT CODECS
// ═══════════════════════════════════════════════════════════════════════════

inventory::submit! { &CodecRegistration {
    format: "cube", extensions: &["cube"], mime: "application/vnd.cube-lut",
    detect_fn: Some(crate::domain::decoder::is_cube_lut),
    detection_priority: 210,
    decode_fn: None, encode_fn: None,
    decode_lut_fn: Some(crate::domain::decoder::decode_cube),
    encode_lut_fn: Some(crate::domain::encoder::cube::encode),
}}

inventory::submit! { &CodecRegistration {
    format: "csp", extensions: &["csp"], mime: "application/vnd.cinespace-lut",
    detect_fn: Some(crate::domain::decoder::is_csp_lut),
    detection_priority: 210,
    decode_fn: None, encode_fn: None,
    decode_lut_fn: Some(crate::domain::decoder::decode_csp),
    encode_lut_fn: Some(crate::domain::encoder::lutcsp::encode),
}}

inventory::submit! { &CodecRegistration {
    format: "3dl", extensions: &["3dl"], mime: "application/vnd.autodesk-3dl",
    detect_fn: Some(crate::domain::decoder::is_3dl_lut),
    detection_priority: 310,
    decode_fn: None, encode_fn: None,
    decode_lut_fn: Some(crate::domain::decoder::decode_3dl),
    encode_lut_fn: Some(crate::domain::encoder::lut3dl::encode),
}}

inventory::submit! { &CodecRegistration {
    format: "hald", extensions: &["hald"], mime: "image/png",
    detect_fn: None, // Hald is a PNG — detected as PNG, requires explicit hint
    detection_priority: 400,
    decode_fn: None, encode_fn: None,
    decode_lut_fn: None, // Hald decode needs pixel data first (PNG decode then parse)
    encode_lut_fn: Some(crate::domain::encoder::hald::encode),
}}
