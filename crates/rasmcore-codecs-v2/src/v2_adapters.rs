//! V2 codec adapters — register encoders/decoders via derive macros.
//!
//! Each adapter bridges between the f32 pipeline and a pure codec.
//! The codec implementation is independent of the pipeline. The adapter uses
//! pipeline conversion helpers for f32 ↔ native format conversion.
//!
//! The derive macros (#[derive(V2Encoder)], #[derive(V2Decoder)]) generate
//! the inventory registration automatically. You can also register manually
//! via inventory::submit! if you need custom behavior.

use rasmcore_pipeline_v2::color_math::{
    f32_linear_to_srgb_rgba8,
    srgb_rgba8_to_f32_linear, srgb_rgb8_to_f32_linear,
};
use rasmcore_pipeline_v2::node::PipelineError;
use rasmcore_pipeline_v2::registry::{DecodedImageV2, ParamMap};

// ═══════════════════════════════════════════════════════════════════════════════
// PNG Encoder Adapter
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(rasmcore_macros::V2Encoder)]
#[codec(name = "png", display_name = "PNG", mime = "image/png", extensions = "png")]
pub struct PngEncoderAdapter;

impl PngEncoderAdapter {
    pub fn encode(
        pixels: &[f32],
        width: u32,
        height: u32,
        _params: &ParamMap,
    ) -> Result<Vec<u8>, PipelineError> {
        // Conversion: f32 linear → sRGB u8 RGBA
        let rgba8 = f32_linear_to_srgb_rgba8(pixels);

        // Pure codec: png crate (independent of pipeline)
        let mut buf = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut buf, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(png::Compression::Fast);
            let mut writer = encoder
                .write_header()
                .map_err(|e| PipelineError::ComputeError(format!("png encode: {e}")))?;
            writer
                .write_image_data(&rgba8)
                .map_err(|e| PipelineError::ComputeError(format!("png encode: {e}")))?;
        }
        Ok(buf)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PNG Decoder Adapter
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(rasmcore_macros::V2Decoder)]
#[codec(name = "png", display_name = "PNG", extensions = "png")]
pub struct PngDecoderAdapter;

impl PngDecoderAdapter {
    pub fn detect(data: &[u8]) -> bool {
        data.len() >= 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    }

    pub fn decode(data: &[u8]) -> Result<DecodedImageV2, PipelineError> {
        // Pure codec: png crate
        let decoder = png::Decoder::new(data);
        let mut reader = decoder
            .read_info()
            .map_err(|e| PipelineError::ComputeError(format!("png decode: {e}")))?;
        let mut raw = vec![0u8; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut raw)
            .map_err(|e| PipelineError::ComputeError(format!("png decode: {e}")))?;
        raw.truncate(info.buffer_size());

        // Conversion: sRGB u8 → f32 linear
        let pixels = match info.color_type {
            png::ColorType::Rgba => srgb_rgba8_to_f32_linear(&raw),
            png::ColorType::Rgb => srgb_rgb8_to_f32_linear(&raw),
            png::ColorType::Grayscale => {
                let mut out = Vec::with_capacity(raw.len() * 4);
                for &g in &raw {
                    let v = rasmcore_pipeline_v2::color_math::srgb_to_linear(g as f32 / 255.0);
                    out.extend_from_slice(&[v, v, v, 1.0]);
                }
                out
            }
            png::ColorType::GrayscaleAlpha => {
                let mut out = Vec::with_capacity(raw.len() / 2 * 4);
                for chunk in raw.chunks_exact(2) {
                    let v = rasmcore_pipeline_v2::color_math::srgb_to_linear(chunk[0] as f32 / 255.0);
                    out.extend_from_slice(&[v, v, v, chunk[1] as f32 / 255.0]);
                }
                out
            }
            other => {
                return Err(PipelineError::ComputeError(format!(
                    "png: unsupported color type {other:?}"
                )));
            }
        };

        Ok(DecodedImageV2 {
            pixels,
            width: info.width,
            height: info.height,
            color_space: rasmcore_pipeline_v2::color_space::ColorSpace::Linear,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: extract quality from ParamMap, delegate to V2 encoder module
// ═══════════════════════════════════════════════════════════════════════════════

fn quality_from_params(params: &ParamMap) -> Option<u8> {
    let q = params.get_u32("quality");
    if q > 0 && q <= 100 { Some(q as u8) } else { None }
}

/// Helper: convert ops::DecodedImage → registry::DecodedImageV2
fn decoded_to_v2(
    d: rasmcore_pipeline_v2::ops::DecodedImage,
) -> DecodedImageV2 {
    DecodedImageV2 {
        pixels: d.pixels,
        width: d.info.width,
        height: d.info.height,
        color_space: d.info.color_space,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Macros for bulk adapter generation
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a V2 encoder adapter that delegates to `crate::encoders::encode`.
macro_rules! v2_encoder_adapter {
    (
        adapter: $adapter:ident,
        name: $name:literal,
        display_name: $display:literal,
        mime: $mime:literal,
        extensions: $ext:literal
    ) => {
        #[derive(rasmcore_macros::V2Encoder)]
        #[codec(name = $name, display_name = $display, mime = $mime, extensions = $ext)]
        pub struct $adapter;

        impl $adapter {
            pub fn encode(
                pixels: &[f32],
                width: u32,
                height: u32,
                params: &ParamMap,
            ) -> Result<Vec<u8>, PipelineError> {
                let quality = quality_from_params(params);
                crate::encoders::encode(pixels, width, height, $name, quality)
            }
        }
    };
}

/// Generate a V2 decoder adapter that delegates to `crate::decoders::decode_with_hint`.
macro_rules! v2_decoder_adapter {
    (
        adapter: $adapter:ident,
        name: $name:literal,
        display_name: $display:literal,
        extensions: $ext:literal,
        detect: |$data:ident| $detect:expr
    ) => {
        #[derive(rasmcore_macros::V2Decoder)]
        #[codec(name = $name, display_name = $display, extensions = $ext)]
        pub struct $adapter;

        impl $adapter {
            pub fn detect($data: &[u8]) -> bool {
                $detect
            }

            pub fn decode(data: &[u8]) -> Result<DecodedImageV2, PipelineError> {
                let d = crate::decoders::decode_with_hint(data, $name)?;
                Ok(decoded_to_v2(d))
            }
        }
    };
}

/// Generate a V2 decoder adapter for linear (f32-native) formats.
/// Forces `ColorSpace::Linear` on the result.
macro_rules! v2_linear_decoder_adapter {
    (
        adapter: $adapter:ident,
        name: $name:literal,
        display_name: $display:literal,
        extensions: $ext:literal,
        detect: |$data:ident| $detect:expr
    ) => {
        #[derive(rasmcore_macros::V2Decoder)]
        #[codec(name = $name, display_name = $display, extensions = $ext)]
        pub struct $adapter;

        impl $adapter {
            pub fn detect($data: &[u8]) -> bool {
                $detect
            }

            pub fn decode(data: &[u8]) -> Result<DecodedImageV2, PipelineError> {
                let d = crate::decoders::decode_with_hint(data, $name)?;
                let mut v2 = decoded_to_v2(d);
                v2.color_space = rasmcore_pipeline_v2::color_space::ColorSpace::Linear;
                Ok(v2)
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB Encoder Adapters
// ═══════════════════════════════════════════════════════════════════════════════

v2_encoder_adapter! {
    adapter: JpegEncoderAdapter,
    name: "jpeg",
    display_name: "JPEG",
    mime: "image/jpeg",
    extensions: "jpg,jpeg,jfif"
}

v2_encoder_adapter! {
    adapter: WebpEncoderAdapter,
    name: "webp",
    display_name: "WebP",
    mime: "image/webp",
    extensions: "webp"
}

v2_encoder_adapter! {
    adapter: GifEncoderAdapter,
    name: "gif",
    display_name: "GIF",
    mime: "image/gif",
    extensions: "gif"
}

v2_encoder_adapter! {
    adapter: BmpEncoderAdapter,
    name: "bmp",
    display_name: "BMP",
    mime: "image/bmp",
    extensions: "bmp,dib"
}

v2_encoder_adapter! {
    adapter: QoiEncoderAdapter,
    name: "qoi",
    display_name: "QOI",
    mime: "image/x-qoi",
    extensions: "qoi"
}

v2_encoder_adapter! {
    adapter: IcoEncoderAdapter,
    name: "ico",
    display_name: "ICO",
    mime: "image/x-icon",
    extensions: "ico,cur"
}

v2_encoder_adapter! {
    adapter: TgaEncoderAdapter,
    name: "tga",
    display_name: "TGA",
    mime: "image/x-tga",
    extensions: "tga,targa"
}

v2_encoder_adapter! {
    adapter: PnmEncoderAdapter,
    name: "pnm",
    display_name: "PNM",
    mime: "image/x-portable-anymap",
    extensions: "pnm,ppm"
}

v2_encoder_adapter! {
    adapter: TiffEncoderAdapter,
    name: "tiff",
    display_name: "TIFF",
    mime: "image/tiff",
    extensions: "tiff,tif"
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear Encoder Adapters (f32 direct, no gamma)
// ═══════════════════════════════════════════════════════════════════════════════

v2_encoder_adapter! {
    adapter: ExrEncoderAdapter,
    name: "exr",
    display_name: "OpenEXR",
    mime: "image/x-exr",
    extensions: "exr"
}

v2_encoder_adapter! {
    adapter: HdrEncoderAdapter,
    name: "hdr",
    display_name: "Radiance HDR",
    mime: "image/vnd.radiance",
    extensions: "hdr"
}

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB Decoder Adapters
// ═══════════════════════════════════════════════════════════════════════════════

v2_decoder_adapter! {
    adapter: JpegDecoderAdapter,
    name: "jpeg",
    display_name: "JPEG",
    extensions: "jpg,jpeg,jfif",
    detect: |d| d.len() >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF
}

v2_decoder_adapter! {
    adapter: WebpDecoderAdapter,
    name: "webp",
    display_name: "WebP",
    extensions: "webp",
    detect: |d| d.len() >= 12 && &d[..4] == b"RIFF" && &d[8..12] == b"WEBP"
}

v2_decoder_adapter! {
    adapter: GifDecoderAdapter,
    name: "gif",
    display_name: "GIF",
    extensions: "gif",
    detect: |d| d.len() >= 3 && d[0] == 0x47 && d[1] == 0x49 && d[2] == 0x46
}

v2_decoder_adapter! {
    adapter: BmpDecoderAdapter,
    name: "bmp",
    display_name: "BMP",
    extensions: "bmp,dib",
    detect: |d| d.len() >= 2 && d[0] == 0x42 && d[1] == 0x4D
}

v2_decoder_adapter! {
    adapter: QoiDecoderAdapter,
    name: "qoi",
    display_name: "QOI",
    extensions: "qoi",
    detect: |d| d.len() >= 4 && &d[..4] == b"qoif"
}

v2_decoder_adapter! {
    adapter: IcoDecoderAdapter,
    name: "ico",
    display_name: "ICO",
    extensions: "ico,cur",
    detect: |d| {
        d.len() >= 4
            && d[0] == 0x00 && d[1] == 0x00
            && ((d[2] == 0x01 && d[3] == 0x00) || (d[2] == 0x02 && d[3] == 0x00))
    }
}

v2_decoder_adapter! {
    adapter: TgaDecoderAdapter,
    name: "tga",
    display_name: "TGA",
    extensions: "tga,targa",
    detect: |d| {
        // TGA has no reliable magic bytes — heuristic based on image type field
        if d.len() < 18 { return false; }
        let img_type = d[2];
        matches!(img_type, 1 | 2 | 3 | 9 | 10 | 11)
    }
}

v2_decoder_adapter! {
    adapter: TiffDecoderAdapter,
    name: "tiff",
    display_name: "TIFF",
    extensions: "tiff,tif",
    detect: |d| {
        d.len() >= 4
            && ((d[..4] == [0x49, 0x49, 0x2A, 0x00])
                || (d[..4] == [0x4D, 0x4D, 0x00, 0x2A]))
    }
}

v2_decoder_adapter! {
    adapter: DdsDecoderAdapter,
    name: "dds",
    display_name: "DDS",
    extensions: "dds",
    detect: |d| d.len() >= 4 && d[..4] == [0x44, 0x44, 0x53, 0x20]
}

v2_decoder_adapter! {
    adapter: PnmDecoderAdapter,
    name: "pnm",
    display_name: "PNM",
    extensions: "pnm,ppm,pgm,pbm",
    detect: |d| d.len() >= 2 && d[0] == b'P' && d[1].is_ascii_digit()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear Decoder Adapters (f32-native, output Linear color space)
// ═══════════════════════════════════════════════════════════════════════════════

v2_linear_decoder_adapter! {
    adapter: ExrDecoderAdapter,
    name: "exr",
    display_name: "OpenEXR",
    extensions: "exr",
    detect: |d| d.len() >= 4 && d[..4] == [0x76, 0x2F, 0x31, 0x01]
}

v2_linear_decoder_adapter! {
    adapter: HdrDecoderAdapter,
    name: "hdr",
    display_name: "Radiance HDR",
    extensions: "hdr",
    detect: |d| d.len() >= 2 && d[0] == b'#' && d[1] == b'?'
}

v2_linear_decoder_adapter! {
    adapter: FitsDecoderAdapter,
    name: "fits",
    display_name: "FITS",
    extensions: "fits,fit",
    detect: |d| d.len() >= 6 && d.starts_with(b"SIMPLE")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use rasmcore_pipeline_v2::registry::ParamMap;

    #[test]
    fn png_roundtrip_via_v2_registry() {
        let pixels = vec![
            0.5f32, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
        ];
        let params = ParamMap::new();

        let encoded = rasmcore_pipeline_v2::encode_via_registry("png", &pixels, 2, 2, &params)
            .expect("PNG encoder not found")
            .expect("PNG encode failed");

        assert!(!encoded.is_empty());

        let decoded = rasmcore_pipeline_v2::decode_via_registry(&encoded)
            .expect("PNG decoder not found")
            .expect("PNG decode failed");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);

        for i in 0..3 {
            assert!(
                (decoded.pixels[i] - pixels[i]).abs() < 0.01,
                "ch{i}: {:.4} vs {:.4}", decoded.pixels[i], pixels[i]
            );
        }
    }

    #[test]
    fn png_encoder_in_registry() {
        let encoders = rasmcore_pipeline_v2::registered_encoders();
        assert!(encoders.iter().any(|e| e.name == "png"), "PNG encoder not found");
    }

    #[test]
    fn png_decoder_in_registry() {
        let decoders = rasmcore_pipeline_v2::registered_decoders();
        assert!(decoders.iter().any(|d| d.name == "png"), "PNG decoder not found");
    }

    #[test]
    fn jpeg_encoder_in_registry() {
        let encoders = rasmcore_pipeline_v2::registered_encoders();
        assert!(encoders.iter().any(|e| e.name == "jpeg"), "JPEG encoder not found");
    }

    #[test]
    fn jpeg_decoder_in_registry() {
        let decoders = rasmcore_pipeline_v2::registered_decoders();
        assert!(decoders.iter().any(|d| d.name == "jpeg"), "JPEG decoder not found");
    }

    #[test]
    fn all_srgb_encoders_registered() {
        let encoders = rasmcore_pipeline_v2::registered_encoders();
        for name in &["png", "jpeg", "webp", "gif", "bmp", "qoi", "ico", "tga", "pnm", "tiff"] {
            assert!(
                encoders.iter().any(|e| e.name == *name),
                "encoder '{name}' not found in registry"
            );
        }
    }

    #[test]
    fn linear_encoders_registered() {
        let encoders = rasmcore_pipeline_v2::registered_encoders();
        for name in &["exr", "hdr"] {
            assert!(
                encoders.iter().any(|e| e.name == *name),
                "linear encoder '{name}' not found in registry"
            );
        }
    }

    #[test]
    fn all_decoders_registered() {
        let decoders = rasmcore_pipeline_v2::registered_decoders();
        for name in &[
            "png", "jpeg", "webp", "gif", "bmp", "qoi", "ico", "tga",
            "tiff", "dds", "pnm", "exr", "hdr", "fits",
        ] {
            assert!(
                decoders.iter().any(|d| d.name == *name),
                "decoder '{name}' not found in registry"
            );
        }
    }

    #[test]
    fn jpeg_detect_magic() {
        assert!(super::JpegDecoderAdapter::detect(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!super::JpegDecoderAdapter::detect(&[0x00, 0x00]));
    }

    #[test]
    fn webp_detect_magic() {
        let mut header = vec![0u8; 16];
        header[..4].copy_from_slice(b"RIFF");
        header[8..12].copy_from_slice(b"WEBP");
        assert!(super::WebpDecoderAdapter::detect(&header));
        assert!(!super::WebpDecoderAdapter::detect(&[0x00; 16]));
    }

    #[test]
    fn exr_detect_magic() {
        assert!(super::ExrDecoderAdapter::detect(&[0x76, 0x2F, 0x31, 0x01]));
        assert!(!super::ExrDecoderAdapter::detect(&[0x00; 4]));
    }

    #[test]
    fn tiff_detect_le_be() {
        // Little-endian TIFF
        assert!(super::TiffDecoderAdapter::detect(&[0x49, 0x49, 0x2A, 0x00]));
        // Big-endian TIFF
        assert!(super::TiffDecoderAdapter::detect(&[0x4D, 0x4D, 0x00, 0x2A]));
        assert!(!super::TiffDecoderAdapter::detect(&[0x00; 4]));
    }

    #[test]
    fn ico_detect_icon_and_cursor() {
        // ICO (type 1)
        assert!(super::IcoDecoderAdapter::detect(&[0x00, 0x00, 0x01, 0x00]));
        // CUR (type 2)
        assert!(super::IcoDecoderAdapter::detect(&[0x00, 0x00, 0x02, 0x00]));
        assert!(!super::IcoDecoderAdapter::detect(&[0x00, 0x00, 0x03, 0x00]));
    }

    #[test]
    fn fits_detect_magic() {
        assert!(super::FitsDecoderAdapter::detect(b"SIMPLE  ="));
        assert!(!super::FitsDecoderAdapter::detect(b"NOTFIT"));
    }

    #[test]
    fn hdr_detect_magic() {
        assert!(super::HdrDecoderAdapter::detect(b"#?RADIANCE\n"));
        assert!(super::HdrDecoderAdapter::detect(b"#?RGBE\n"));
        assert!(!super::HdrDecoderAdapter::detect(b"NOT_HDR"));
    }

    #[test]
    fn jpeg_encode_decode_via_v2_registry() {
        // 2x2 opaque mid-gray pixels in linear space
        let pixels = vec![
            0.5f32, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
        ];
        let params = ParamMap::new();

        let encoded = rasmcore_pipeline_v2::encode_via_registry("jpeg", &pixels, 2, 2, &params)
            .expect("JPEG encoder not found")
            .expect("JPEG encode failed");

        assert!(!encoded.is_empty());
        // Verify JPEG magic bytes
        assert_eq!(encoded[0], 0xFF);
        assert_eq!(encoded[1], 0xD8);

        // Decoder returns sRGB values (not linear). The pipeline's promote
        // node linearizes later. Here we just verify decode succeeds and
        // produces correct dimensions.
        let decoded = rasmcore_pipeline_v2::decode_via_registry(&encoded)
            .expect("JPEG decoder not found")
            .expect("JPEG decode failed");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.pixels.len(), 2 * 2 * 4);
        // All channels should be in valid range
        for v in &decoded.pixels {
            assert!(*v >= 0.0 && *v <= 1.0, "pixel value out of range: {v}");
        }
    }
}
