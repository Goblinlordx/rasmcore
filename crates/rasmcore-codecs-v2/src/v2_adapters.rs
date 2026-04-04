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
}
