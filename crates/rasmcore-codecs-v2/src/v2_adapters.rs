//! V2 codec adapters — register encoders/decoders via the V2 registry.
//!
//! Each adapter bridges between the f32 pipeline and a pure codec implementation.
//! The codec implementation is independent of the pipeline. The adapter uses
//! pipeline conversion helpers to convert between f32 and native pixel formats.
//!
//! The adapter is what gets registered via inventory. The codec itself has no
//! pipeline dependency.

use rasmcore_pipeline_v2::color_math::{
    f32_linear_to_srgb_rgba8,
    srgb_rgba8_to_f32_linear, srgb_rgb8_to_f32_linear,
};
use rasmcore_pipeline_v2::node::PipelineError;
use rasmcore_pipeline_v2::registry::{
    DecoderFactoryRegistration, DecodedImageV2, EncoderFactoryRegistration,
    ParamDescriptor, ParamMap, ParamType,
};

// ═══════════════════════════════════════════════════════════════════════════════
// PNG Encoder Adapter
// ═══════════════════════════════════════════════════════════════════════════════

static PNG_ENCODER_PARAMS: [ParamDescriptor; 1] = [ParamDescriptor {
    name: "compression_level",
    value_type: ParamType::U32,
    min: Some(0.0),
    max: Some(9.0),
    step: Some(1.0),
    default: Some(6.0),
    hint: Some("slider"),
    constraints: &[],
}];

fn png_encode(
    pixels: &[f32],
    width: u32,
    height: u32,
    _params: &ParamMap,
) -> Result<Vec<u8>, PipelineError> {
    // Layer 2: conversion (pipeline helper)
    let rgba8 = f32_linear_to_srgb_rgba8(pixels);

    // Layer 1: pure codec (png crate, independent of pipeline)
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|e| PipelineError::ComputeError(format!("png encode header: {e}")))?;
        writer
            .write_image_data(&rgba8)
            .map_err(|e| PipelineError::ComputeError(format!("png encode data: {e}")))?;
    }
    Ok(buf)
}

inventory::submit! {
    &EncoderFactoryRegistration {
        name: "png",
        display_name: "PNG",
        mime: "image/png",
        extensions: &["png"],
        params: &PNG_ENCODER_PARAMS,
        encode: png_encode,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PNG Decoder Adapter
// ═══════════════════════════════════════════════════════════════════════════════

fn png_detect(data: &[u8]) -> bool {
    data.len() >= 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
}

fn png_decode(data: &[u8]) -> Result<DecodedImageV2, PipelineError> {
    // Layer 1: pure codec (png crate)
    let decoder = png::Decoder::new(data);
    let mut reader = decoder
        .read_info()
        .map_err(|e| PipelineError::ComputeError(format!("png decode: {e}")))?;
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let output_info = reader
        .next_frame(&mut raw)
        .map_err(|e| PipelineError::ComputeError(format!("png decode frame: {e}")))?;
    raw.truncate(output_info.buffer_size());

    let width = output_info.width;
    let height = output_info.height;

    // Layer 2: conversion (pipeline helper)
    let pixels = match output_info.color_type {
        png::ColorType::Rgba => srgb_rgba8_to_f32_linear(&raw),
        png::ColorType::Rgb => srgb_rgb8_to_f32_linear(&raw),
        png::ColorType::Grayscale => {
            // Gray → RGBA
            let mut out = Vec::with_capacity(raw.len() * 4);
            for &g in &raw {
                let v = rasmcore_pipeline_v2::color_math::srgb_to_linear(g as f32 / 255.0);
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(1.0);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(raw.len() / 2 * 4);
            for chunk in raw.chunks_exact(2) {
                let v = rasmcore_pipeline_v2::color_math::srgb_to_linear(chunk[0] as f32 / 255.0);
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(chunk[1] as f32 / 255.0);
            }
            out
        }
        _ => {
            return Err(PipelineError::ComputeError(format!(
                "png: unsupported color type {:?}",
                output_info.color_type
            )));
        }
    };

    Ok(DecodedImageV2 {
        pixels,
        width,
        height,
        color_space: rasmcore_pipeline_v2::color_space::ColorSpace::Linear,
    })
}

inventory::submit! {
    &DecoderFactoryRegistration {
        name: "png",
        display_name: "PNG",
        extensions: &["png"],
        detect: png_detect,
        decode: png_decode,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_via_v2_registry() {
        // Create a 2x2 solid gray image in f32 linear
        let pixels = vec![
            0.5f32, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
        ]; // 2x2 pixels
        let params = ParamMap::new();

        // Encode via registry
        let encoded = rasmcore_pipeline_v2::encode_via_registry("png", &pixels, 2, 2, &params)
            .expect("PNG encoder not found")
            .expect("PNG encode failed");

        assert!(!encoded.is_empty());
        assert!(png_detect(&encoded));

        // Decode via registry
        let decoded = rasmcore_pipeline_v2::decode_via_registry(&encoded)
            .expect("PNG decoder not found")
            .expect("PNG decode failed");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.pixels.len(), 2 * 2 * 4);

        // Round-trip should be close (u8 quantization limits precision)
        for i in 0..3 {
            assert!(
                (decoded.pixels[i] - pixels[i]).abs() < 0.01,
                "ch{i}: {:.4} vs {:.4}",
                decoded.pixels[i],
                pixels[i]
            );
        }
    }

    #[test]
    fn png_encoder_in_registry() {
        let encoders = rasmcore_pipeline_v2::registered_encoders();
        assert!(
            encoders.iter().any(|e| e.name == "png"),
            "PNG encoder not found in registry"
        );
    }

    #[test]
    fn png_decoder_in_registry() {
        let decoders = rasmcore_pipeline_v2::registered_decoders();
        assert!(
            decoders.iter().any(|d| d.name == "png"),
            "PNG decoder not found in registry"
        );
    }
}
