use crate::domain::error::ImageError;
use crate::domain::types::{DisposalMethod, FrameSequence, ImageInfo, PixelFormat};

/// GIF encode configuration.
#[derive(Debug, Clone, Default)]
pub struct GifEncodeConfig {
    /// Repeat count: 0 = infinite loop, n = repeat n times (default: 0 = infinite).
    pub repeat: u16,
}

/// Encode raw pixel data to GIF using the gif crate directly.
///
/// Quantizes the input to a 256-color palette via NeuQuant, then writes
/// a single-frame GIF89a with the specified repeat configuration.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    config: &GifEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let width = info.width as u16;
    let height = info.height as u16;

    // Convert input to RGBA8 for uniform processing
    let rgba = match info.format {
        PixelFormat::Rgba8 => pixels.to_vec(),
        PixelFormat::Rgb8 => pixels
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        PixelFormat::Gray8 => pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "GIF encode requires RGB8, RGBA8, or Gray8".into(),
            ));
        }
    };

    // Quantize to 256-color palette using NeuQuant
    let nq = color_quant::NeuQuant::new(10, 256, &rgba);
    let mut indices = Vec::with_capacity((info.width * info.height) as usize);
    let mut has_transparency = false;

    for pixel in rgba.chunks_exact(4) {
        if pixel[3] < 128 {
            has_transparency = true;
            indices.push(0u8); // transparent pixels → index 0 (will be remapped)
        } else {
            indices.push(nq.index_of(&[pixel[0], pixel[1], pixel[2], pixel[3]]) as u8);
        }
    }

    // Build palette (RGB triplets)
    let color_map = nq.color_map_rgb();

    // If transparency needed, find or allocate a transparent index
    let transparent_index = if has_transparency {
        // Use index 0 as transparent — swap its palette entry to be distinctive
        Some(0u8)
    } else {
        None
    };

    // Write GIF
    let mut buf = Vec::new();
    {
        let mut encoder = gif::Encoder::new(&mut buf, width, height, &color_map)
            .map_err(|e| ImageError::ProcessingFailed(format!("GIF encoder init: {e}")))?;

        // Set repeat behavior
        let repeat = if config.repeat == 0 {
            gif::Repeat::Infinite
        } else {
            gif::Repeat::Finite(config.repeat)
        };
        encoder
            .set_repeat(repeat)
            .map_err(|e| ImageError::ProcessingFailed(format!("GIF repeat: {e}")))?;

        let mut frame = gif::Frame {
            width,
            height,
            buffer: std::borrow::Cow::Borrowed(&indices),
            ..Default::default()
        };

        if let Some(ti) = transparent_index {
            frame.transparent = Some(ti);
        }

        encoder
            .write_frame(&frame)
            .map_err(|e| ImageError::ProcessingFailed(format!("GIF frame: {e}")))?;
    }

    Ok(buf)
}

/// Encode a FrameSequence to an animated GIF.
///
/// Each frame is independently quantized to a 256-color palette via NeuQuant.
/// Per-frame delays and disposal methods are preserved from the FrameInfo metadata.
pub fn encode_sequence(
    seq: &FrameSequence,
    config: &GifEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    if seq.is_empty() {
        return Err(ImageError::InvalidInput(
            "cannot encode empty frame sequence as GIF".into(),
        ));
    }

    let canvas_w = seq.canvas_width as u16;
    let canvas_h = seq.canvas_height as u16;

    let mut buf = Vec::new();
    {
        // Use empty global color table — each frame has its own local palette
        let mut encoder = gif::Encoder::new(&mut buf, canvas_w, canvas_h, &[])
            .map_err(|e| ImageError::ProcessingFailed(format!("GIF encoder init: {e}")))?;

        let repeat = if config.repeat == 0 {
            gif::Repeat::Infinite
        } else {
            gif::Repeat::Finite(config.repeat)
        };
        encoder
            .set_repeat(repeat)
            .map_err(|e| ImageError::ProcessingFailed(format!("GIF repeat: {e}")))?;

        for (image, frame_info) in &seq.frames {
            let w = frame_info.width as u16;
            let h = frame_info.height as u16;

            // Convert to RGBA8 for uniform NeuQuant processing
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
                        "GIF encode requires RGB8, RGBA8, or Gray8 frames".into(),
                    ));
                }
            };

            // Quantize to 256-color palette
            let nq = color_quant::NeuQuant::new(10, 256, &rgba);
            let mut indices = Vec::with_capacity((w as u32 * h as u32) as usize);
            let mut has_transparency = false;

            for pixel in rgba.chunks_exact(4) {
                if pixel[3] < 128 {
                    has_transparency = true;
                    indices.push(0u8);
                } else {
                    indices.push(nq.index_of(&[pixel[0], pixel[1], pixel[2], pixel[3]]) as u8);
                }
            }

            let palette = nq.color_map_rgb();

            let mut frame = gif::Frame {
                width: w,
                height: h,
                left: frame_info.x_offset as u16,
                top: frame_info.y_offset as u16,
                delay: (frame_info.delay_ms / 10) as u16,
                dispose: match frame_info.disposal {
                    DisposalMethod::None => gif::DisposalMethod::Keep,
                    DisposalMethod::Background => gif::DisposalMethod::Background,
                    DisposalMethod::Previous => gif::DisposalMethod::Previous,
                },
                palette: Some(palette),
                buffer: std::borrow::Cow::Owned(indices),
                ..Default::default()
            };

            if has_transparency {
                frame.transparent = Some(0);
            }

            encoder
                .write_frame(&frame)
                .map_err(|e| ImageError::ProcessingFailed(format!("GIF frame: {e}")))?;
        }
    }

    Ok(buf)
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "gif",
        format: "gif",
        mime: "image/gif",
        extensions: &["gif"],
        fn_name: "encode_gif",
        encode_fn: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{
        ColorSpace, DecodedImage, DisposalMethod, FrameInfo, FrameSequence,
    };

    fn make_rgb_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_produces_valid_gif() {
        let (pixels, info) = make_rgb_image(16, 16);
        let result = encode_pixels(&pixels, &info, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&result[..3], b"GIF");
        assert_eq!(&result[3..6], b"89a"); // GIF89a for animation support
    }

    #[test]
    fn encode_rgba8_produces_valid_gif() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&result[..3], b"GIF");
    }

    #[test]
    fn encode_gray8_produces_valid_gif() {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&result[..3], b"GIF");
    }

    #[test]
    fn default_config_repeat_is_infinite() {
        assert_eq!(GifEncodeConfig::default().repeat, 0);
    }

    #[test]
    fn encode_decode_roundtrip_preserves_dimensions() {
        let (pixels, info) = make_rgb_image(32, 32);
        let encoded = encode_pixels(&pixels, &info, &GifEncodeConfig::default()).unwrap();

        // Decode with our native decoder
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 32);
        assert_eq!(decoded.info.height, 32);
    }

    #[test]
    fn encode_with_transparency() {
        // Create RGBA image with some transparent pixels
        let mut pixels = vec![0u8; 16 * 16 * 4];
        for i in 0..16 * 16 {
            pixels[i * 4] = 200; // R
            pixels[i * 4 + 1] = 100; // G
            pixels[i * 4 + 2] = 50; // B
            pixels[i * 4 + 3] = if i < 64 { 0 } else { 255 }; // first quarter transparent
        }
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&result[..3], b"GIF");
    }

    // ── encode_sequence tests ──────────────────────────────────────

    fn make_solid_frame(
        w: u32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
        index: u32,
        delay_ms: u32,
    ) -> (DecodedImage, FrameInfo) {
        let pixels: Vec<u8> = (0..(w * h)).flat_map(|_| [r, g, b]).collect();
        let image = DecodedImage {
            pixels,
            info: ImageInfo {
                width: w,
                height: h,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
            icc_profile: None,
        };
        let info = FrameInfo {
            index,
            delay_ms,
            disposal: DisposalMethod::None,
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
        };
        (image, info)
    }

    #[test]
    fn encode_sequence_3_frames_roundtrip() {
        let mut seq = FrameSequence::new(4, 4);
        seq.push(
            make_solid_frame(4, 4, 255, 0, 0, 0, 100).0,
            make_solid_frame(4, 4, 255, 0, 0, 0, 100).1,
        );
        seq.push(
            make_solid_frame(4, 4, 0, 255, 0, 1, 200).0,
            make_solid_frame(4, 4, 0, 255, 0, 1, 200).1,
        );
        seq.push(
            make_solid_frame(4, 4, 0, 0, 255, 2, 300).0,
            make_solid_frame(4, 4, 0, 0, 255, 2, 300).1,
        );

        let encoded = encode_sequence(&seq, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&encoded[..3], b"GIF");

        // Decode back
        let frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        assert_eq!(frames.len(), 3, "should have 3 frames");

        // Verify delays (stored as centiseconds, we set 100/200/300 ms → 10/20/30 cs)
        assert_eq!(frames[0].1.delay_ms, 100);
        assert_eq!(frames[1].1.delay_ms, 200);
        assert_eq!(frames[2].1.delay_ms, 300);
    }

    #[test]
    fn encode_sequence_preserves_frame_count() {
        let mut seq = FrameSequence::new(8, 8);
        for i in 0..5 {
            let (img, fi) = make_solid_frame(8, 8, (i * 50) as u8, 0, 0, i, 100);
            seq.push(img, fi);
        }

        let encoded = encode_sequence(&seq, &GifEncodeConfig::default()).unwrap();
        let count = crate::domain::decoder::frame_count(&encoded).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn encode_sequence_empty_returns_error() {
        let seq = FrameSequence::new(4, 4);
        let result = encode_sequence(&seq, &GifEncodeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn encode_sequence_with_repeat_config() {
        let mut seq = FrameSequence::new(4, 4);
        let (img, fi) = make_solid_frame(4, 4, 128, 128, 128, 0, 100);
        seq.push(img, fi);

        let config = GifEncodeConfig { repeat: 3 };
        let encoded = encode_sequence(&seq, &config).unwrap();
        assert_eq!(&encoded[..3], b"GIF");
    }
}
