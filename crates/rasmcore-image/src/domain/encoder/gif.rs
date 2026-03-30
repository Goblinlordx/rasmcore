use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

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
}
