/// TGA encode configuration (no parameters).
#[derive(Debug, Clone, Default)]
pub struct TgaEncodeConfig;

/// Encode pixels to TGA format.
pub fn encode_pixels(
    pixels: &[u8],
    info: &crate::domain::types::ImageInfo,
    _config: &TgaEncodeConfig,
) -> Result<Vec<u8>, crate::domain::error::ImageError> {
    super::native_trivial::encode_tga(pixels, info)
}

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "tga",
        format: "tga",
        mime: "image/x-tga",
        extensions: &["tga"],
        fn_name: "encode_tga",
        encode_fn: None,
    }
}
