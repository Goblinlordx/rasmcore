/// PNM encode configuration (no parameters).
#[derive(Debug, Clone, Default)]
pub struct PnmEncodeConfig;

/// Encode pixels to PNM format.
pub fn encode_pixels(
    pixels: &[u8],
    info: &crate::domain::types::ImageInfo,
    _config: &PnmEncodeConfig,
) -> Result<Vec<u8>, crate::domain::error::ImageError> {
    super::native_trivial::encode_pnm(pixels, info)
}

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "pnm",
        format: "pnm",
        mime: "image/x-portable-anymap",
        extensions: &["pnm", "ppm", "pgm", "pbm"],
        fn_name: "encode_pnm",
        encode_fn: None,
    }
}
