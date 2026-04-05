/// QOI encode configuration. QOI is a fixed lossless format with no parameters.
#[derive(Debug, Clone, Default)]
pub struct QoiEncodeConfig;

/// Encode pixels to QOI format.
pub fn encode_pixels(
    pixels: &[u8],
    info: &crate::domain::types::ImageInfo,
    _config: &QoiEncodeConfig,
) -> Result<Vec<u8>, crate::domain::error::ImageError> {
    super::native_trivial::encode_qoi(pixels, info)
}

// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "qoi",
        format: "qoi",
        mime: "image/qoi",
        extensions: &["qoi"],
        fn_name: "encode_qoi",
        encode_fn: None,
    }
}
