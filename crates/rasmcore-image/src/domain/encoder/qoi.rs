/// QOI encode configuration. QOI is a fixed lossless format with no parameters.
#[derive(Debug, Clone, Default)]
pub struct QoiEncodeConfig;

// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "qoi",
        format: "qoi",
        mime: "image/qoi",
        extensions: &["qoi"],
        fn_name: "encode_qoi",
    }
}
