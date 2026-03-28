/// VP8 encoder configuration.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// Quality level 1-100. Higher values produce larger, higher-quality output.
    /// Maps to VP8 quantizer index matching libwebp behavior.
    pub quality: u8,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self { quality: 75 }
    }
}
