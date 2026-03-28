/// JPEG XL encoder configuration.
#[derive(Debug, Clone)]
pub struct JxlEncodeConfig {
    /// Quality level 1-100 for lossy mode (default: 75).
    /// Maps to VarDCT distance parameter.
    pub quality: u8,
    /// Encoding effort 1-9 (default: 7).
    /// Higher = better compression ratio but slower.
    pub effort: u8,
    /// Use lossless mode (default: false).
    /// When true, uses Modular encoding (exact pixel preservation).
    pub lossless: bool,
}

impl Default for JxlEncodeConfig {
    fn default() -> Self {
        Self {
            quality: 75,
            effort: 7,
            lossless: false,
        }
    }
}
