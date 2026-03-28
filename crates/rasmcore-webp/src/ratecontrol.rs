//! VP8 rate control — quality-to-quantizer mapping.
//!
//! Maps user-facing quality (1-100) to VP8 encoder parameters
//! (QP index, filter strength) matching libwebp behavior.

use crate::filter::FilterType;

/// Complete encoder parameters derived from quality setting.
#[derive(Debug, Clone)]
pub struct EncodeParams {
    /// Luma quantizer index (0-127).
    pub qp_y: u8,
    /// Chroma quantizer index (0-127).
    pub qp_uv: u8,
    /// Loop filter strength (0-63).
    pub filter_level: u8,
    /// Loop filter sharpness (0-7).
    pub filter_sharpness: u8,
    /// Loop filter type.
    pub filter_type: FilterType,
}

// TODO: Implement in webp-ratecontrol track:
// pub fn quality_to_params(quality: u8) -> EncodeParams
