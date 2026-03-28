//! VP8 loop filter — deblocking filter (RFC 6386 Section 15).
//!
//! Smooths block boundaries to reduce visible artifacts after quantization.

/// Loop filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Fast filter, adjusts 2 pixels per edge.
    Simple,
    /// Stronger filter, adjusts 4 pixels per edge with HEV detection.
    Normal,
}

// TODO: Implement in webp-ratecontrol track:
// pub fn apply_loop_filter(pixels: &mut YuvImage, filter_level: u8,
//                          sharpness: u8, filter_type: FilterType,
//                          mb_info: &[MacroblockInfo])
