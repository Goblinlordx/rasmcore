//! Pure Rust HEVC/H.265 intra-frame decoder.
//!
//! Decodes HEVC I-frames as used in HEIC still images.
//! Scoped to intra-only decode — no inter prediction, no DPB.
//!
//! This crate is part of the "nonfree" distribution due to HEVC patent encumbrance.

pub mod bitread;
pub mod cabac;
pub mod error;
pub mod filter;
pub mod nal;
pub mod params;
pub mod predict;
#[cfg(test)]
pub mod testutil;
pub mod transform;
pub mod types;

pub use error::HevcError;
pub use nal::{NalIterator, parse_nal_unit};
pub use params::{DecoderContext, Pps, Sps, Vps};
pub use types::{DecodedFrame, NalUnit, NalUnitType};

/// Decode an HEVC I-frame from raw NAL unit data.
///
/// # Arguments
/// * `bitstream` - Raw HEVC NAL units (Annex B byte stream format)
/// * `codec_config` - hvcC configuration record bytes containing VPS/SPS/PPS
///
/// # Returns
/// Decoded frame with RGB8 pixel data.
///
/// # Errors
/// Returns `HevcError` if the bitstream is malformed or uses unsupported features.
pub fn decode(bitstream: &[u8], codec_config: &[u8]) -> Result<DecodedFrame, HevcError> {
    let _ = (bitstream, codec_config);
    Err(HevcError::DecodeFailed(
        "HEVC decoder not yet implemented — scaffold only".into(),
    ))
}
