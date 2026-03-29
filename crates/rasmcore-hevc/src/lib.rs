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
pub mod frame;
pub mod nal;
pub mod params;
pub mod predict;
pub mod syntax;
pub mod testutil;
pub mod transform;
pub mod types;

pub use error::HevcError;
pub use frame::{decode_frame, decode_frame_with_matrix};
pub use nal::{NalIterator, parse_nal_unit};
pub use params::{DecoderContext, Pps, Sps, Vps};
pub use types::{DecodedFrame, NalUnit, NalUnitType};

/// Decode an HEVC I-frame from raw NAL unit data.
///
/// # Arguments
/// * `bitstream` - Raw HEVC NAL units (Annex B byte stream format)
/// * `codec_config` - Optional hvcC configuration record NAL arrays
///
/// # Returns
/// Decoded frame with RGB8 pixel data.
///
/// # Errors
/// Returns `HevcError` if the bitstream is malformed or uses unsupported features.
pub fn decode(bitstream: &[u8], _codec_config: &[u8]) -> Result<DecodedFrame, HevcError> {
    // For now, assume all parameter sets are in-band (Annex B stream).
    // hvcC config parsing will be wired in the heic-integration track.
    decode_frame(bitstream, None)
}
