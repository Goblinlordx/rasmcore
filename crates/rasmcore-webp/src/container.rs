//! WebP RIFF container wrapping.
//!
//! Wraps VP8 frame data in a standard WebP file:
//! `RIFF` + file_size + `WEBP` + `VP8 ` + chunk_size + vp8_data

// TODO: Implement in webp-bitstream track:
// pub fn wrap_vp8(vp8_data: &[u8]) -> Vec<u8>
