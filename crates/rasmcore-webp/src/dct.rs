//! 4×4 integer DCT and Walsh-Hadamard Transform (RFC 6386 Section 14).
//!
//! Reusable integer transforms for VP8 and similar codecs.
//! All arithmetic is integer-only — no floating point.

// TODO: Implement in webp-dct-quant track:
// pub fn forward_dct(src: &[u8; 16], reference: &[u8; 16], out: &mut [i16; 16])
// pub fn inverse_dct(coeffs: &[i16; 16], reference: &[u8; 16], dst: &mut [u8; 16])
// pub fn forward_wht(dc_coeffs: &[i16; 16], out: &mut [i16; 16])
// pub fn inverse_wht(coeffs: &[i16; 16], out: &mut [i16; 16])
