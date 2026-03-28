//! VP8 coefficient token tree and probability context (RFC 6386 Section 13).
//!
//! Encodes quantized DCT coefficients using a token tree with
//! band-based and complexity-based probability context.

// TODO: Implement in webp-bitstream track:
// pub struct TokenWriter { ... }
// - Token tree encoding
// - Probability context (band, complexity, plane)
// - Default probability tables
