//! Quantization and dequantization for JPEG.
//!
//! ITU-T T.81 Section A.3.4. Maps DCT coefficients to integer values
//! using quantization tables derived from quality parameter.
//!
//! Features to implement:
//! - Standard quantization tables (Annex K)
//! - Quality-to-table scaling (libjpeg algorithm)
//! - Custom quantization tables
//! - Trellis quantization (Viterbi-based rate-distortion optimization)
//!
//! Perf notes from zenjpeg:
//! - Trellis quant uses dynamic programming per 8x8 block
//! - Dead zone modulation for psychovisual optimization (jpegli approach)
//! - LUT-based quantize/dequantize for speed

// Stub — implementation in a future track.
