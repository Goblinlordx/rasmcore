//! RGB to YCbCr color space conversion for JPEG.
//!
//! Uses rasmcore-color for parameterized conversion (BT.601 for JPEG).
//! Handles chroma subsampling (4:2:0, 4:2:2, 4:4:4, 4:1:1).
//!
//! Perf notes from zenjpeg:
//! - Uses `wide` crate for SIMD color conversion
//! - Fixed-point integer arithmetic avoids float overhead
//! - Future: WASM SIMD128 acceleration

// Stub — implementation in a future track.
