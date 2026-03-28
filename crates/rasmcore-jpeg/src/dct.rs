//! 8x8 Discrete Cosine Transform for JPEG.
//!
//! ITU-T T.81 Section A.3.3. The core transform of JPEG compression.
//!
//! Algorithms to implement:
//! - Forward DCT: spatial domain → frequency domain (for encoding)
//! - Inverse DCT: frequency domain → spatial domain (for decoding)
//!
//! Perf notes:
//! - Fast integer DCT (AAN/Loeffler algorithm): reduces 4096 multiplies to ~80
//! - rasmcore-webp has a 4x4 integer DCT that can inform the 8x8 design
//! - zenjpeg uses f32 pipeline with SIMD — we target integer-only for determinism
//! - Future: WASM SIMD128 butterfly operations

// Stub — implementation in a future track.
