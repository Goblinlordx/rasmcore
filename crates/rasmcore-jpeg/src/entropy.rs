//! Entropy coding for JPEG — Huffman and arithmetic.
//!
//! Uses rasmcore-deflate::huffman for the shared Huffman module.
//!
//! ITU-T T.81 Sections F (Huffman) and D (Arithmetic).
//!
//! Features to implement:
//! - DC coefficient DPCM (differential coding)
//! - AC coefficient run-length encoding (zigzag scan)
//! - Standard Huffman tables (Annex K)
//! - Optimized Huffman tables (two-pass: count frequencies, build optimal tree)
//! - Arithmetic coding (CABAC-like, optional)
//!
//! Perf notes:
//! - Huffman: table-driven decode (single lookup for codes <= 12 bits)
//! - Arithmetic: inherently sequential, no SIMD benefit
//! - DC DPCM: simple subtraction, branch-free with conditional move

// Stub — implementation in a future track.
