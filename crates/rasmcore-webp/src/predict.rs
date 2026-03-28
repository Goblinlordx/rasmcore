//! VP8 intra-prediction modes (RFC 6386 Sections 12-13).
//!
//! Reusable prediction functions for VP8 and similar codecs.
//! Each function predicts block contents from already-encoded neighbors.

/// 16×16 luma prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra16Mode {
    /// Fill with average of top + left neighbors.
    DC = 0,
    /// Replicate top row downward.
    V = 1,
    /// Replicate left column rightward.
    H = 2,
    /// TrueMotion: above + left - above_left per pixel.
    TM = 3,
}

/// 4×4 luma sub-block prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra4Mode {
    DC = 0,
    TM = 1,
    V = 2,
    H = 3,
    /// Lower-diagonal.
    LD = 4,
    /// Right-diagonal.
    RD = 5,
    /// Vertical-right.
    VR = 6,
    /// Vertical-left.
    VL = 7,
    /// Horizontal-down.
    HD = 8,
    /// Horizontal-up.
    HU = 9,
}

/// 8×8 chroma prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMode {
    DC = 0,
    V = 1,
    H = 2,
    TM = 3,
}

/// Sum of absolute differences between two byte slices.
pub fn sad(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

// TODO: Prediction functions to be implemented in webp-prediction track:
// pub fn predict_16x16(mode, above, left, above_left, dst)
// pub fn predict_4x4(mode, above, left, above_left, above_right, dst)
// pub fn predict_8x8(mode, above, left, above_left, dst)
// pub fn select_best_16x16(actual, above, left, above_left) -> Intra16Mode
