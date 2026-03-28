//! VP8 intra-prediction modes (RFC 6386 Sections 12-13).
//!
//! Reusable prediction functions for VP8 and similar codecs.
//! Each function predicts block contents from already-encoded neighbors.
//!
//! Neighbor layout for a block at position (row, col):
//! ```text
//!          above_left | above[0..N]
//!          -----------+------------
//!          left[0]    | block
//!          left[1]    |
//!          ...        |
//! ```

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

impl Intra4Mode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::DC,
            1 => Self::TM,
            2 => Self::V,
            3 => Self::H,
            4 => Self::LD,
            5 => Self::RD,
            6 => Self::VR,
            7 => Self::VL,
            8 => Self::HD,
            _ => Self::HU,
        }
    }
}

/// 8×8 chroma prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMode {
    DC = 0,
    V = 1,
    H = 2,
    TM = 3,
}

const ALL_INTRA16: [Intra16Mode; 4] = [
    Intra16Mode::DC,
    Intra16Mode::V,
    Intra16Mode::H,
    Intra16Mode::TM,
];

const ALL_INTRA4: [Intra4Mode; 10] = [
    Intra4Mode::DC,
    Intra4Mode::TM,
    Intra4Mode::V,
    Intra4Mode::H,
    Intra4Mode::LD,
    Intra4Mode::RD,
    Intra4Mode::VR,
    Intra4Mode::VL,
    Intra4Mode::HD,
    Intra4Mode::HU,
];

const ALL_CHROMA: [ChromaMode; 4] = [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

/// Clamp to u8 range.
#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Average with rounding: (a + b + 1) >> 1
#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// Average of three with rounding: (a + 2*b + c + 2) >> 2
#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

// =============================================================================
// 16×16 Luma Prediction
// =============================================================================

/// Predict a 16×16 luma macroblock.
///
/// - `above`: 16 pixels from the row above the block.
/// - `left`: 16 pixels from the column left of the block.
/// - `above_left`: the pixel at the top-left corner (above[-1] / left[-1]).
/// - `dst`: output buffer, must be at least 256 bytes (16×16).
pub fn predict_16x16(
    mode: Intra16Mode,
    above: &[u8; 16],
    left: &[u8; 16],
    above_left: u8,
    has_above: bool,
    has_left: bool,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= 256);
    match mode {
        Intra16Mode::DC => predict_16x16_dc(above, left, has_above, has_left, dst),
        Intra16Mode::V => predict_16x16_v(above, dst),
        Intra16Mode::H => predict_16x16_h(left, dst),
        Intra16Mode::TM => predict_16x16_tm(above, left, above_left, dst),
    }
}

/// DC prediction matching VP8 spec / image-webp decoder:
/// - Both neighbors: average all 32 samples
/// - Only above or only left: average 16 samples from available neighbor
/// - Neither: use constant 128
fn predict_16x16_dc(
    above: &[u8; 16],
    left: &[u8; 16],
    has_above: bool,
    has_left: bool,
    dst: &mut [u8],
) {
    let dc = match (has_above, has_left) {
        (true, true) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum::<u32>()
                + left.iter().map(|&v| v as u32).sum::<u32>();
            ((sum + 16) >> 5) as u8
        }
        (true, false) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum();
            ((sum + 8) >> 4) as u8
        }
        (false, true) => {
            let sum: u32 = left.iter().map(|&v| v as u32).sum();
            ((sum + 8) >> 4) as u8
        }
        (false, false) => 128,
    };
    dst[..256].fill(dc);
}

fn predict_16x16_v(above: &[u8; 16], dst: &mut [u8]) {
    for row in 0..16 {
        dst[row * 16..row * 16 + 16].copy_from_slice(above);
    }
}

fn predict_16x16_h(left: &[u8; 16], dst: &mut [u8]) {
    for row in 0..16 {
        dst[row * 16..row * 16 + 16].fill(left[row]);
    }
}

fn predict_16x16_tm(above: &[u8; 16], left: &[u8; 16], above_left: u8, dst: &mut [u8]) {
    let al = above_left as i32;
    for row in 0..16 {
        let l = left[row] as i32;
        for col in 0..16 {
            dst[row * 16 + col] = clip(l + above[col] as i32 - al);
        }
    }
}

// =============================================================================
// 4×4 Sub-block Prediction
// =============================================================================

/// Predict a 4×4 luma sub-block.
///
/// - `above`: 4 pixels from the row above the block.
/// - `left`: 4 pixels from the column left of the block.
/// - `above_left`: the pixel at the top-left corner.
/// - `above_right`: 4 pixels from above-right (for diagonal modes). If not
///   available, repeat `above[3]`.
/// - `dst`: output buffer, must be at least 16 bytes (4×4).
pub fn predict_4x4(
    mode: Intra4Mode,
    above: &[u8; 4],
    left: &[u8; 4],
    above_left: u8,
    above_right: &[u8; 4],
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= 16);
    match mode {
        Intra4Mode::DC => predict_4x4_dc(above, left, dst),
        Intra4Mode::TM => predict_4x4_tm(above, left, above_left, dst),
        Intra4Mode::V => predict_4x4_v(above, dst),
        Intra4Mode::H => predict_4x4_h(left, dst),
        Intra4Mode::LD => predict_4x4_ld(above, above_right, dst),
        Intra4Mode::RD => predict_4x4_rd(above, left, above_left, dst),
        Intra4Mode::VR => predict_4x4_vr(above, left, above_left, dst),
        Intra4Mode::VL => predict_4x4_vl(above, above_right, dst),
        Intra4Mode::HD => predict_4x4_hd(above, left, above_left, dst),
        Intra4Mode::HU => predict_4x4_hu(left, dst),
    }
}

fn predict_4x4_dc(above: &[u8; 4], left: &[u8; 4], dst: &mut [u8]) {
    let sum: u32 =
        above.iter().map(|&v| v as u32).sum::<u32>() + left.iter().map(|&v| v as u32).sum::<u32>();
    let dc = ((sum + 4) >> 3) as u8;
    dst[..16].fill(dc);
}

fn predict_4x4_tm(above: &[u8; 4], left: &[u8; 4], above_left: u8, dst: &mut [u8]) {
    let al = above_left as i32;
    for row in 0..4 {
        let l = left[row] as i32;
        for col in 0..4 {
            dst[row * 4 + col] = clip(l + above[col] as i32 - al);
        }
    }
}

fn predict_4x4_v(above: &[u8; 4], dst: &mut [u8]) {
    for row in 0..4 {
        dst[row * 4..row * 4 + 4].copy_from_slice(above);
    }
}

fn predict_4x4_h(left: &[u8; 4], dst: &mut [u8]) {
    for row in 0..4 {
        dst[row * 4..row * 4 + 4].fill(left[row]);
    }
}

/// Lower-diagonal: samples come from above and above-right, going down-left.
/// RFC 6386 Section 12.3:
///   dst[r][c] = avg3(above[c+r], above[c+r+1], above[c+r+2])
/// with above extended by above_right.
fn predict_4x4_ld(above: &[u8; 4], above_right: &[u8; 4], dst: &mut [u8]) {
    // Build extended above array: above[0..4] + above_right[0..4]
    let ext: [u8; 8] = [
        above[0],
        above[1],
        above[2],
        above[3],
        above_right[0],
        above_right[1],
        above_right[2],
        above_right[3],
    ];
    for r in 0..4 {
        for c in 0..4 {
            let i = c + r;
            dst[r * 4 + c] = if i == 6 {
                // Special case: last pixel uses avg3(ext[6], ext[7], ext[7])
                avg3(ext[6], ext[7], ext[7])
            } else {
                avg3(ext[i], ext[i + 1], ext[i + 2])
            };
        }
    }
}

/// Right-diagonal: samples come from above and left, going down-right.
fn predict_4x4_rd(above: &[u8; 4], left: &[u8; 4], above_left: u8, dst: &mut [u8]) {
    // Build a column of reference pixels: left[3], left[2], left[1], left[0], above_left, above[0..4]
    let ref_pixels: [u8; 9] = [
        left[3], left[2], left[1], left[0], above_left, above[0], above[1], above[2], above[3],
    ];
    // dst[r][c] = avg3(ref[4 - r + c - 1], ref[4 - r + c], ref[4 - r + c + 1])
    // index into ref_pixels: base = 4 + c - r
    for r in 0..4 {
        for c in 0..4 {
            let i = 4 + c - r;
            dst[r * 4 + c] = avg3(ref_pixels[i - 1], ref_pixels[i], ref_pixels[i + 1]);
        }
    }
}

/// Vertical-right: like RD but shifted to give a more vertical slant.
fn predict_4x4_vr(above: &[u8; 4], left: &[u8; 4], above_left: u8, dst: &mut [u8]) {
    let a = above_left;
    let (a0, a1, a2, a3) = (above[0], above[1], above[2], above[3]);
    let (l0, l1, l2) = (left[0], left[1], left[2]);

    // Row 0
    dst[0] = avg2(a, a0);
    dst[1] = avg2(a0, a1);
    dst[2] = avg2(a1, a2);
    dst[3] = avg2(a2, a3);
    // Row 1
    dst[4] = avg3(l0, a, a0);
    dst[5] = avg3(a, a0, a1);
    dst[6] = avg3(a0, a1, a2);
    dst[7] = avg3(a1, a2, a3);
    // Row 2
    dst[8] = avg3(l1, l0, a);
    dst[9] = avg2(a, a0); // same as dst[0]
    dst[10] = avg2(a0, a1); // same as dst[1]
    dst[11] = avg2(a1, a2); // same as dst[2]
    // Row 3
    dst[12] = avg3(l2, l1, l0);
    dst[13] = avg3(l0, a, a0); // same as dst[4]
    dst[14] = avg3(a, a0, a1); // same as dst[5]
    dst[15] = avg3(a0, a1, a2); // same as dst[6]
}

/// Vertical-left: samples go up and to the left.
fn predict_4x4_vl(above: &[u8; 4], above_right: &[u8; 4], dst: &mut [u8]) {
    let (a0, a1, a2, a3) = (above[0], above[1], above[2], above[3]);
    let ar0 = above_right[0];

    // Row 0
    dst[0] = avg2(a0, a1);
    dst[1] = avg2(a1, a2);
    dst[2] = avg2(a2, a3);
    dst[3] = avg2(a3, ar0);
    // Row 1
    dst[4] = avg3(a0, a1, a2);
    dst[5] = avg3(a1, a2, a3);
    dst[6] = avg3(a2, a3, ar0);
    dst[7] = avg3(a3, ar0, above_right[1]);
    // Row 2
    dst[8] = avg2(a1, a2); // same as dst[1]
    dst[9] = avg2(a2, a3); // same as dst[2]
    dst[10] = avg2(a3, ar0); // same as dst[3]
    dst[11] = avg2(ar0, above_right[1]);
    // Row 3
    dst[12] = avg3(a1, a2, a3); // same as dst[5]
    dst[13] = avg3(a2, a3, ar0); // same as dst[6]
    dst[14] = avg3(a3, ar0, above_right[1]); // same as dst[7]
    dst[15] = avg3(ar0, above_right[1], above_right[2]);
}

/// Horizontal-down: samples go from left and above toward lower-right.
fn predict_4x4_hd(above: &[u8; 4], left: &[u8; 4], above_left: u8, dst: &mut [u8]) {
    let a = above_left;
    let (a0, a1, a2) = (above[0], above[1], above[2]);
    let (l0, l1, l2, l3) = (left[0], left[1], left[2], left[3]);

    // Row 0
    dst[0] = avg2(a, l0);
    dst[1] = avg3(l0, a, a0);
    dst[2] = avg3(a, a0, a1);
    dst[3] = avg3(a0, a1, a2);
    // Row 1
    dst[4] = avg2(l0, l1);
    dst[5] = avg3(l1, l0, a);
    dst[6] = avg2(a, l0); // same as dst[0]
    dst[7] = avg3(l0, a, a0); // same as dst[1]
    // Row 2
    dst[8] = avg2(l1, l2);
    dst[9] = avg3(l2, l1, l0);
    dst[10] = avg2(l0, l1); // same as dst[4]
    dst[11] = avg3(l1, l0, a); // same as dst[5]
    // Row 3
    dst[12] = avg2(l2, l3);
    dst[13] = avg3(l3, l2, l1);
    dst[14] = avg2(l1, l2); // same as dst[8]
    dst[15] = avg3(l2, l1, l0); // same as dst[9]
}

/// Horizontal-up: samples go from left upward.
fn predict_4x4_hu(left: &[u8; 4], dst: &mut [u8]) {
    let (l0, l1, l2, l3) = (left[0], left[1], left[2], left[3]);

    // Row 0
    dst[0] = avg2(l0, l1);
    dst[1] = avg3(l0, l1, l2);
    dst[2] = avg2(l1, l2);
    dst[3] = avg3(l1, l2, l3);
    // Row 1
    dst[4] = avg2(l1, l2); // same as dst[2]
    dst[5] = avg3(l1, l2, l3); // same as dst[3]
    dst[6] = avg2(l2, l3);
    dst[7] = avg3(l2, l3, l3);
    // Row 2
    dst[8] = avg2(l2, l3); // same as dst[6]
    dst[9] = avg3(l2, l3, l3); // same as dst[7]
    dst[10] = l3;
    dst[11] = l3;
    // Row 3
    dst[12] = l3;
    dst[13] = l3;
    dst[14] = l3;
    dst[15] = l3;
}

// =============================================================================
// 8×8 Chroma Prediction
// =============================================================================

/// Predict an 8×8 chroma block.
///
/// - `above`: 8 pixels from the row above the block.
/// - `left`: 8 pixels from the column left of the block.
/// - `above_left`: the pixel at the top-left corner.
/// - `dst`: output buffer, must be at least 64 bytes (8×8).
pub fn predict_8x8(
    mode: ChromaMode,
    above: &[u8; 8],
    left: &[u8; 8],
    above_left: u8,
    has_above: bool,
    has_left: bool,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= 64);
    match mode {
        ChromaMode::DC => predict_8x8_dc(above, left, has_above, has_left, dst),
        ChromaMode::V => predict_8x8_v(above, dst),
        ChromaMode::H => predict_8x8_h(left, dst),
        ChromaMode::TM => predict_8x8_tm(above, left, above_left, dst),
    }
}

fn predict_8x8_dc(
    above: &[u8; 8],
    left: &[u8; 8],
    has_above: bool,
    has_left: bool,
    dst: &mut [u8],
) {
    let dc = match (has_above, has_left) {
        (true, true) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum::<u32>()
                + left.iter().map(|&v| v as u32).sum::<u32>();
            ((sum + 8) >> 4) as u8
        }
        (true, false) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum();
            ((sum + 4) >> 3) as u8
        }
        (false, true) => {
            let sum: u32 = left.iter().map(|&v| v as u32).sum();
            ((sum + 4) >> 3) as u8
        }
        (false, false) => 128,
    };
    dst[..64].fill(dc);
}

fn predict_8x8_v(above: &[u8; 8], dst: &mut [u8]) {
    for row in 0..8 {
        dst[row * 8..row * 8 + 8].copy_from_slice(above);
    }
}

fn predict_8x8_h(left: &[u8; 8], dst: &mut [u8]) {
    for row in 0..8 {
        dst[row * 8..row * 8 + 8].fill(left[row]);
    }
}

fn predict_8x8_tm(above: &[u8; 8], left: &[u8; 8], above_left: u8, dst: &mut [u8]) {
    let al = above_left as i32;
    for row in 0..8 {
        let l = left[row] as i32;
        for col in 0..8 {
            dst[row * 8 + col] = clip(l + above[col] as i32 - al);
        }
    }
}

// =============================================================================
// Sum of Absolute Differences
// =============================================================================

/// Sum of absolute differences between two byte slices.
/// Sum of absolute differences between two byte slices.
///
/// On WASM, processes 16 bytes at a time using u8x16 SIMD.
pub fn sad(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "wasm32")]
    {
        sad_simd128(a, b)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        sad_scalar(a, b)
    }
}

fn sad_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

#[cfg(target_arch = "wasm32")]
fn sad_simd128(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::wasm32::*;
    let len = a.len();
    let mut sum = 0u32;
    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        // SAFETY: i + 16 <= len is checked, so a[i..i+16] and b[i..i+16] are in bounds.
        let (va, vb) = unsafe {
            (
                v128_load(a[i..].as_ptr() as *const v128),
                v128_load(b[i..].as_ptr() as *const v128),
            )
        };
        // |a - b| = max(a, b) - min(a, b) for unsigned bytes
        let abs_diff = u8x16_sub_sat(u8x16_max(va, vb), u8x16_min(va, vb));

        // Horizontal sum: widen to i16x8, then i32x4, then extract
        let lo = u16x8_extend_low_u8x16(abs_diff);
        let hi = u16x8_extend_high_u8x16(abs_diff);
        let sum16 = i16x8_add(lo, hi); // 8 partial sums

        let lo32 = u32x4_extend_low_u16x8(sum16);
        let hi32 = u32x4_extend_high_u16x8(sum16);
        let sum32 = i32x4_add(lo32, hi32); // 4 partial sums

        sum += i32x4_extract_lane::<0>(sum32) as u32
            + i32x4_extract_lane::<1>(sum32) as u32
            + i32x4_extract_lane::<2>(sum32) as u32
            + i32x4_extract_lane::<3>(sum32) as u32;

        i += 16;
    }

    // Handle remainder
    while i < len {
        sum += (a[i] as i32 - b[i] as i32).unsigned_abs();
        i += 1;
    }

    sum
}

// =============================================================================
// Mode Selection
// =============================================================================

/// Select the best 16×16 prediction mode by minimizing SAD.
///
/// Tries all 4 modes and returns the one with the lowest SAD against `actual`.
pub fn select_best_16x16(
    actual: &[u8],
    above: &[u8; 16],
    left: &[u8; 16],
    above_left: u8,
    has_above: bool,
    has_left: bool,
) -> Intra16Mode {
    let mut best_mode = Intra16Mode::DC;
    let mut best_sad = u32::MAX;
    let mut pred = [0u8; 256];

    for &mode in &ALL_INTRA16 {
        predict_16x16(
            mode, above, left, above_left, has_above, has_left, &mut pred,
        );
        let cost = sad(&actual[..256], &pred);
        if cost < best_sad {
            best_sad = cost;
            best_mode = mode;
        }
    }
    best_mode
}

/// Select the best 4×4 prediction mode by minimizing SAD.
pub fn select_best_4x4(
    actual: &[u8],
    above: &[u8; 4],
    left: &[u8; 4],
    above_left: u8,
    above_right: &[u8; 4],
) -> Intra4Mode {
    let mut best_mode = Intra4Mode::DC;
    let mut best_sad = u32::MAX;
    let mut pred = [0u8; 16];

    for &mode in &ALL_INTRA4 {
        predict_4x4(mode, above, left, above_left, above_right, &mut pred);
        let cost = sad(&actual[..16], &pred);
        if cost < best_sad {
            best_sad = cost;
            best_mode = mode;
        }
    }
    best_mode
}

/// Select the best 8×8 chroma prediction mode by minimizing SAD.
pub fn select_best_8x8(
    actual: &[u8],
    above: &[u8; 8],
    left: &[u8; 8],
    above_left: u8,
    has_above: bool,
    has_left: bool,
) -> ChromaMode {
    let mut best_mode = ChromaMode::DC;
    let mut best_sad = u32::MAX;
    let mut pred = [0u8; 64];

    for &mode in &ALL_CHROMA {
        predict_8x8(
            mode, above, left, above_left, has_above, has_left, &mut pred,
        );
        let cost = sad(&actual[..64], &pred);
        if cost < best_sad {
            best_sad = cost;
            best_mode = mode;
        }
    }
    best_mode
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- 16×16 prediction tests ----

    #[test]
    fn predict_16x16_dc_averages_neighbors() {
        // above = all 100, left = all 200  →  DC = (16*100 + 16*200 + 16) / 32 = 150
        let above = [100u8; 16];
        let left = [200u8; 16];
        let mut dst = [0u8; 256];
        predict_16x16(Intra16Mode::DC, &above, &left, 0, true, true, &mut dst);
        assert!(dst.iter().all(|&v| v == 150));
    }

    #[test]
    fn predict_16x16_v_copies_above_row() {
        let above: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let left = [0u8; 16];
        let mut dst = [0u8; 256];
        predict_16x16(Intra16Mode::V, &above, &left, 0, true, true, &mut dst);
        for row in 0..16 {
            assert_eq!(&dst[row * 16..row * 16 + 16], &above);
        }
    }

    #[test]
    fn predict_16x16_h_copies_left_column() {
        let above = [0u8; 16];
        let left: [u8; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let mut dst = [0u8; 256];
        predict_16x16(Intra16Mode::H, &above, &left, 0, true, true, &mut dst);
        for row in 0..16 {
            assert!(dst[row * 16..row * 16 + 16].iter().all(|&v| v == left[row]));
        }
    }

    #[test]
    fn predict_16x16_tm_uses_truemotion() {
        // TM: dst[r][c] = clip(left[r] + above[c] - above_left)
        let above = [100u8; 16];
        let left = [50u8; 16];
        let above_left = 75u8;
        let mut dst = [0u8; 256];
        predict_16x16(
            Intra16Mode::TM,
            &above,
            &left,
            above_left,
            true,
            true,
            &mut dst,
        );
        // Expected: 50 + 100 - 75 = 75
        assert!(dst.iter().all(|&v| v == 75));
    }

    #[test]
    fn predict_16x16_tm_clips_to_valid_range() {
        let above = [250u8; 16];
        let left = [250u8; 16];
        let above_left = 10u8;
        let mut dst = [0u8; 256];
        predict_16x16(
            Intra16Mode::TM,
            &above,
            &left,
            above_left,
            true,
            true,
            &mut dst,
        );
        // 250 + 250 - 10 = 490 → clipped to 255
        assert!(dst.iter().all(|&v| v == 255));
    }

    // ---- 4×4 prediction tests ----

    #[test]
    fn predict_4x4_dc_averages_neighbors() {
        let above = [100u8; 4];
        let left = [200u8; 4];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::DC, &above, &left, 0, &[0; 4], &mut dst);
        // (4*100 + 4*200 + 4) / 8 = 1204/8 = 150
        assert!(dst.iter().all(|&v| v == 150));
    }

    #[test]
    fn predict_4x4_v_copies_above() {
        let above = [10, 20, 30, 40];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::V, &above, &[0; 4], 0, &[0; 4], &mut dst);
        for row in 0..4 {
            assert_eq!(&dst[row * 4..row * 4 + 4], &above);
        }
    }

    #[test]
    fn predict_4x4_h_copies_left() {
        let left = [10, 20, 30, 40];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::H, &[0; 4], &left, 0, &[0; 4], &mut dst);
        for row in 0..4 {
            assert!(dst[row * 4..row * 4 + 4].iter().all(|&v| v == left[row]));
        }
    }

    #[test]
    fn predict_4x4_tm_correct() {
        let above = [100; 4];
        let left = [50; 4];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::TM, &above, &left, 75, &[0; 4], &mut dst);
        // 50 + 100 - 75 = 75
        assert!(dst.iter().all(|&v| v == 75));
    }

    #[test]
    fn predict_4x4_ld_known_output() {
        // Increasing sequence: above=[10,20,30,40], above_right=[50,60,70,80]
        let above = [10, 20, 30, 40];
        let above_right = [50, 60, 70, 80];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::LD, &above, &[0; 4], 0, &above_right, &mut dst);
        // Row 0: avg3(10,20,30), avg3(20,30,40), avg3(30,40,50), avg3(40,50,60)
        assert_eq!(dst[0], avg3(10, 20, 30));
        assert_eq!(dst[1], avg3(20, 30, 40));
        assert_eq!(dst[2], avg3(30, 40, 50));
        assert_eq!(dst[3], avg3(40, 50, 60));
        // Row 1: avg3(20,30,40), avg3(30,40,50), avg3(40,50,60), avg3(50,60,70)
        assert_eq!(dst[4], avg3(20, 30, 40));
    }

    #[test]
    fn predict_4x4_rd_known_output() {
        let above = [40, 50, 60, 70];
        let left = [30, 20, 10, 0];
        let above_left = 35u8;
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::RD, &above, &left, above_left, &[0; 4], &mut dst);
        // ref_pixels = [0, 10, 20, 30, 35, 40, 50, 60, 70]
        //               0   1   2   3   4   5   6   7   8
        // dst[0][0] = avg3(ref[3], ref[4], ref[5]) = avg3(30, 35, 40) = (30+70+40+2)/4 = 35
        assert_eq!(dst[0], avg3(30, 35, 40));
        // dst[0][1] = avg3(ref[4], ref[5], ref[6]) = avg3(35, 40, 50)
        assert_eq!(dst[1], avg3(35, 40, 50));
    }

    #[test]
    fn predict_4x4_hu_known_output() {
        let left = [10, 20, 30, 40];
        let mut dst = [0u8; 16];
        predict_4x4(Intra4Mode::HU, &[0; 4], &left, 0, &[0; 4], &mut dst);
        // Row 0: avg2(10,20), avg3(10,20,30), avg2(20,30), avg3(20,30,40)
        assert_eq!(dst[0], avg2(10, 20));
        assert_eq!(dst[1], avg3(10, 20, 30));
        assert_eq!(dst[2], avg2(20, 30));
        assert_eq!(dst[3], avg3(20, 30, 40));
        // Last row: all left[3]
        assert!(dst[12..16].iter().all(|&v| v == 40));
    }

    #[test]
    fn predict_4x4_all_modes_produce_output() {
        let above = [128; 4];
        let left = [128; 4];
        let above_right = [128; 4];
        for &mode in &ALL_INTRA4 {
            let mut dst = [0u8; 16];
            predict_4x4(mode, &above, &left, 128, &above_right, &mut dst);
            // All modes with uniform input should produce 128
            assert!(
                dst.iter().all(|&v| v == 128),
                "Mode {mode:?} failed with uniform 128 input"
            );
        }
    }

    // ---- 8×8 chroma prediction tests ----

    #[test]
    fn predict_8x8_dc_averages_neighbors() {
        let above = [100u8; 8];
        let left = [200u8; 8];
        let mut dst = [0u8; 64];
        predict_8x8(ChromaMode::DC, &above, &left, 0, true, true, &mut dst);
        // (8*100 + 8*200 + 8) / 16 = 2408/16 = 150
        assert!(dst.iter().all(|&v| v == 150));
    }

    #[test]
    fn predict_8x8_v_copies_above_row() {
        let above = [0, 10, 20, 30, 40, 50, 60, 70];
        let mut dst = [0u8; 64];
        predict_8x8(ChromaMode::V, &above, &[0; 8], 0, true, true, &mut dst);
        for row in 0..8 {
            assert_eq!(&dst[row * 8..row * 8 + 8], &above);
        }
    }

    #[test]
    fn predict_8x8_h_copies_left_column() {
        let left = [10, 20, 30, 40, 50, 60, 70, 80];
        let mut dst = [0u8; 64];
        predict_8x8(ChromaMode::H, &[0; 8], &left, 0, true, true, &mut dst);
        for row in 0..8 {
            assert!(dst[row * 8..row * 8 + 8].iter().all(|&v| v == left[row]));
        }
    }

    #[test]
    fn predict_8x8_tm_correct() {
        let above = [100u8; 8];
        let left = [50u8; 8];
        let mut dst = [0u8; 64];
        predict_8x8(ChromaMode::TM, &above, &left, 75, true, true, &mut dst);
        assert!(dst.iter().all(|&v| v == 75));
    }

    // ---- SAD tests ----

    #[test]
    fn sad_identical_is_zero() {
        let a = [128u8; 16];
        assert_eq!(sad(&a, &a), 0);
    }

    #[test]
    fn sad_known_value() {
        let a = [10u8; 4];
        let b = [20u8; 4];
        assert_eq!(sad(&a, &b), 40); // 4 * 10
    }

    // ---- Mode selection tests ----

    #[test]
    fn select_16x16_flat_block_picks_dc() {
        // A flat block with uniform value matches DC prediction perfectly.
        let actual = [128u8; 256];
        let above = [128u8; 16];
        let left = [128u8; 16];
        let mode = select_best_16x16(&actual, &above, &left, 128, true, true);
        // All modes produce 128 for uniform input, DC should win (or tie at 0 SAD).
        // DC is tried first, so it should be selected.
        assert_eq!(mode, Intra16Mode::DC);
    }

    #[test]
    fn select_16x16_vertical_gradient_picks_v() {
        // Columns all same, rows differ — V mode should be perfect.
        let above: [u8; 16] = [
            0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255,
        ];
        let left = [128u8; 16]; // irrelevant for V
        let mut actual = [0u8; 256];
        // Fill actual as if V prediction is perfect
        for row in 0..16 {
            actual[row * 16..row * 16 + 16].copy_from_slice(&above);
        }
        let mode = select_best_16x16(&actual, &above, &left, 128, true, true);
        assert_eq!(mode, Intra16Mode::V);
    }

    #[test]
    fn select_16x16_horizontal_gradient_picks_h() {
        let above = [128u8; 16];
        let left: [u8; 16] = [
            0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255,
        ];
        let mut actual = [0u8; 256];
        for row in 0..16 {
            actual[row * 16..row * 16 + 16].fill(left[row]);
        }
        let mode = select_best_16x16(&actual, &above, &left, 128, true, true);
        assert_eq!(mode, Intra16Mode::H);
    }

    #[test]
    fn select_4x4_vertical_gradient_picks_v() {
        // V mode copies above row to every row. Use neighbors that make TM/H/DC wrong.
        let above = [10, 20, 30, 40];
        // left and above_left chosen so TM produces a different (worse) result.
        let left = [10u8; 4];
        let above_left = 10u8;
        let above_right = [50; 4];
        let mut actual = [0u8; 16];
        for row in 0..4 {
            actual[row * 4..row * 4 + 4].copy_from_slice(&above);
        }
        let mode = select_best_4x4(&actual, &above, &left, above_left, &above_right);
        // TM would produce left[r] + above[c] - above_left = 10 + above[c] - 10 = above[c]
        // which is the same as V! So TM ties with V. DC ties at first, so check V or TM wins.
        assert!(
            mode == Intra4Mode::V || mode == Intra4Mode::TM,
            "Expected V or TM (both perfect), got {mode:?}"
        );
    }

    #[test]
    fn select_8x8_flat_picks_dc() {
        let actual = [100u8; 64];
        let above = [100u8; 8];
        let left = [100u8; 8];
        let mode = select_best_8x8(&actual, &above, &left, 100, true, true);
        assert_eq!(mode, ChromaMode::DC);
    }

    #[test]
    fn select_8x8_horizontal_gradient_picks_h() {
        let above = [128u8; 8];
        let left: [u8; 8] = [10, 30, 50, 70, 90, 110, 130, 150];
        let mut actual = [0u8; 64];
        for row in 0..8 {
            actual[row * 8..row * 8 + 8].fill(left[row]);
        }
        let mode = select_best_8x8(&actual, &above, &left, 128, true, true);
        assert_eq!(mode, ChromaMode::H);
    }
}
