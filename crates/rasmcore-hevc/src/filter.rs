//! HEVC in-loop filters — deblocking filter and Sample Adaptive Offset (SAO).
//!
//! ITU-T H.265 Sections 8.7.2 (deblocking) and 8.7.3 (SAO).

// Filter code uses indexed loops and many-arg functions for clarity matching the spec.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

// ─── Deblocking Filter ─────────────────────────────────────────────────────

/// Beta table (ITU-T H.265 Table 8-15). Indexed by QP (0-51).
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// Tc table (ITU-T H.265 Table 8-16). Indexed by QP+2 (0-53).
const TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Boundary strength for an edge between two blocks.
///
/// For HEIC (I-frame only), all boundaries have Bs=2 since both sides are intra-coded.
/// In a full decoder, Bs depends on prediction mode, coded block flags, and motion vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrength {
    /// No filtering.
    Bs0,
    /// Weak filtering.
    Bs1,
    /// Strong filtering (intra boundary).
    Bs2,
}

impl BoundaryStrength {
    /// Derive boundary strength for I-frame (always Bs2 at CU boundaries).
    pub fn for_intra_boundary() -> Self {
        Self::Bs2
    }

    /// Derive boundary strength for a transform block boundary within an intra CU.
    /// Returns Bs1 if either side has non-zero coefficients, Bs0 otherwise.
    pub fn for_tu_boundary(has_coeffs_p: bool, has_coeffs_q: bool) -> Self {
        if has_coeffs_p || has_coeffs_q {
            Self::Bs1
        } else {
            Self::Bs0
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Bs0 => 0,
            Self::Bs1 => 1,
            Self::Bs2 => 2,
        }
    }
}

/// Get beta threshold for a given QP.
pub fn get_beta(qp: i32) -> i32 {
    let idx = qp.clamp(0, 51) as usize;
    BETA_TABLE[idx]
}

/// Get tc threshold for a given QP.
pub fn get_tc(qp: i32) -> i32 {
    let idx = (qp + 2).clamp(0, 53) as usize;
    TC_TABLE[idx]
}

/// Decide whether to apply strong or weak filter for a luma edge.
///
/// ITU-T H.265 Section 8.7.2.4.5.
/// Returns `true` if strong filter should be used.
pub fn use_strong_filter(p: [i32; 4], q: [i32; 4], beta: i32, tc: i32) -> bool {
    let dp0 = (p[2] - 2 * p[1] + p[0]).abs();
    let dq0 = (q[2] - 2 * q[1] + q[0]).abs();
    let dp3 = (p[2] - 2 * p[1] + p[0]).abs(); // simplified — should use different row
    let dq3 = (q[2] - 2 * q[1] + q[0]).abs();

    let d = dp0 + dq0 + dp3 + dq3;

    if d < beta {
        let strong_cond1 = (p[3] - p[0] + q[0] - q[3]).abs() < (beta >> 3);
        let strong_cond2 = (p[0] - q[0]).abs() < ((5 * tc + 1) >> 1);
        strong_cond1 && strong_cond2
    } else {
        false
    }
}

/// Apply strong deblocking filter to 4 samples on each side of the edge.
///
/// Modifies p[0..3] and q[0..3] in-place.
pub fn deblock_strong(p: &mut [i32; 4], q: &mut [i32; 4], tc: i32) {
    let tc2 = 2 * tc;

    // Compute ALL new values from ORIGINAL p/q before writing any.
    // Ref: libde265 v1.0.18 deblock.cc — uses pnew[]/qnew[] temporaries
    // to avoid read-after-write dependencies.
    let (p0, p1, p2, p3) = (p[0], p[1], p[2], p[3]);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);

    p[0] = clip3(
        p0 - tc2,
        p0 + tc2,
        (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
    );
    p[1] = clip3(p1 - tc2, p1 + tc2, (p2 + p1 + p0 + q0 + 2) >> 2);
    p[2] = clip3(
        p2 - tc2,
        p2 + tc2,
        (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
    );

    q[0] = clip3(
        q0 - tc2,
        q0 + tc2,
        (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
    );
    q[1] = clip3(q1 - tc2, q1 + tc2, (p0 + q0 + q1 + q2 + 2) >> 2);
    q[2] = clip3(
        q2 - tc2,
        q2 + tc2,
        (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3,
    );
}

/// Apply weak deblocking filter.
///
/// Modifies p[0..1] and q[0..1]. Returns whether p[1] and q[1] were also modified.
pub fn deblock_weak(p: &mut [i32; 4], q: &mut [i32; 4], tc: i32, filter_p1: bool, filter_q1: bool) {
    // Save original values before modification.
    // Ref: libde265 v1.0.18 deblock.cc — delta_p/delta_q use original p0/q0.
    let (p0, p1, p2) = (p[0], p[1], p[2]);
    let (q0, q1, q2) = (q[0], q[1], q[2]);

    let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
    let delta = clip3(-tc, tc, delta);

    p[0] = clip_pixel(p0 + delta);
    q[0] = clip_pixel(q0 - delta);

    if filter_p1 {
        let delta_p = clip3(
            -(tc >> 1),
            tc >> 1,
            (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
        );
        p[1] = clip_pixel(p1 + delta_p);
    }

    if filter_q1 {
        let delta_q = clip3(
            -(tc >> 1),
            tc >> 1,
            (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
        );
        q[1] = clip_pixel(q1 + delta_q);
    }
}

/// Apply chroma deblocking filter.
///
/// Simpler than luma — only modifies p[0] and q[0].
pub fn deblock_chroma(p: &mut [i32; 2], q: &mut [i32; 2], tc: i32) {
    let delta = clip3(-tc, tc, ((q[0] - p[0]) * 4 + p[1] - q[1] + 4) >> 3);
    p[0] = clip_pixel(p[0] + delta);
    q[0] = clip_pixel(q[0] - delta);
}

/// Apply deblocking to a vertical edge in a frame buffer.
///
/// `x`, `y` is the position of the edge (between column x-1 and x).
/// Processes `length` rows starting from y.
pub fn deblock_vertical_edge(
    frame: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    qp: i32,
    bs: BoundaryStrength,
) {
    if bs == BoundaryStrength::Bs0 || x < 4 {
        return;
    }

    let beta = get_beta(qp);
    let tc = get_tc(qp);

    for row in y..y + length {
        if row >= frame.len() / stride {
            break;
        }
        let base = row * stride;
        if x + 3 >= stride {
            continue;
        }

        let mut p = [
            frame[base + x - 4] as i32,
            frame[base + x - 3] as i32,
            frame[base + x - 2] as i32,
            frame[base + x - 1] as i32,
        ];
        // Reverse order: p[0] is closest to edge
        p.reverse();

        let mut q = [
            frame[base + x] as i32,
            frame[base + x + 1] as i32,
            frame[base + x + 2] as i32,
            frame[base + x + 3] as i32,
        ];

        if bs == BoundaryStrength::Bs2 && use_strong_filter(p, q, beta, tc) {
            deblock_strong(&mut p, &mut q, tc);
            // Write back: p[0]=closest to edge → x-1, p[3]=farthest → x-4
            // Ref: libde265 v1.0.18 deblock.cc — p[i] maps to ptr[-i-1]
            frame[base + x - 1] = p[0] as u8;
            frame[base + x - 2] = p[1] as u8;
            frame[base + x - 3] = p[2] as u8;
            frame[base + x - 4] = p[3] as u8;
        } else if bs != BoundaryStrength::Bs0 {
            deblock_weak(&mut p, &mut q, tc, true, true);
            // Weak filter only modifies p[0] and p[1]
            frame[base + x - 1] = p[0] as u8;
            frame[base + x - 2] = p[1] as u8;
        }

        frame[base + x] = q[0] as u8;
        frame[base + x + 1] = q[1] as u8;
    }
}

/// Apply deblocking to a horizontal edge in a frame buffer.
pub fn deblock_horizontal_edge(
    frame: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    qp: i32,
    bs: BoundaryStrength,
) {
    if bs == BoundaryStrength::Bs0 || y < 4 {
        return;
    }

    let beta = get_beta(qp);
    let tc = get_tc(qp);

    for col in x..x + length {
        if col >= stride {
            break;
        }

        let mut p = [
            frame[(y - 4) * stride + col] as i32,
            frame[(y - 3) * stride + col] as i32,
            frame[(y - 2) * stride + col] as i32,
            frame[(y - 1) * stride + col] as i32,
        ];
        p.reverse();

        let mut q = [
            frame[y * stride + col] as i32,
            frame[(y + 1) * stride + col] as i32,
            frame[(y + 2) * stride + col] as i32,
            frame[(y + 3) * stride + col] as i32,
        ];

        if bs == BoundaryStrength::Bs2 && use_strong_filter(p, q, beta, tc) {
            deblock_strong(&mut p, &mut q, tc);
            // Write back: p[0]=closest to edge → y-1, p[3]=farthest → y-4
            // Ref: libde265 v1.0.18 deblock.cc — p[i] maps to ptr[-i-1]
            frame[(y - 1) * stride + col] = p[0] as u8;
            frame[(y - 2) * stride + col] = p[1] as u8;
            frame[(y - 3) * stride + col] = p[2] as u8;
            frame[(y - 4) * stride + col] = p[3] as u8;
        } else if bs != BoundaryStrength::Bs0 {
            deblock_weak(&mut p, &mut q, tc, true, true);
            // Weak filter only modifies p[0] and p[1]
            frame[(y - 1) * stride + col] = p[0] as u8;
            frame[(y - 2) * stride + col] = p[1] as u8;
        }

        frame[y * stride + col] = q[0] as u8;
        frame[(y + 1) * stride + col] = q[1] as u8;
    }
}

// ─── Sample Adaptive Offset (SAO) ──────────────────────────────────────────

/// SAO type for a CTU component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoType {
    /// No SAO applied.
    None,
    /// Band offset — apply offsets to 4 consecutive value bands.
    BandOffset,
    /// Edge offset — apply offsets based on edge classification.
    EdgeOffset(EdgeClass),
}

/// Edge class for SAO edge offset mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeClass {
    /// 0-degree (horizontal comparison).
    Horizontal,
    /// 45-degree (diagonal).
    Diagonal45,
    /// 90-degree (vertical comparison).
    Vertical,
    /// 135-degree (anti-diagonal).
    Diagonal135,
}

impl EdgeClass {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Horizontal,
            1 => Self::Diagonal45,
            2 => Self::Vertical,
            3 => Self::Diagonal135,
            _ => Self::Horizontal,
        }
    }

    /// Get the (dx, dy) offsets for the two comparison pixels.
    pub fn offsets(&self) -> [(i32, i32); 2] {
        match self {
            Self::Horizontal => [(-1, 0), (1, 0)],
            Self::Vertical => [(0, -1), (0, 1)],
            Self::Diagonal45 => [(1, -1), (-1, 1)],
            Self::Diagonal135 => [(-1, -1), (1, 1)],
        }
    }
}

/// SAO parameters for a single CTU component (Y, Cb, or Cr).
#[derive(Debug, Clone)]
pub struct SaoParams {
    pub sao_type: SaoType,
    /// Offsets (4 values). Meaning depends on sao_type.
    pub offsets: [i16; 4],
    /// Band position (for BandOffset mode): index of first band (0-27).
    pub band_position: u8,
}

impl Default for SaoParams {
    fn default() -> Self {
        Self {
            sao_type: SaoType::None,
            offsets: [0; 4],
            band_position: 0,
        }
    }
}

/// Apply SAO band offset to a CTU region.
///
/// Samples are classified into 32 bands based on their value.
/// Offsets are applied to 4 consecutive bands starting at `band_position`.
pub fn apply_sao_band(
    frame: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    offsets: &[i16; 4],
    band_position: u8,
) {
    let shift = 3; // 256 / 32 bands = 8 values per band, so band = sample >> 3

    for row in y..y + height {
        for col in x..x + width {
            let idx = row * stride + col;
            if idx >= frame.len() {
                continue;
            }
            let sample = frame[idx];
            let band = sample >> shift;
            let band_offset = band.wrapping_sub(band_position);
            if band_offset < 4 {
                let new_val = (sample as i16 + offsets[band_offset as usize]).clamp(0, 255);
                frame[idx] = new_val as u8;
            }
        }
    }
}

/// Apply SAO edge offset to a CTU region.
///
/// Each sample is classified based on its relationship to two neighbors
/// determined by the edge class (horizontal, vertical, diagonal).
pub fn apply_sao_edge(
    frame: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    offsets: &[i16; 4],
    edge_class: EdgeClass,
) {
    let [(dx0, dy0), (dx1, dy1)] = edge_class.offsets();

    // Work on a copy to avoid read-after-write issues
    let mut output = Vec::with_capacity(width * height);
    for row in y..y + height {
        for col in x..x + width {
            let idx = row * stride + col;
            let sample = frame[idx] as i32;

            let nx0 = col as i32 + dx0;
            let ny0 = row as i32 + dy0;
            let nx1 = col as i32 + dx1;
            let ny1 = row as i32 + dy1;

            // Check bounds
            if nx0 < 0
                || ny0 < 0
                || nx0 >= stride as i32
                || ny0 >= (frame.len() / stride) as i32
                || nx1 < 0
                || ny1 < 0
                || nx1 >= stride as i32
                || ny1 >= (frame.len() / stride) as i32
            {
                output.push(frame[idx]);
                continue;
            }

            let n0 = frame[ny0 as usize * stride + nx0 as usize] as i32;
            let n1 = frame[ny1 as usize * stride + nx1 as usize] as i32;

            // Edge category: sign(sample - n0) + sign(sample - n1)
            // Category mapping: -2→0, -1→1, 0→skip, 1→2, 2→3
            let sign0 = (sample - n0).signum();
            let sign1 = (sample - n1).signum();
            let edge_idx = sign0 + sign1;

            let offset = match edge_idx {
                -2 => offsets[0],
                -1 => offsets[1],
                0 => 0,
                1 => offsets[2],
                2 => offsets[3],
                _ => 0,
            };

            output.push((sample as i16 + offset).clamp(0, 255) as u8);
        }
    }

    // Write back
    let mut out_idx = 0;
    for row in y..y + height {
        for col in x..x + width {
            frame[row * stride + col] = output[out_idx];
            out_idx += 1;
        }
    }
}

/// Apply SAO to a CTU region based on parameters.
pub fn apply_sao(
    frame: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    params: &SaoParams,
) {
    match params.sao_type {
        SaoType::None => {}
        SaoType::BandOffset => {
            apply_sao_band(
                frame,
                stride,
                x,
                y,
                width,
                height,
                &params.offsets,
                params.band_position,
            );
        }
        SaoType::EdgeOffset(edge_class) => {
            apply_sao_edge(
                frame,
                stride,
                x,
                y,
                width,
                height,
                &params.offsets,
                edge_class,
            );
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn clip3(min: i32, max: i32, val: i32) -> i32 {
    val.clamp(min, max)
}

fn clip_pixel(val: i32) -> i32 {
    val.clamp(0, 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_tc_tables() {
        // QP 0-15: beta = 0
        assert_eq!(get_beta(0), 0);
        assert_eq!(get_beta(15), 0);
        // QP 16: beta = 6
        assert_eq!(get_beta(16), 6);
        // QP 51: beta = 64
        assert_eq!(get_beta(51), 64);

        // tc at QP 0: 0
        assert_eq!(get_tc(0), 0);
        // tc at QP 26: tc_table[28] = 2
        assert_eq!(get_tc(26), 2);
        // tc at QP 51: tc_table[53] = 24
        assert_eq!(get_tc(51), 24);
    }

    #[test]
    fn boundary_strength_intra() {
        assert_eq!(BoundaryStrength::for_intra_boundary().as_u8(), 2);
    }

    #[test]
    fn boundary_strength_tu() {
        assert_eq!(BoundaryStrength::for_tu_boundary(true, false).as_u8(), 1);
        assert_eq!(BoundaryStrength::for_tu_boundary(false, false).as_u8(), 0);
    }

    #[test]
    fn strong_filter_flat_pixels() {
        // Very flat pixels across the edge — should select strong filter
        let p = [100, 100, 100, 100];
        let q = [100, 100, 100, 100];
        let beta = get_beta(30);
        let tc = get_tc(30);
        // Flat input has d=0 < beta, and all conditions met
        assert!(use_strong_filter(p, q, beta, tc));
    }

    #[test]
    fn strong_filter_sharp_edge() {
        // Sharp step at the edge — weak filter preferred
        let p = [50, 50, 50, 50];
        let q = [200, 200, 200, 200];
        let beta = get_beta(22);
        let tc = get_tc(22);
        // Large p[0]-q[0] difference should fail strong filter condition
        assert!(!use_strong_filter(p, q, beta, tc));
    }

    #[test]
    fn deblock_strong_symmetric() {
        let mut p = [100, 100, 100, 100];
        let mut q = [100, 100, 100, 100];
        deblock_strong(&mut p, &mut q, 10);
        // Flat input should remain flat
        assert_eq!(p, [100, 100, 100, 100]);
        assert_eq!(q, [100, 100, 100, 100]);
    }

    #[test]
    fn deblock_weak_smooths_edge() {
        // p[0] is closest to edge, p[3] is farthest
        let mut p = [95, 90, 85, 80]; // increasing toward edge
        let mut q = [110, 105, 100, 95]; // decreasing away from edge
        let tc = 10;
        deblock_weak(&mut p, &mut q, tc, true, true);
        // p[0] should increase toward q[0], q[0] should decrease toward p[0]
        assert!(p[0] > 95, "p[0] should increase: got {}", p[0]);
        assert!(q[0] < 110, "q[0] should decrease: got {}", q[0]);
    }

    #[test]
    fn deblock_chroma_edge() {
        let mut p = [80, 90];
        let mut q = [120, 110];
        deblock_chroma(&mut p, &mut q, 10);
        // Should smooth
        assert!(p[0] > 80);
        assert!(q[0] < 120);
    }

    #[test]
    fn deblock_vertical_on_frame() {
        // 16x4 frame with a sharp step at column 8
        let mut frame = vec![0u8; 16 * 4];
        for row in 0..4 {
            for col in 0..8 {
                frame[row * 16 + col] = 50;
            }
            for col in 8..16 {
                frame[row * 16 + col] = 200;
            }
        }

        deblock_vertical_edge(&mut frame, 16, 8, 0, 4, 30, BoundaryStrength::Bs2);

        // Pixels near the edge should be smoothed
        // Column 7 (p side) should increase, column 8 (q side) should decrease
        assert!(frame[7] > 50, "p[0] should be smoothed up: {}", frame[7]);
        assert!(frame[8] < 200, "q[0] should be smoothed down: {}", frame[8]);
    }

    #[test]
    fn sao_band_offset() {
        // 4x4 block with value 100 (band = 100/8 = 12)
        let mut frame = vec![100u8; 4 * 4];
        let offsets = [5, 10, -5, -10i16];
        // band_position = 12 → bands 12,13,14,15 get offsets
        apply_sao_band(&mut frame, 4, 0, 0, 4, 4, &offsets, 12);
        // Sample value 100, band 12, offset index 0 → +5 = 105
        assert_eq!(frame[0], 105);
    }

    #[test]
    fn sao_band_offset_no_match() {
        // Value 200 (band = 25), band_position = 12 → band_offset = 13, not < 4 → no change
        let mut frame = vec![200u8; 4 * 4];
        let offsets = [5, 10, -5, -10i16];
        apply_sao_band(&mut frame, 4, 0, 0, 4, 4, &offsets, 12);
        assert_eq!(frame[0], 200);
    }

    #[test]
    fn sao_edge_offset_horizontal() {
        // 8x1 row with a valley: ..100, 100, 50, 100, 100..
        let mut frame = vec![100u8; 8];
        frame[2] = 50; // valley

        apply_sao_edge(
            &mut frame,
            8,
            1,
            0,
            6,
            1,
            &[10, 5, -5, -10],
            EdgeClass::Horizontal,
        );

        // Sample at x=2 (value 50): neighbors are 100 and 100
        // sign(50-100) + sign(50-100) = -1 + -1 = -2 → category 0 → offset[0] = +10
        assert_eq!(frame[2], 60, "valley should get positive offset");
    }

    #[test]
    fn sao_edge_offset_vertical() {
        // 1x8 column with a peak
        let mut frame = vec![100u8; 8];
        frame[2] = 200; // peak

        apply_sao_edge(
            &mut frame,
            1,
            0,
            1,
            1,
            6,
            &[10, 5, -5, -10],
            EdgeClass::Vertical,
        );

        // Sample at y=2 (value 200): neighbors at y=1 (100) and y=3 (100)
        // sign(200-100) + sign(200-100) = 1+1 = 2 → category 3 → offset[3] = -10
        assert_eq!(frame[2], 190, "peak should get negative offset");
    }

    #[test]
    fn sao_params_none() {
        let mut frame = vec![128u8; 16];
        let params = SaoParams::default();
        apply_sao(&mut frame, 4, 0, 0, 4, 4, &params);
        assert_eq!(frame[0], 128); // No change
    }

    #[test]
    fn edge_class_offsets() {
        assert_eq!(EdgeClass::Horizontal.offsets(), [(-1, 0), (1, 0)]);
        assert_eq!(EdgeClass::Vertical.offsets(), [(0, -1), (0, 1)]);
        assert_eq!(EdgeClass::Diagonal45.offsets(), [(1, -1), (-1, 1)]);
        assert_eq!(EdgeClass::Diagonal135.offsets(), [(-1, -1), (1, 1)]);
    }

    #[test]
    fn determinism_deblock() {
        let make_frame = || {
            let mut f = vec![0u8; 16 * 4];
            for row in 0..4 {
                for col in 0..8 {
                    f[row * 16 + col] = 50;
                }
                for col in 8..16 {
                    f[row * 16 + col] = 200;
                }
            }
            f
        };

        let mut f1 = make_frame();
        let mut f2 = make_frame();
        deblock_vertical_edge(&mut f1, 16, 8, 0, 4, 30, BoundaryStrength::Bs2);
        deblock_vertical_edge(&mut f2, 16, 8, 0, 4, 30, BoundaryStrength::Bs2);
        assert_eq!(f1, f2);
    }
}
