//! VP8 segmentation — adaptive per-MB quantization via segment map.
//!
//! Classifies macroblocks into up to 4 segments by spatial activity (variance).
//! High-detail regions get lower QP (better quality), smooth regions get higher QP
//! (more compression). This is the VP8 equivalent of adaptive quantization.

use crate::quant::{self, SegmentQuant};

/// Number of segments to use. VP8 supports up to 4.
pub const NUM_SEGMENTS: usize = 4;

/// Segment map: per-MB segment IDs and per-segment QP configuration.
pub struct SegmentMap {
    /// Segment ID (0-3) for each macroblock, row-major order.
    pub ids: Vec<u8>,
    /// Per-segment quantization parameters.
    pub quants: [SegmentQuant; NUM_SEGMENTS],
    /// QP delta for each segment relative to base QP (for bitstream header).
    pub qp_deltas: [i8; NUM_SEGMENTS],
    /// Base QP index.
    pub base_qp: u8,
    /// Number of MB columns.
    pub mb_w: usize,
    /// Whether segmentation should be signaled in the bitstream.
    /// False for tiny images or uniform content where overhead isn't worth it.
    pub enabled: bool,
}

/// Compute per-MB spatial activity (variance of luma pixels).
///
/// Returns a vector of variance values, one per macroblock, in row-major order.
fn compute_mb_activity(y_plane: &[u8], stride: usize, mb_w: usize, mb_h: usize) -> Vec<u32> {
    let mut activities = Vec::with_capacity(mb_w * mb_h);

    for mb_row in 0..mb_h {
        for mb_col in 0..mb_w {
            let base_y = mb_row * 16;
            let base_x = mb_col * 16;

            let mut sum = 0u32;
            let mut sum_sq = 0u64;

            for row in 0..16 {
                let y = base_y + row;
                for col in 0..16 {
                    let x = base_x + col;
                    let v = y_plane[y * stride + x] as u32;
                    sum += v;
                    sum_sq += (v as u64) * (v as u64);
                }
            }

            // Variance = E[X^2] - E[X]^2, scaled by 256 (16x16 pixels)
            let mean_sq = (sum as u64 * sum as u64) / 256;
            let variance = (sum_sq.saturating_sub(mean_sq) / 256) as u32;
            activities.push(variance);
        }
    }

    activities
}

/// Build a segment map by classifying MBs into 4 segments by activity level.
///
/// Segments are assigned using percentile thresholds:
/// - Segment 0: lowest activity (smooth) — highest QP delta (+delta)
/// - Segment 1: low-medium activity — slight QP increase (+delta/3)
/// - Segment 2: medium-high activity — base QP (delta 0)
/// - Segment 3: highest activity (detailed) — lowest QP delta (-delta)
///
/// For very small images (< 8 MBs) or uniform content, returns a flat map
/// with all MBs in segment 0 and no QP deltas (segmentation disabled).
pub fn compute_segment_map(
    y_plane: &[u8],
    stride: usize,
    mb_w: usize,
    mb_h: usize,
    base_qp: u8,
) -> SegmentMap {
    let total_mbs = mb_w * mb_h;

    // For very small images, segmentation overhead isn't worth it
    if total_mbs < 8 {
        return flat_segment_map(total_mbs, mb_w, base_qp);
    }

    let activities = compute_mb_activity(y_plane, stride, mb_w, mb_h);

    // Find percentile thresholds
    let mut sorted = activities.clone();
    sorted.sort_unstable();
    let n = sorted.len();

    let p25 = sorted[n / 4];
    let p50 = sorted[n / 2];
    let p75 = sorted[n * 3 / 4];

    // If all MBs have the same activity, segmentation is pointless
    if p25 == p75 {
        return flat_segment_map(total_mbs, mb_w, base_qp);
    }

    // Assign segments based on activity percentiles
    let ids: Vec<u8> = activities
        .iter()
        .map(|&a| {
            if a <= p25 {
                0 // smooth — boost QP
            } else if a <= p50 {
                1 // low-medium
            } else if a <= p75 {
                2 // medium-high — base QP
            } else {
                3 // high detail — lower QP
            }
        })
        .collect();

    // QP delta range scales with base QP. At low QP (high quality), small deltas.
    // At high QP (low quality), larger deltas for more adaptive range.
    let delta_range = ((base_qp as i32) / 5).clamp(2, 15) as i8;

    let qp_deltas = [
        delta_range,     // segment 0: smooth → higher QP (more compression)
        delta_range / 3, // segment 1: slight increase
        0,               // segment 2: base QP
        -delta_range,    // segment 3: detailed → lower QP (better quality)
    ];

    // Build per-segment SegmentQuant with clamped QPs
    let quants = std::array::from_fn(|i| {
        let qp = (base_qp as i16 + qp_deltas[i] as i16).clamp(0, 127) as u8;
        quant::build_segment_quant(qp)
    });

    SegmentMap {
        ids,
        quants,
        qp_deltas,
        base_qp,
        mb_w,
        enabled: true,
    }
}

/// Create a flat segment map with all MBs in segment 0 (no adaptive QP).
fn flat_segment_map(total_mbs: usize, mb_w: usize, base_qp: u8) -> SegmentMap {
    let q = quant::build_segment_quant(base_qp);
    SegmentMap {
        ids: vec![0; total_mbs],
        quants: [q.clone(), q.clone(), q.clone(), q],
        qp_deltas: [0; 4],
        base_qp,
        mb_w,
        enabled: false,
    }
}

impl SegmentMap {
    /// Get the segment ID for a macroblock at (col, row).
    #[inline]
    pub fn get(&self, mb_col: usize, mb_row: usize) -> u8 {
        self.ids[mb_row * self.mb_w + mb_col]
    }

    /// Get the SegmentQuant for a given segment ID.
    #[inline]
    pub fn quant(&self, segment_id: u8) -> &SegmentQuant {
        &self.quants[segment_id as usize]
    }
}
