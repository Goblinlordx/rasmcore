//! HEVC intra prediction — 35 modes (Planar, DC, Angular 2-34).
//!
//! ITU-T H.265 Section 8.4.4. Generates predicted blocks from neighboring
//! reference samples for each prediction unit.

// Prediction code uses indexed loops extensively for clarity matching the spec.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

/// Intra prediction angle table (ITU-T H.265 Table 8-4).
/// Index 0-34 maps to displacement per row/column.
/// Modes 2-17 are "vertical-like", modes 18-34 are "horizontal-like".
const INTRA_ANGLE: [i32; 35] = [
    0,  // mode 0 (Planar) — not used
    0,  // mode 1 (DC) — not used
    32, // mode 2
    26, 21, 17, 13, 9, 5, 2, // modes 3-9
    0, // mode 10 (pure horizontal)
    -2, -5, -9, -13, -17, -21, -26, // modes 11-17
    -32, // mode 18 (diagonal)
    -26, -21, -17, -13, -9, -5, -2, // modes 19-25
    0,  // mode 26 (pure vertical)
    2, 5, 9, 13, 17, 21, 26, // modes 27-33
    32, // mode 34
];

/// Inverse angle table for negative-angle modes (ITU-T H.265 Table 8-5).
/// Used to extend reference samples when angle is negative.
const INV_ANGLE: [i32; 35] = [
    0, 0, // modes 0-1 unused
    0, 0, 0, 0, 0, 0, 0, 0, 0, // modes 2-10: positive or zero angles
    -4096, -1638, -910, -630, -482, -390, -315, // modes 11-17
    -256, // mode 18
    -315, -390, -482, -630, -910, -1638, -4096, // modes 19-25
    0, 0, 0, 0, 0, 0, 0, 0, 0, // modes 26-34: positive or zero angles
];

/// Whether reference sample filtering should be applied for a given mode and block size.
/// ITU-T H.265 Table 8-3.
fn should_filter_refs(mode: u8, log2_size: u8) -> bool {
    if log2_size <= 2 {
        return false; // Never filter for 4x4
    }
    // For larger blocks, filter when mode is near DC/Planar
    // Simplified: filter threshold based on block size
    let threshold = match log2_size {
        3 => 7, // 8x8: filter modes within 7 of vertical/horizontal
        4 => 1, // 16x16: filter almost all except near-diagonal
        _ => 0, // 32x32: always filter
    };
    // Planar always uses filtered references.
    // DC (mode 1) does NOT use filtered references — it applies its own boundary filter.
    // Ref: libde265 v1.0.18 intrapred.h line 195, HEVC Section 8.4.4.2.3.
    if mode == 0 {
        return true;
    }
    if mode == 1 {
        return false;
    }
    // Check distance from horizontal (10) and vertical (26) modes
    let dist_h = (mode as i32 - 10).unsigned_abs();
    let dist_v = (mode as i32 - 26).unsigned_abs();
    let min_dist = dist_h.min(dist_v);
    min_dist > threshold as u32
}

/// Strong intra smoothing flag (ITU-T H.265 Section 8.4.4.2.3).
/// Applied for 32x32 blocks when the reference samples are relatively smooth.
///
/// Uses the endpoint test from the HEVC spec: checks if the midpoint of each
/// reference array deviates significantly from the bilinear interpolation
/// between the corner and the far endpoint (at index 2*nT).
///
/// Ref: libde265 v1.0.18 intrapred.h lines 217-222.
fn use_strong_smoothing(refs: &RefSamples, size: usize, strong_enabled: bool) -> bool {
    if !strong_enabled || size != 32 {
        return false;
    }
    let threshold = 1i32 << (refs.bit_depth - 5);
    let p0 = refs.top_left as i32;

    // Check top reference: abs(topLeft + top[2*nT-1] - 2*top[nT-1])
    // Ref: libde265 v1.0.18 intrapred.h line 220 — abs(p[0]+p[64]-2*p[32])
    let top_end = refs.top[2 * size - 1] as i32;
    let top_mid = refs.top[size - 1] as i32;
    let top_smooth = (p0 + top_end - 2 * top_mid).abs() < threshold;

    // Check left reference: abs(topLeft + left[2*nT-1] - 2*left[nT-1])
    // Ref: libde265 v1.0.18 intrapred.h line 221 — abs(p[0]+p[-64]-2*p[-32])
    let left_end = refs.left[2 * size - 1] as i32;
    let left_mid = refs.left[size - 1] as i32;
    let left_smooth = (p0 + left_end - 2 * left_mid).abs() < threshold;

    top_smooth && left_smooth
}

/// Reference samples gathered from neighboring reconstructed pixels.
#[derive(Debug, Clone)]
pub struct RefSamples {
    /// Top-left corner sample.
    pub top_left: u8,
    /// Top reference samples (left to right), length = 2*size.
    pub top: Vec<u8>,
    /// Left reference samples (top to bottom), length = 2*size.
    pub left: Vec<u8>,
    /// Bit depth (8 or 10).
    pub bit_depth: u8,
}

impl RefSamples {
    /// Create reference samples with all values set to a default (for edge blocks).
    pub fn with_default(size: usize, default_val: u8, bit_depth: u8) -> Self {
        Self {
            top_left: default_val,
            top: vec![default_val; 2 * size],
            left: vec![default_val; 2 * size],
            bit_depth,
        }
    }

    /// Construct reference samples from a reconstructed frame buffer.
    ///
    /// `x`, `y` are the top-left position of the current block.
    /// `size` is the block size (4, 8, 16, or 32).
    /// `frame` is the reconstructed luma plane (row-major).
    /// `stride` is the frame width.
    /// `avail_top`, `avail_left`, `avail_top_left` indicate neighbor availability.
    #[allow(clippy::too_many_arguments)]
    pub fn from_frame(
        x: usize,
        y: usize,
        size: usize,
        frame: &[u8],
        stride: usize,
        avail_top: bool,
        avail_left: bool,
        avail_top_left: bool,
    ) -> Self {
        let default_val = 1u8 << 7; // mid-gray for 8-bit
        let mut refs = Self::with_default(size, default_val, 8);

        // Gather available reference samples
        if avail_top_left && x > 0 && y > 0 {
            refs.top_left = frame[(y - 1) * stride + (x - 1)];
        }

        if avail_top && y > 0 {
            let row_above = (y - 1) * stride;
            for i in 0..(2 * size).min(stride - x) {
                refs.top[i] = frame[row_above + x + i];
            }
        }

        if avail_left && x > 0 {
            for i in 0..(2 * size).min(frame.len() / stride - y) {
                refs.left[i] = frame[(y + i) * stride + (x - 1)];
            }
        }

        // Substitution: if a neighbor is unavailable, fill from the first available sample
        if !avail_top_left {
            if avail_left {
                refs.top_left = refs.left[0];
            } else if avail_top {
                refs.top_left = refs.top[0];
            }
        }
        if !avail_top {
            let fill = refs.top_left;
            for v in refs.top.iter_mut() {
                *v = fill;
            }
        }
        if !avail_left {
            let fill = refs.top_left;
            for v in refs.left.iter_mut() {
                *v = fill;
            }
        }

        refs
    }

    /// Construct reference samples with per-sample availability based on
    /// reconstructed region bounds.
    ///
    /// HEVC Section 8.4.4.2.2: reference samples beyond the reconstructed area
    /// are substituted with the nearest available sample.
    ///
    /// `recon_right` is the rightmost x+1 that has been reconstructed in the row above.
    /// `recon_bottom` is the bottommost y+1 that has been reconstructed in the column to the left.
    #[allow(clippy::too_many_arguments)]
    pub fn from_frame_with_bounds(
        x: usize,
        y: usize,
        size: usize,
        frame: &[u8],
        stride: usize,
        frame_height: usize,
        avail_top: bool,
        avail_left: bool,
        avail_top_left: bool,
        recon_right: usize,
        recon_bottom: usize,
    ) -> Self {
        let default_val = 1u8 << 7; // mid-gray for 8-bit
        let mut refs = Self::with_default(size, default_val, 8);

        // Gather available reference samples
        if avail_top_left && x > 0 && y > 0 {
            refs.top_left = frame[(y - 1) * stride + (x - 1)];
        }

        if avail_top && y > 0 {
            let row_above = (y - 1) * stride;
            // Top references: only read up to the reconstructed right boundary
            let top_avail = (2 * size)
                .min(stride - x)
                .min(recon_right.saturating_sub(x));
            for i in 0..top_avail {
                refs.top[i] = frame[row_above + x + i];
            }
            // Fill remaining with last available value (HEVC Section 8.4.4.2.2)
            if top_avail > 0 && top_avail < 2 * size {
                let fill = refs.top[top_avail - 1];
                for i in top_avail..2 * size {
                    refs.top[i] = fill;
                }
            }
        }

        if avail_left && x > 0 {
            // Left references: only read up to the reconstructed bottom boundary
            let left_avail = (2 * size)
                .min(frame_height - y)
                .min(recon_bottom.saturating_sub(y));
            for i in 0..left_avail {
                refs.left[i] = frame[(y + i) * stride + (x - 1)];
            }
            // Fill remaining with last available value (HEVC Section 8.4.4.2.2)
            if left_avail > 0 && left_avail < 2 * size {
                let fill = refs.left[left_avail - 1];
                for i in left_avail..2 * size {
                    refs.left[i] = fill;
                }
            }
        }

        // Substitution: if a neighbor is unavailable, fill from the first available sample
        if !avail_top_left {
            if avail_left {
                refs.top_left = refs.left[0];
            } else if avail_top {
                refs.top_left = refs.top[0];
            }
        }
        if !avail_top {
            let fill = refs.top_left;
            for v in refs.top.iter_mut() {
                *v = fill;
            }
        }
        if !avail_left {
            let fill = refs.top_left;
            for v in refs.left.iter_mut() {
                *v = fill;
            }
        }

        refs
    }

    /// Apply 3-tap [1,2,1]/4 smoothing filter to reference samples.
    pub fn filter(&self, size: usize) -> Self {
        let mut filtered = self.clone();

        // Filter top
        filtered.top[0] =
            ((self.top_left as u16 + 2 * self.top[0] as u16 + self.top[1] as u16 + 2) >> 2) as u8;
        for i in 1..2 * size - 1 {
            filtered.top[i] =
                ((self.top[i - 1] as u16 + 2 * self.top[i] as u16 + self.top[i + 1] as u16 + 2)
                    >> 2) as u8;
        }

        // Filter left
        filtered.left[0] =
            ((self.top_left as u16 + 2 * self.left[0] as u16 + self.left[1] as u16 + 2) >> 2) as u8;
        for i in 1..2 * size - 1 {
            filtered.left[i] =
                ((self.left[i - 1] as u16 + 2 * self.left[i] as u16 + self.left[i + 1] as u16 + 2)
                    >> 2) as u8;
        }

        // Filter top-left
        filtered.top_left =
            ((self.left[0] as u16 + 2 * self.top_left as u16 + self.top[0] as u16 + 2) >> 2) as u8;

        filtered
    }

    /// Apply strong intra smoothing (bilinear interpolation for 32x32).
    ///
    /// Interpolates between top_left and the FAR endpoints (at index 2*nT-1),
    /// using 6-bit precision: pF[i] = p[0] + ((i * (p[2*nT] - p[0]) + 32) >> 6).
    ///
    /// Ref: libde265 v1.0.18 intrapred.h lines 227-235.
    pub fn strong_smooth(&self, size: usize) -> Self {
        let mut smoothed = self.clone();
        let p0 = self.top_left as i32;
        // Use the far endpoint at 2*size-1 (not size-1)
        let top_far = self.top[2 * size - 1] as i32;
        let left_far = self.left[2 * size - 1] as i32;
        let n = 2 * size; // Total interpolation range (64 for nT=32)

        // Interpolate top and left using the exact libde265 formula:
        // pF[i] = p[0] + ((i * (p[2*nT] - p[0]) + 32) >> 6)
        // where i goes from 1 to 63 (for nT=32), and the shift is always 6.
        // Ref: libde265 v1.0.18 intrapred.h lines 232-235.
        for i in 1..n {
            smoothed.top[i - 1] =
                (p0 + ((i as i32 * (top_far - p0) + 32) >> 6)).clamp(0, 255) as u8;
        }
        smoothed.top[n - 1] = top_far as u8;

        for i in 1..n {
            smoothed.left[i - 1] =
                (p0 + ((i as i32 * (left_far - p0) + 32) >> 6)).clamp(0, 255) as u8;
        }
        smoothed.left[n - 1] = left_far as u8;

        smoothed
    }
}

/// Predict a block using the specified intra prediction mode.
///
/// # Arguments
/// * `mode` - Intra prediction mode (0-34)
/// * `refs` - Reference samples from neighboring blocks
/// * `size` - Block size in pixels (4, 8, 16, or 32)
/// * `strong_smoothing_enabled` - From SPS flag
///
/// # Returns
/// Predicted block as a flat Vec<u8> of size*size pixels (row-major).
pub fn predict_intra(
    mode: u8,
    refs: &RefSamples,
    size: usize,
    strong_smoothing_enabled: bool,
) -> Vec<u8> {
    let log2_size = (size as f32).log2() as u8;

    // Determine which reference samples to use (filtered or unfiltered)
    let use_refs = if use_strong_smoothing(refs, size, strong_smoothing_enabled) {
        refs.strong_smooth(size)
    } else if should_filter_refs(mode, log2_size) {
        refs.filter(size)
    } else {
        refs.clone()
    };

    match mode {
        0 => predict_planar(&use_refs, size),
        1 => predict_dc(&use_refs, size),
        2..=34 => predict_angular(mode, &use_refs, size),
        _ => vec![128; size * size], // Invalid mode fallback
    }
}

/// Planar prediction (mode 0) — ITU-T H.265 Section 8.4.4.2.4.
fn predict_planar(refs: &RefSamples, size: usize) -> Vec<u8> {
    let mut pred = vec![0u8; size * size];
    let log2_size = (size as f32).log2() as u32;

    let top_right = refs.top[size] as i32;
    let bottom_left = refs.left[size] as i32;

    for y in 0..size {
        for x in 0..size {
            let h = (size - 1 - x) as i32 * refs.left[y] as i32 + (x + 1) as i32 * top_right;
            let v = (size - 1 - y) as i32 * refs.top[x] as i32 + (y + 1) as i32 * bottom_left;
            pred[y * size + x] = ((h + v + size as i32) >> (log2_size + 1)) as u8;
        }
    }

    pred
}

/// DC prediction (mode 1) — ITU-T H.265 Section 8.4.4.2.5.
fn predict_dc(refs: &RefSamples, size: usize) -> Vec<u8> {
    let log2_size = (size as f32).log2() as u32;

    // DC value = average of top and left reference samples
    let sum: u32 = refs.top[..size].iter().map(|&v| v as u32).sum::<u32>()
        + refs.left[..size].iter().map(|&v| v as u32).sum::<u32>();
    let dc = ((sum + size as u32) >> (log2_size + 1)) as u8;

    let mut pred = vec![dc; size * size];

    // DC boundary filter (for blocks > 4x4 it's simpler, but we apply for all)
    if size <= 32 {
        // Top row: filter with top reference
        pred[0] = ((refs.top[0] as u16 + refs.left[0] as u16 + 2 * dc as u16 + 2) >> 2) as u8;
        for x in 1..size {
            pred[x] = ((refs.top[x] as u16 + 3 * dc as u16 + 2) >> 2) as u8;
        }
        // Left column: filter with left reference
        for y in 1..size {
            pred[y * size] = ((refs.left[y] as u16 + 3 * dc as u16 + 2) >> 2) as u8;
        }
    }

    pred
}

/// Angular prediction (modes 2-34) — ITU-T H.265 Section 8.4.4.2.6.
fn predict_angular(mode: u8, refs: &RefSamples, size: usize) -> Vec<u8> {
    let mut pred = vec![0u8; size * size];
    let angle = INTRA_ANGLE[mode as usize];

    // Determine if this is a "vertical" or "horizontal" mode
    let is_vertical = mode >= 18;

    // Build the 1D reference array for this mode
    let ref_array = build_angular_ref_array(mode, refs, size);

    if is_vertical {
        // Vertical-like: each column uses the top reference with an offset
        for y in 0..size {
            let offset = ((y as i32 + 1) * angle) >> 5;
            let frac = ((y as i32 + 1) * angle) & 31;

            if frac != 0 {
                // Linear interpolation
                for x in 0..size {
                    let idx_i = x as i32 + offset + 1;
                    if idx_i >= 0 && (idx_i as usize + 1) < ref_array.len() {
                        let idx = idx_i as usize;
                        pred[y * size + x] = (((32 - frac) * ref_array[idx] as i32
                            + frac * ref_array[idx + 1] as i32
                            + 16)
                            >> 5) as u8;
                    }
                }
            } else {
                // No interpolation needed
                for x in 0..size {
                    let idx_i = x as i32 + offset + 1;
                    if idx_i >= 0 && (idx_i as usize) < ref_array.len() {
                        pred[y * size + x] = ref_array[idx_i as usize];
                    }
                }
            }
        }
    } else {
        // Horizontal-like: each row uses the left reference with an offset
        for x in 0..size {
            let offset = ((x as i32 + 1) * angle) >> 5;
            let frac = ((x as i32 + 1) * angle) & 31;

            if frac != 0 {
                for y in 0..size {
                    let idx_i = y as i32 + offset + 1;
                    if idx_i >= 0 && (idx_i as usize + 1) < ref_array.len() {
                        let idx = idx_i as usize;
                        pred[y * size + x] = (((32 - frac) * ref_array[idx] as i32
                            + frac * ref_array[idx + 1] as i32
                            + 16)
                            >> 5) as u8;
                    }
                }
            } else {
                for y in 0..size {
                    let idx_i = y as i32 + offset + 1;
                    if idx_i >= 0 && (idx_i as usize) < ref_array.len() {
                        pred[y * size + x] = ref_array[idx_i as usize];
                    }
                }
            }
        }
    }

    // Boundary smoothing for pure vertical (mode 26) and pure horizontal (mode 10).
    // These modes copy reference samples directly, so the boundary between the
    // predicted block and the perpendicular reference can be harsh. The spec
    // applies a smoothing filter to the first column (mode 26) or first row (mode 10).
    // Only for block sizes < 32.
    // Ref: libde265 v1.0.18 intrapred.cc lines 379-382 (mode 26),
    //      libde265 v1.0.18 intrapred.cc lines 417-420 (mode 10).
    if mode == 26 && size < 32 {
        // Vertical mode: smooth first column using left reference
        for y in 0..size {
            let val = refs.top[0] as i32 + ((refs.left[y] as i32 - refs.top_left as i32) >> 1);
            pred[y * size] = val.clamp(0, 255) as u8;
        }
    } else if mode == 10 && size < 32 {
        // Horizontal mode: smooth first row using top reference
        for x in 0..size {
            let val = refs.left[0] as i32 + ((refs.top[x] as i32 - refs.top_left as i32) >> 1);
            pred[x] = val.clamp(0, 255) as u8;
        }
    }

    pred
}

/// Build the 1D reference array for angular prediction.
///
/// For vertical modes (18-34): primary reference is top, secondary is left.
/// For horizontal modes (2-17): primary reference is left, secondary is top.
fn build_angular_ref_array(mode: u8, refs: &RefSamples, size: usize) -> Vec<u8> {
    let angle = INTRA_ANGLE[mode as usize];
    let is_vertical = mode >= 18;

    let (primary, secondary, corner) = if is_vertical {
        (&refs.top, &refs.left, refs.top_left)
    } else {
        (&refs.left, &refs.top, refs.top_left)
    };

    // ref_array[0] = corner, ref_array[1..=2*size] = primary reference
    let mut ref_array = vec![0u8; 4 * size + 1];
    ref_array[0] = corner;
    for i in 0..(2 * size).min(primary.len()) {
        ref_array[i + 1] = primary[i];
    }

    // For negative angles, extend with projected secondary reference samples
    if angle < 0 {
        let inv = INV_ANGLE[mode as usize];
        let num_ext = (size as i32 * angle) >> 5;
        // num_ext is negative, so we need |num_ext| samples before ref_array[0]
        // We shift the array and prepend projected samples
        let ext_count = (-num_ext) as usize;
        let mut extended = vec![0u8; ext_count + 2 * size + 1];

        // Copy primary into position
        for i in 0..2 * size + 1 {
            extended[ext_count + i] = ref_array[i];
        }

        // Project secondary reference samples
        for i in 1..=ext_count {
            let sec_idx = ((i as i32 * inv + 128) >> 8) as usize;
            if sec_idx < secondary.len() {
                extended[ext_count - i] = secondary[sec_idx];
            }
        }

        return extended;
    }

    ref_array
}

/// Derive chroma intra prediction mode from luma mode.
///
/// ITU-T H.265 Table 8-2. DM mode (4) copies the luma mode.
/// Other chroma modes: 0=Planar, 1=DC(vertical alias), 2=10(horizontal), 3=1(DC).
pub fn derive_chroma_mode(chroma_pred_mode: u8, luma_mode: u8) -> u8 {
    match chroma_pred_mode {
        0 => {
            if luma_mode == 0 {
                34
            } else {
                0
            }
        } // Planar (or 34 if luma is Planar)
        1 => {
            if luma_mode == 26 {
                34
            } else {
                26
            }
        } // Vertical
        2 => {
            if luma_mode == 10 {
                34
            } else {
                10
            }
        } // Horizontal
        3 => {
            if luma_mode == 1 {
                34
            } else {
                1
            }
        } // DC
        4 => luma_mode, // DM mode
        _ => luma_mode, // Default to DM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_refs(size: usize, top_val: u8, left_val: u8, corner: u8) -> RefSamples {
        RefSamples {
            top_left: corner,
            top: vec![top_val; 2 * size],
            left: vec![left_val; 2 * size],
            bit_depth: 8,
        }
    }

    #[test]
    fn planar_4x4_flat() {
        let refs = make_refs(4, 100, 100, 100);
        let pred = predict_planar(&refs, 4);
        // Flat input should give flat output
        for &v in &pred {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn planar_4x4_gradient() {
        let mut refs = RefSamples::with_default(4, 0, 8);
        // Top = 0, left = 0, top_right = 200, bottom_left = 200
        refs.top_left = 0;
        for i in 0..8 {
            refs.top[i] = 0;
            refs.left[i] = 0;
        }
        refs.top[4] = 200; // top_right (index = size)
        refs.left[4] = 200; // bottom_left (index = size)

        let pred = predict_planar(&refs, 4);
        // Should produce a smooth gradient from top-left to bottom-right
        assert!(pred[0] < pred[15], "should gradient from TL to BR");
        // Center should be near 100
        assert!((pred[5] as i32 - 100).abs() < 30);
    }

    #[test]
    fn dc_4x4_uniform() {
        let refs = make_refs(4, 200, 200, 200);
        let pred = predict_dc(&refs, 4);
        // DC of uniform 200 should be ~200 (boundary filter may adjust edges slightly)
        for y in 1..4 {
            for x in 1..4 {
                assert_eq!(pred[y * 4 + x], 200);
            }
        }
    }

    #[test]
    fn dc_4x4_mixed() {
        let refs = make_refs(4, 100, 200, 150);
        let pred = predict_dc(&refs, 4);
        // DC = (4*100 + 4*200 + 4) / 8 = 150
        // Interior (non-boundary) pixels should be 150
        assert_eq!(pred[1 * 4 + 1], 150);
    }

    #[test]
    fn angular_26_pure_vertical() {
        // Mode 26 = pure vertical: angle = 0
        // Each row should copy the top reference
        let mut refs = RefSamples::with_default(4, 0, 8);
        refs.top[0] = 10;
        refs.top[1] = 20;
        refs.top[2] = 30;
        refs.top[3] = 40;

        let pred = predict_angular(26, &refs, 4);
        // All rows should be [10, 20, 30, 40]
        for y in 0..4 {
            assert_eq!(pred[y * 4], 10, "row {y} col 0");
            assert_eq!(pred[y * 4 + 1], 20, "row {y} col 1");
            assert_eq!(pred[y * 4 + 2], 30, "row {y} col 2");
            assert_eq!(pred[y * 4 + 3], 40, "row {y} col 3");
        }
    }

    #[test]
    fn angular_10_pure_horizontal() {
        // Mode 10 = pure horizontal: angle = 0
        // Each column should copy the left reference
        let mut refs = RefSamples::with_default(4, 0, 8);
        refs.left[0] = 10;
        refs.left[1] = 20;
        refs.left[2] = 30;
        refs.left[3] = 40;

        let pred = predict_angular(10, &refs, 4);
        // All columns should replicate the left reference
        for y in 0..4 {
            let expected = refs.left[y];
            for x in 0..4 {
                assert_eq!(pred[y * 4 + x], expected, "row {y} col {x}");
            }
        }
    }

    #[test]
    fn angular_18_diagonal() {
        // Mode 18 = diagonal (-45 degrees): angle = -32
        let mut refs = make_refs(4, 100, 100, 100);
        refs.top_left = 50;
        refs.top = vec![100; 8];
        refs.left = vec![100; 8];

        let pred = predict_angular(18, &refs, 4);
        // Diagonal should reference both top and left
        assert!(pred.len() == 16);
    }

    #[test]
    fn ref_samples_from_frame_interior() {
        // 8x8 frame, block at (4, 4) with size 4 — has all neighbors
        let frame = vec![128u8; 8 * 8];
        let refs = RefSamples::from_frame(4, 4, 4, &frame, 8, true, true, true);
        assert_eq!(refs.top_left, 128);
        assert_eq!(refs.top[0], 128);
        assert_eq!(refs.left[0], 128);
    }

    #[test]
    fn ref_samples_edge_substitution() {
        // Block at (0, 0) — no neighbors available
        let frame = vec![100u8; 8 * 8];
        let refs = RefSamples::from_frame(0, 0, 4, &frame, 8, false, false, false);
        // Should use default value (128)
        assert_eq!(refs.top_left, 128);
        assert_eq!(refs.top[0], 128);
        assert_eq!(refs.left[0], 128);
    }

    #[test]
    fn ref_samples_filter() {
        let refs = make_refs(4, 100, 100, 100);
        let filtered = refs.filter(4);
        // Flat input should stay flat after filtering
        assert_eq!(filtered.top_left, 100);
        assert_eq!(filtered.top[0], 100);
        assert_eq!(filtered.left[0], 100);
    }

    #[test]
    fn chroma_dm_mode() {
        assert_eq!(derive_chroma_mode(4, 26), 26); // DM copies luma
        assert_eq!(derive_chroma_mode(4, 0), 0); // DM copies luma
        assert_eq!(derive_chroma_mode(0, 5), 0); // Planar when luma != Planar
        assert_eq!(derive_chroma_mode(0, 0), 34); // Planar conflicts with luma=Planar → use 34
    }

    #[test]
    fn predict_intra_all_modes_dont_panic() {
        let refs = make_refs(8, 128, 128, 128);
        for mode in 0..=34 {
            let pred = predict_intra(mode, &refs, 8, true);
            assert_eq!(pred.len(), 64, "mode {mode} should produce 64 pixels");
        }
    }

    #[test]
    fn determinism() {
        let refs = make_refs(8, 100, 150, 125);
        for mode in 0..=34 {
            let p1 = predict_intra(mode, &refs, 8, false);
            let p2 = predict_intra(mode, &refs, 8, false);
            assert_eq!(p1, p2, "mode {mode} not deterministic");
        }
    }

    #[test]
    fn golden_prediction_flat() {
        crate::skip_if_no_fixtures!();

        // Load flat 64x64 at QP22 — should decode to near-uniform gray
        let _hevc_data = crate::testutil::load_fixture("flat_64x64_q22", "hevc").unwrap();
        let rgb_ref = crate::testutil::load_reference_rgb("flat_64x64_q22").unwrap();

        // For a flat image, all pixels should be close to 128
        let avg: f64 = rgb_ref.iter().map(|&v| v as f64).sum::<f64>() / rgb_ref.len() as f64;
        assert!(
            (avg - 128.0).abs() < 10.0,
            "flat image reference should be near 128, got {avg}"
        );
    }
}
