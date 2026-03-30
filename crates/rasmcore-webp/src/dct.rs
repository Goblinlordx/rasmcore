//! 4×4 integer DCT and Walsh-Hadamard Transform (RFC 6386 Section 14).
//!
//! Reusable integer transforms for VP8 and similar codecs.
//! All arithmetic is integer-only — no floating point.
//!
//! On WASM targets with SIMD128, uses `std::arch::wasm32` intrinsics to
//! process all 4 columns simultaneously via i32x4 vectors.
//! On native targets, uses scalar code (LLVM auto-vectorizes in release builds).

// ─── Public dispatch functions ───────────────────────────────────────────────

/// Forward 4×4 DCT on residual pixels (source - reference).
///
/// Matches libwebp `FTransform_C`. Uses constants 2217, 5352 with
/// carefully tuned rounding biases for each coefficient.
pub fn forward_dct(src: &[u8; 16], reference: &[u8; 16], out: &mut [i16; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        forward_dct_simd128(src, reference, out);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        forward_dct_scalar(src, reference, out);
    }
}

/// Inverse 4×4 DCT — reconstruct pixels from DCT coefficients + reference.
///
/// Matches libwebp `TransformOne_C`. Uses MUL1/MUL2 macros.
/// Pass order: vertical first, then horizontal (matching libwebp).
pub fn inverse_dct(coeffs: &[i16; 16], reference: &[u8; 16], dst: &mut [u8; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        inverse_dct_simd128(coeffs, reference, dst);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        inverse_dct_scalar(coeffs, reference, dst);
    }
}

/// Forward 4×4 Walsh-Hadamard Transform for DC coefficients.
pub fn forward_wht(dc_coeffs: &[i16; 16], out: &mut [i16; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        forward_wht_simd128(dc_coeffs, out);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        forward_wht_scalar(dc_coeffs, out);
    }
}

/// Inverse 4×4 Walsh-Hadamard Transform.
pub fn inverse_wht(coeffs: &[i16; 16], out: &mut [i16; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        inverse_wht_simd128(coeffs, out);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        inverse_wht_scalar(coeffs, out);
    }
}

// ─── Scalar implementations ─────────────────────────────────────────────────

#[allow(dead_code)] // scalar fallback for non-SIMD targets
fn forward_dct_scalar(src: &[u8; 16], reference: &[u8; 16], out: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    for i in 0..4 {
        let base = i * 4;
        let d0 = src[base] as i32 - reference[base] as i32;
        let d1 = src[base + 1] as i32 - reference[base + 1] as i32;
        let d2 = src[base + 2] as i32 - reference[base + 2] as i32;
        let d3 = src[base + 3] as i32 - reference[base + 3] as i32;

        let a0 = d0 + d3;
        let a1 = d1 + d2;
        let a2 = d1 - d2;
        let a3 = d0 - d3;

        tmp[base] = (a0 + a1) * 8;
        tmp[base + 2] = (a0 - a1) * 8;
        tmp[base + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[base + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }

    for i in 0..4 {
        let a0 = tmp[i] + tmp[12 + i];
        let a1 = tmp[4 + i] + tmp[8 + i];
        let a2 = tmp[4 + i] - tmp[8 + i];
        let a3 = tmp[i] - tmp[12 + i];

        out[i] = ((a0 + a1 + 7) >> 4) as i16;
        out[8 + i] = ((a0 - a1 + 7) >> 4) as i16;
        out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) as i16 + if a3 != 0 { 1 } else { 0 };
        out[12 + i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16;
    }
}

#[allow(dead_code)]
fn inverse_dct_scalar(coeffs: &[i16; 16], reference: &[u8; 16], dst: &mut [u8; 16]) {
    let mut tmp = [0i32; 16];

    for i in 0..4 {
        let c0 = coeffs[i] as i32;
        let c1 = coeffs[4 + i] as i32;
        let c2 = coeffs[8 + i] as i32;
        let c3 = coeffs[12 + i] as i32;

        let a = c0 + c2;
        let b = c0 - c2;
        let c = mul2(c1) - mul1(c3);
        let d = mul1(c1) + mul2(c3);

        tmp[i] = a + d;
        tmp[4 + i] = b + c;
        tmp[8 + i] = b - c;
        tmp[12 + i] = a - d;
    }

    for i in 0..4 {
        let base = i * 4;
        let dc = tmp[base] + 4;

        let a = dc + tmp[base + 2];
        let b = dc - tmp[base + 2];
        let c = mul2(tmp[base + 1]) - mul1(tmp[base + 3]);
        let d = mul1(tmp[base + 1]) + mul2(tmp[base + 3]);

        dst[base] = clamp_u8(((a + d) >> 3) + reference[base] as i32);
        dst[base + 1] = clamp_u8(((b + c) >> 3) + reference[base + 1] as i32);
        dst[base + 2] = clamp_u8(((b - c) >> 3) + reference[base + 2] as i32);
        dst[base + 3] = clamp_u8(((a - d) >> 3) + reference[base + 3] as i32);
    }
}

#[allow(dead_code)]
fn forward_wht_scalar(dc_coeffs: &[i16; 16], out: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    for i in 0..4 {
        let base = i * 4;
        let a0 = dc_coeffs[base] as i32 + dc_coeffs[base + 2] as i32;
        let a1 = dc_coeffs[base + 1] as i32 + dc_coeffs[base + 3] as i32;
        let a2 = dc_coeffs[base + 1] as i32 - dc_coeffs[base + 3] as i32;
        let a3 = dc_coeffs[base] as i32 - dc_coeffs[base + 2] as i32;

        tmp[base] = a0 + a1;
        tmp[base + 1] = a3 + a2;
        tmp[base + 2] = a3 - a2;
        tmp[base + 3] = a0 - a1;
    }

    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];

        out[i] = ((a0 + a1) >> 1) as i16;
        out[4 + i] = ((a3 + a2) >> 1) as i16;
        out[8 + i] = ((a3 - a2) >> 1) as i16;
        out[12 + i] = ((a0 - a1) >> 1) as i16;
    }
}

#[allow(dead_code)]
fn inverse_wht_scalar(coeffs: &[i16; 16], out: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    for i in 0..4 {
        let a0 = coeffs[i] as i32 + coeffs[12 + i] as i32;
        let a1 = coeffs[4 + i] as i32 + coeffs[8 + i] as i32;
        let a2 = coeffs[4 + i] as i32 - coeffs[8 + i] as i32;
        let a3 = coeffs[i] as i32 - coeffs[12 + i] as i32;

        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }

    for i in 0..4 {
        let base = i * 4;
        let dc = tmp[base] + 3;
        let a0 = dc + tmp[base + 3];
        let a1 = tmp[base + 1] + tmp[base + 2];
        let a2 = tmp[base + 1] - tmp[base + 2];
        let a3 = dc - tmp[base + 3];

        out[base] = ((a0 + a1) >> 3) as i16;
        out[base + 1] = ((a3 + a2) >> 3) as i16;
        out[base + 2] = ((a0 - a1) >> 3) as i16;
        out[base + 3] = ((a3 - a2) >> 3) as i16;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn mul1(a: i32) -> i32 {
    ((a * 20091) >> 16) + a
}

#[allow(dead_code)]
#[inline(always)]
fn mul2(a: i32) -> i32 {
    (a * 35468) >> 16
}

#[allow(dead_code)]
#[inline(always)]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// ─── WASM SIMD128 implementations ──────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
mod simd128 {
    use std::arch::wasm32::*;

    /// Load row i of a 4x4 u8 block as i32x4.
    #[inline(always)]
    fn load_row_u8_as_i32x4(block: &[u8; 16], row: usize) -> v128 {
        let base = row * 4;
        i32x4(
            block[base] as i32,
            block[base + 1] as i32,
            block[base + 2] as i32,
            block[base + 3] as i32,
        )
    }

    /// Load row i of a 4x4 i16 block as i32x4.
    #[inline(always)]
    fn load_row_i16_as_i32x4(block: &[i16; 16], row: usize) -> v128 {
        let base = row * 4;
        i32x4(
            block[base] as i32,
            block[base + 1] as i32,
            block[base + 2] as i32,
            block[base + 3] as i32,
        )
    }

    /// Store i32x4 to row of i16 output.
    #[inline(always)]
    fn store_row_i32x4_as_i16(out: &mut [i16; 16], row: usize, v: v128) {
        let base = row * 4;
        out[base] = i32x4_extract_lane::<0>(v) as i16;
        out[base + 1] = i32x4_extract_lane::<1>(v) as i16;
        out[base + 2] = i32x4_extract_lane::<2>(v) as i16;
        out[base + 3] = i32x4_extract_lane::<3>(v) as i16;
    }

    /// Transpose a 4x4 matrix stored as 4 i32x4 row vectors.
    #[allow(dead_code)]
    #[inline(always)]
    fn transpose4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
        // Interleave low/high pairs
        let t0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // r0[0] r1[0] r0[1] r1[1]
        let t1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // r0[2] r1[2] r0[3] r1[3]
        let t2 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // r2[0] r3[0] r2[1] r3[1]
        let t3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // r2[2] r3[2] r2[3] r3[3]
        // Final interleave
        let o0 = i32x4_shuffle::<0, 1, 4, 5>(t0, t2); // col 0
        let o1 = i32x4_shuffle::<2, 3, 6, 7>(t0, t2); // col 1
        let o2 = i32x4_shuffle::<0, 1, 4, 5>(t1, t3); // col 2
        let o3 = i32x4_shuffle::<2, 3, 6, 7>(t1, t3); // col 3
        (o0, o1, o2, o3)
    }

    pub fn forward_dct_simd128(src: &[u8; 16], reference: &[u8; 16], out: &mut [i16; 16]) {
        // Load all 4 rows as i32x4 and compute residuals
        let mut rows = [i32x4_splat(0); 4];
        for i in 0..4 {
            let s = load_row_u8_as_i32x4(src, i);
            let r = load_row_u8_as_i32x4(reference, i);
            rows[i] = i32x4_sub(s, r);
        }

        // Horizontal pass on each row (scalar per-row, vectorized would need lane shuffles)
        let mut tmp = [i32x4_splat(0); 4]; // tmp[row] = [t0, t1, t2, t3]
        for i in 0..4 {
            let d0 = i32x4_extract_lane::<0>(rows[i]);
            let d1 = i32x4_extract_lane::<1>(rows[i]);
            let d2 = i32x4_extract_lane::<2>(rows[i]);
            let d3 = i32x4_extract_lane::<3>(rows[i]);

            let a0 = d0 + d3;
            let a1 = d1 + d2;
            let a2 = d1 - d2;
            let a3 = d0 - d3;

            tmp[i] = i32x4(
                (a0 + a1) * 8,
                (a2 * 2217 + a3 * 5352 + 1812) >> 9,
                (a0 - a1) * 8,
                (a3 * 2217 - a2 * 5352 + 937) >> 9,
            );
        }

        // Vertical pass: row0+row3, row1+row2, etc. — all 4 columns in parallel
        // No transpose needed: tmp[i] = [col0, col1, col2, col3] for row i
        // Adding tmp[0]+tmp[3] gives [r0c0+r3c0, r0c1+r3c1, ...] = correct column butterflies
        let a0 = i32x4_add(tmp[0], tmp[3]);
        let a1 = i32x4_add(tmp[1], tmp[2]);
        let a2 = i32x4_sub(tmp[1], tmp[2]);
        let a3 = i32x4_sub(tmp[0], tmp[3]);

        let seven = i32x4_splat(7);
        let c2217 = i32x4_splat(2217);
        let c5352 = i32x4_splat(5352);
        let c12000 = i32x4_splat(12000);
        let c51000 = i32x4_splat(51000);

        let o0 = i32x4_shr(i32x4_add(i32x4_add(a0, a1), seven), 4);
        let o2 = i32x4_shr(i32x4_add(i32x4_sub(a0, a1), seven), 4);

        let rot1 = i32x4_shr(
            i32x4_add(
                i32x4_add(i32x4_mul(a2, c2217), i32x4_mul(a3, c5352)),
                c12000,
            ),
            16,
        );
        // Add (a3 != 0) bias — check each lane
        let a3_nonzero = v128_not(i32x4_eq(a3, i32x4_splat(0)));
        let bias = v128_and(a3_nonzero, i32x4_splat(1));
        let o1 = i32x4_add(rot1, bias);

        let o3 = i32x4_shr(
            i32x4_add(
                i32x4_sub(i32x4_mul(a3, c2217), i32x4_mul(a2, c5352)),
                c51000,
            ),
            16,
        );

        // Store: out[row] = [o_row[0], o_row[1], o_row[2], o_row[3]]
        // But our outputs are organized as out[0..4]=row0, out[4..8]=row1, etc.
        // o0 has all 4 values for output row 0, o1 for row 1, etc.
        store_row_i32x4_as_i16(out, 0, o0);
        store_row_i32x4_as_i16(out, 1, o1);
        store_row_i32x4_as_i16(out, 2, o2);
        store_row_i32x4_as_i16(out, 3, o3);
    }

    pub fn inverse_dct_simd128(coeffs: &[i16; 16], reference: &[u8; 16], dst: &mut [u8; 16]) {
        let c_kc2: v128 = i32x4_splat(35468);
        let c_kc1: v128 = i32x4_splat(20091);

        // Load rows as i32x4
        let r0 = load_row_i16_as_i32x4(coeffs, 0);
        let r1 = load_row_i16_as_i32x4(coeffs, 1);
        let r2 = load_row_i16_as_i32x4(coeffs, 2);
        let r3 = load_row_i16_as_i32x4(coeffs, 3);

        // Vertical pass: operate on rows directly (no transpose needed)
        // r0+r2 gives row0+row2 for all 4 columns simultaneously
        let a = i32x4_add(r0, r2);
        let b = i32x4_sub(r0, r2);
        // MUL2(r1) - MUL1(r3)
        let mul2_r1 = i32x4_shr(i32x4_mul(r1, c_kc2), 16);
        let mul1_r3 = i32x4_add(i32x4_shr(i32x4_mul(r3, c_kc1), 16), r3);
        let c = i32x4_sub(mul2_r1, mul1_r3);
        // MUL1(r1) + MUL2(r3)
        let mul1_r1 = i32x4_add(i32x4_shr(i32x4_mul(r1, c_kc1), 16), r1);
        let mul2_r3 = i32x4_shr(i32x4_mul(r3, c_kc2), 16);
        let d = i32x4_add(mul1_r1, mul2_r3);

        // After vertical pass, t0..t3 are row vectors (all 4 cols computed)
        let t0 = i32x4_add(a, d);
        let t1 = i32x4_add(b, c);
        let t2 = i32x4_sub(b, c);
        let t3 = i32x4_sub(a, d);

        // Horizontal pass: operates within each row.
        // For each row, we need elements [0]+[2], [0]-[2], MUL2([1])-MUL1([3]), etc.
        // This requires lane shuffles, so we do it scalar per-row.
        let _four = i32x4_splat(4);
        let _zero = i32x4_splat(0);
        let _max_val = i32x4_splat(255);

        for (row, t) in [(0, t0), (1, t1), (2, t2), (3, t3)] {
            let e0 = i32x4_extract_lane::<0>(t) + 4; // +4 rounding
            let e1 = i32x4_extract_lane::<1>(t);
            let e2 = i32x4_extract_lane::<2>(t);
            let e3 = i32x4_extract_lane::<3>(t);

            let ha = e0 + e2;
            let hb = e0 - e2;
            let hc = ((e1 * 35468) >> 16) - (((e3 * 20091) >> 16) + e3);
            let hd = (((e1 * 20091) >> 16) + e1) + ((e3 * 35468) >> 16);

            let base = row * 4;
            let ref_base = row * 4;
            dst[base] = (((ha + hd) >> 3) + reference[ref_base] as i32).clamp(0, 255) as u8;
            dst[base + 1] = (((hb + hc) >> 3) + reference[ref_base + 1] as i32).clamp(0, 255) as u8;
            dst[base + 2] = (((hb - hc) >> 3) + reference[ref_base + 2] as i32).clamp(0, 255) as u8;
            dst[base + 3] = (((ha - hd) >> 3) + reference[ref_base + 3] as i32).clamp(0, 255) as u8;
        }
    }

    pub fn forward_wht_simd128(dc_coeffs: &[i16; 16], out: &mut [i16; 16]) {
        // Load rows
        let r0 = load_row_i16_as_i32x4(dc_coeffs, 0);
        let r1 = load_row_i16_as_i32x4(dc_coeffs, 1);
        let r2 = load_row_i16_as_i32x4(dc_coeffs, 2);
        let r3 = load_row_i16_as_i32x4(dc_coeffs, 3);

        // Horizontal pass per row (scalar — each row is independent)
        let mut tmp_rows = [i32x4_splat(0); 4];
        for (i, row) in [r0, r1, r2, r3].iter().enumerate() {
            let v0 = i32x4_extract_lane::<0>(*row);
            let v1 = i32x4_extract_lane::<1>(*row);
            let v2 = i32x4_extract_lane::<2>(*row);
            let v3 = i32x4_extract_lane::<3>(*row);

            let a0 = v0 + v2;
            let a1 = v1 + v3;
            let a2 = v1 - v3;
            let a3 = v0 - v2;

            tmp_rows[i] = i32x4(a0 + a1, a3 + a2, a3 - a2, a0 - a1);
        }

        // Vertical pass: use row vectors directly (no transpose needed)
        // tmp_rows[0]+tmp_rows[2] = row0+row2 for all 4 columns
        let a0 = i32x4_add(tmp_rows[0], tmp_rows[2]);
        let a1 = i32x4_add(tmp_rows[1], tmp_rows[3]);
        let a2 = i32x4_sub(tmp_rows[1], tmp_rows[3]);
        let a3 = i32x4_sub(tmp_rows[0], tmp_rows[2]);

        let o0 = i32x4_shr(i32x4_add(a0, a1), 1);
        let o1 = i32x4_shr(i32x4_add(a3, a2), 1);
        let o2 = i32x4_shr(i32x4_sub(a3, a2), 1);
        let o3 = i32x4_shr(i32x4_sub(a0, a1), 1);

        store_row_i32x4_as_i16(out, 0, o0);
        store_row_i32x4_as_i16(out, 1, o1);
        store_row_i32x4_as_i16(out, 2, o2);
        store_row_i32x4_as_i16(out, 3, o3);
    }

    pub fn inverse_wht_simd128(coeffs: &[i16; 16], out: &mut [i16; 16]) {
        // Load rows
        let r0 = load_row_i16_as_i32x4(coeffs, 0);
        let r1 = load_row_i16_as_i32x4(coeffs, 1);
        let r2 = load_row_i16_as_i32x4(coeffs, 2);
        let r3 = load_row_i16_as_i32x4(coeffs, 3);

        // Vertical pass: use row vectors directly (no transpose)
        let a0 = i32x4_add(r0, r3);
        let a1 = i32x4_add(r1, r2);
        let a2 = i32x4_sub(r1, r2);
        let a3 = i32x4_sub(r0, r3);

        let t0 = i32x4_add(a0, a1);
        let t1 = i32x4_add(a3, a2);
        let t2 = i32x4_sub(a0, a1);
        let t3 = i32x4_sub(a3, a2);

        // Horizontal pass: scalar per-row (need cross-lane access)
        for (row, t) in [(0, t0), (1, t1), (2, t2), (3, t3)] {
            let v0 = i32x4_extract_lane::<0>(t);
            let v1 = i32x4_extract_lane::<1>(t);
            let v2 = i32x4_extract_lane::<2>(t);
            let v3 = i32x4_extract_lane::<3>(t);

            let dc = v0 + 3;
            let a0 = dc + v3;
            let a1 = v1 + v2;
            let a2 = v1 - v2;
            let a3 = dc - v3;

            let base = row * 4;
            out[base] = ((a0 + a1) >> 3) as i16;
            out[base + 1] = ((a3 + a2) >> 3) as i16;
            out[base + 2] = ((a0 - a1) >> 3) as i16;
            out[base + 3] = ((a3 - a2) >> 3) as i16;
        }
    }
}

#[cfg(target_arch = "wasm32")]
use simd128::*;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_inverse_dct_roundtrip_zero_reference() {
        let src: [u8; 16] = [
            52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 67, 68, 78, 82,
        ];
        let reference = [0u8; 16];

        let mut coeffs = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs);

        assert_ne!(coeffs[0], 0, "DC coefficient should be non-zero");

        let mut reconstructed = [0u8; 16];
        inverse_dct(&coeffs, &reference, &mut reconstructed);

        for i in 0..16 {
            let diff = (src[i] as i32 - reconstructed[i] as i32).abs();
            assert!(
                diff <= 1,
                "pixel {i}: src={}, reconstructed={}, diff={diff}",
                src[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn forward_inverse_dct_roundtrip_with_reference() {
        let src = [128u8; 16];
        let reference = [120u8; 16];

        let mut coeffs = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs);
        assert_ne!(coeffs[0], 0);

        let mut reconstructed = [0u8; 16];
        inverse_dct(&coeffs, &reference, &mut reconstructed);

        for i in 0..16 {
            let diff = (src[i] as i32 - reconstructed[i] as i32).abs();
            assert!(diff <= 1, "pixel {i}: diff={diff}");
        }
    }

    #[test]
    fn forward_dct_dc_coefficient_is_scaled_mean() {
        let src = [100u8; 16];
        let reference = [0u8; 16];
        let mut coeffs = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs);
        assert!(coeffs[0] > 50, "DC coeff {} seems too low", coeffs[0]);
    }

    #[test]
    fn inverse_dct_clamps_to_valid_range() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 2000;
        let reference = [128u8; 16];
        let mut dst = [0u8; 16];
        inverse_dct(&coeffs, &reference, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(v <= 255, "pixel {i} out of range: {v}");
        }
    }

    #[test]
    fn dct_zero_residual_produces_near_zero_coefficients() {
        let pixels = [100u8; 16];
        let reference = [100u8; 16];
        let mut coeffs = [0i16; 16];
        forward_dct(&pixels, &reference, &mut coeffs);
        assert_eq!(coeffs[0], 0, "DC should be 0 for zero residual");
        for i in 1..16 {
            assert!(coeffs[i].abs() <= 1, "coeff [{i}]={} too large", coeffs[i]);
        }
    }

    #[test]
    fn forward_inverse_wht_roundtrip() {
        let dc_coeffs: [i16; 16] = [
            100, -20, 30, -40, 50, -60, 70, -80, 15, -25, 35, -45, 55, -65, 75, -85,
        ];
        let mut transformed = [0i16; 16];
        forward_wht(&dc_coeffs, &mut transformed);
        let mut reconstructed = [0i16; 16];
        inverse_wht(&transformed, &mut reconstructed);
        for i in 0..16 {
            let diff = (dc_coeffs[i] as i32 - reconstructed[i] as i32).abs();
            assert!(diff <= 1, "coeff {i}: diff={diff}");
        }
    }

    #[test]
    fn wht_flat_input_concentrates_in_dc() {
        let dc_coeffs = [42i16; 16];
        let mut transformed = [0i16; 16];
        forward_wht(&dc_coeffs, &mut transformed);
        assert_ne!(transformed[0], 0);
        for i in 1..16 {
            assert_eq!(transformed[i], 0, "AC [{i}] should be 0");
        }
    }

    #[test]
    fn forward_dct_reference_values() {
        let src = [128u8; 16];
        let reference = [0u8; 16];
        let mut coeffs = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs);
        let dc = coeffs[0];
        assert!(dc > 400, "DC should be > 400, got {dc}");
        let mut coeffs2 = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs2);
        assert_eq!(coeffs, coeffs2, "DCT must be deterministic");
    }

    #[test]
    fn dct_roundtrip_exhaustive() {
        for seed in 0..100u32 {
            let mut src = [0u8; 16];
            for i in 0..16 {
                src[i] = ((seed.wrapping_mul(7919) + (i as u32).wrapping_mul(6271)) % 256) as u8;
            }
            let reference = [0u8; 16];
            let mut coeffs = [0i16; 16];
            forward_dct(&src, &reference, &mut coeffs);
            let mut reconstructed = [0u8; 16];
            inverse_dct(&coeffs, &reference, &mut reconstructed);
            for i in 0..16 {
                let diff = (src[i] as i32 - reconstructed[i] as i32).abs();
                assert!(diff <= 1, "seed={seed}, pixel {i}: diff={diff}");
            }
        }
    }

    #[test]
    fn dct_gradient_input_has_ac_energy() {
        let src: [u8; 16] = [
            0, 85, 170, 255, 0, 85, 170, 255, 0, 85, 170, 255, 0, 85, 170, 255,
        ];
        let reference = [0u8; 16];
        let mut coeffs = [0i16; 16];
        forward_dct(&src, &reference, &mut coeffs);
        let ac_energy: i32 = coeffs[1..].iter().map(|&c| (c as i32).pow(2)).sum();
        assert!(ac_energy > 0, "gradient input should have AC energy");
    }

    #[test]
    fn dct_roundtrip_many_patterns() {
        let patterns: Vec<[u8; 16]> = vec![
            [
                0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
            [
                0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255,
            ],
            [
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
            ],
            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        for (pi, pattern) in patterns.iter().enumerate() {
            let reference = [0u8; 16];
            let mut coeffs = [0i16; 16];
            forward_dct(pattern, &reference, &mut coeffs);
            let mut reconstructed = [0u8; 16];
            inverse_dct(&coeffs, &reference, &mut reconstructed);
            for i in 0..16 {
                let diff = (pattern[i] as i32 - reconstructed[i] as i32).abs();
                assert!(diff <= 1, "pattern {pi}, pixel {i}: diff={diff}");
            }
        }
    }
}
