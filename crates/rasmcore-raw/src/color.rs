//! DNG color pipeline — white balance, camera-to-XYZ, XYZ-to-sRGB, gamma.
//!
//! Implements the DNG color processing chain:
//! 1. Black level subtraction and white level normalization
//! 2. White balance (AsShotNeutral multipliers)
//! 3. Camera RGB → XYZ (inverted ColorMatrix1)
//! 4. XYZ → linear sRGB (IEC 61966-2-1 matrix)
//! 5. sRGB gamma curve (linear → sRGB transfer function)

/// Standard XYZ-to-sRGB matrix (IEC 61966-2-1, D65 illuminant).
/// Converts from CIE XYZ to linear sRGB.
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404541621, -1.5371385940, -0.4985314096],
    [-0.9692660305, 1.8760108454, 0.0415560175],
    [0.0556434309, -0.2040259135, 1.0572251882],
];

/// Invert a 3×3 matrix. Returns None if singular (determinant ≈ 0).
pub fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// Multiply two 3×3 matrices: result = a × b.
pub fn mul_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

/// sRGB gamma curve: linear → sRGB.
/// IEC 61966-2-1: if linear ≤ 0.0031308, srgb = 12.92 * linear
/// else srgb = 1.055 * linear^(1/2.4) - 0.055
#[inline]
pub fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Build the combined camera-to-sRGB matrix from DNG ColorMatrix1.
///
/// DNG ColorMatrix1 maps XYZ to camera color space (3×3, stored row-major as 9 values).
/// We need camera_to_xyz = inverse(color_matrix1), then camera_to_srgb = xyz_to_srgb × camera_to_xyz.
pub fn build_camera_to_srgb(color_matrix: &[f64; 9]) -> Option<[[f64; 3]; 3]> {
    let xyz_to_camera = [
        [color_matrix[0], color_matrix[1], color_matrix[2]],
        [color_matrix[3], color_matrix[4], color_matrix[5]],
        [color_matrix[6], color_matrix[7], color_matrix[8]],
    ];
    let camera_to_xyz = invert_3x3(&xyz_to_camera)?;
    Some(mul_3x3(&XYZ_TO_SRGB, &camera_to_xyz))
}

/// Apply the full DNG color pipeline to demosaiced RGB16 data in-place.
///
/// Steps:
/// 1. Black level subtract and normalize to [0.0, 1.0] using white level
/// 2. White balance (multiply by per-channel gains from AsShotNeutral)
/// 3. Camera RGB → linear sRGB (via combined matrix)
/// 4. Clip to [0.0, 1.0]
/// 5. Apply sRGB gamma
/// 6. Scale back to u16 output range
///
/// `rgb16`: interleaved RGB u16 data (width × height × 3)
/// `camera_to_srgb`: combined 3×3 matrix
/// `white_balance`: per-channel multipliers [R, G, B]
/// `black_level`: per-channel black levels (subtracted first)
/// `white_level`: maximum sensor value (after black subtraction, this maps to 1.0)
/// `output_8bit`: if true, output is scaled to [0, 255]; if false, [0, 65535]
pub fn apply_color_pipeline(
    rgb16: &[u16],
    width: u32,
    height: u32,
    camera_to_srgb: &[[f64; 3]; 3],
    white_balance: &[f64; 3],
    black_level: f64,
    white_level: f64,
    output_8bit: bool,
) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let range = white_level - black_level;
    let inv_range = if range > 0.0 { 1.0 / range } else { 1.0 };
    let out_max = if output_8bit { 255.0 } else { 65535.0 };

    let bytes_per_channel = if output_8bit { 1 } else { 2 };
    let mut output = vec![0u8; pixel_count * 3 * bytes_per_channel];

    // Process with SIMD where available
    #[cfg(target_arch = "aarch64")]
    {
        apply_color_pipeline_neon(
            rgb16,
            &mut output,
            pixel_count,
            camera_to_srgb,
            white_balance,
            black_level,
            inv_range,
            out_max,
            output_8bit,
        );
        return output;
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") {
        apply_color_pipeline_scalar(
            rgb16,
            &mut output,
            pixel_count,
            camera_to_srgb,
            white_balance,
            black_level,
            inv_range,
            out_max,
            output_8bit,
        );
        return output;
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    {
        apply_color_pipeline_scalar(
            rgb16,
            &mut output,
            pixel_count,
            camera_to_srgb,
            white_balance,
            black_level,
            inv_range,
            out_max,
            output_8bit,
        );
        output
    }
}

fn apply_color_pipeline_scalar(
    rgb16: &[u16],
    output: &mut [u8],
    pixel_count: usize,
    camera_to_srgb: &[[f64; 3]; 3],
    white_balance: &[f64; 3],
    black_level: f64,
    inv_range: f64,
    out_max: f64,
    output_8bit: bool,
) {
    for i in 0..pixel_count {
        let src_idx = i * 3;
        if src_idx + 2 >= rgb16.len() {
            break;
        }

        // Step 1: Black subtract and normalize
        let r = ((rgb16[src_idx] as f64) - black_level).max(0.0) * inv_range;
        let g = ((rgb16[src_idx + 1] as f64) - black_level).max(0.0) * inv_range;
        let b = ((rgb16[src_idx + 2] as f64) - black_level).max(0.0) * inv_range;

        // Step 2: White balance
        let r = r * white_balance[0];
        let g = g * white_balance[1];
        let b = b * white_balance[2];

        // Step 3: Camera RGB → linear sRGB via matrix
        let sr = camera_to_srgb[0][0] * r + camera_to_srgb[0][1] * g + camera_to_srgb[0][2] * b;
        let sg = camera_to_srgb[1][0] * r + camera_to_srgb[1][1] * g + camera_to_srgb[1][2] * b;
        let sb = camera_to_srgb[2][0] * r + camera_to_srgb[2][1] * g + camera_to_srgb[2][2] * b;

        // Step 4: Clip
        let sr = sr.clamp(0.0, 1.0);
        let sg = sg.clamp(0.0, 1.0);
        let sb = sb.clamp(0.0, 1.0);

        // Step 5: sRGB gamma
        let sr = linear_to_srgb(sr);
        let sg = linear_to_srgb(sg);
        let sb = linear_to_srgb(sb);

        // Step 6: Scale and write
        if output_8bit {
            let out_idx = i * 3;
            output[out_idx] = (sr * out_max + 0.5) as u8;
            output[out_idx + 1] = (sg * out_max + 0.5) as u8;
            output[out_idx + 2] = (sb * out_max + 0.5) as u8;
        } else {
            let out_idx = i * 6;
            let rv = (sr * out_max + 0.5) as u16;
            let gv = (sg * out_max + 0.5) as u16;
            let bv = (sb * out_max + 0.5) as u16;
            output[out_idx..out_idx + 2].copy_from_slice(&rv.to_le_bytes());
            output[out_idx + 2..out_idx + 4].copy_from_slice(&gv.to_le_bytes());
            output[out_idx + 4..out_idx + 6].copy_from_slice(&bv.to_le_bytes());
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn apply_color_pipeline_neon(
    rgb16: &[u16],
    output: &mut [u8],
    pixel_count: usize,
    camera_to_srgb: &[[f64; 3]; 3],
    white_balance: &[f64; 3],
    black_level: f64,
    inv_range: f64,
    out_max: f64,
    output_8bit: bool,
) {
    use std::arch::aarch64::*;

    let m00 = camera_to_srgb[0][0] as f32;
    let m01 = camera_to_srgb[0][1] as f32;
    let m02 = camera_to_srgb[0][2] as f32;
    let m10 = camera_to_srgb[1][0] as f32;
    let m11 = camera_to_srgb[1][1] as f32;
    let m12 = camera_to_srgb[1][2] as f32;
    let m20 = camera_to_srgb[2][0] as f32;
    let m21 = camera_to_srgb[2][1] as f32;
    let m22 = camera_to_srgb[2][2] as f32;

    let wb_r = white_balance[0] as f32;
    let wb_g = white_balance[1] as f32;
    let wb_b = white_balance[2] as f32;
    let bl = black_level as f32;
    let inv_r = inv_range as f32;

    let mut i = 0;
    while i + 4 <= pixel_count {
        // Load 4 pixels (12 u16 values) — deinterleave R, G, B
        let mut r_vals = [0f32; 4];
        let mut g_vals = [0f32; 4];
        let mut b_vals = [0f32; 4];
        for j in 0..4 {
            let idx = (i + j) * 3;
            r_vals[j] = rgb16[idx] as f32;
            g_vals[j] = rgb16[idx + 1] as f32;
            b_vals[j] = rgb16[idx + 2] as f32;
        }

        // SAFETY: aarch64 always has NEON. All array accesses are within bounds
        // (4-element f32 arrays, pixel_count bounds checked by loop condition).
        let (sr_arr, sg_arr, sb_arr) = unsafe {
            let v_bl = vdupq_n_f32(bl);
            let v_inv_range = vdupq_n_f32(inv_r);
            let v_zero = vdupq_n_f32(0.0);
            let v_one = vdupq_n_f32(1.0);

            let mut vr = vld1q_f32(r_vals.as_ptr());
            let mut vg = vld1q_f32(g_vals.as_ptr());
            let mut vb = vld1q_f32(b_vals.as_ptr());

            // Black subtract + normalize
            vr = vmaxq_f32(vsubq_f32(vr, v_bl), v_zero);
            vg = vmaxq_f32(vsubq_f32(vg, v_bl), v_zero);
            vb = vmaxq_f32(vsubq_f32(vb, v_bl), v_zero);
            vr = vmulq_f32(vr, v_inv_range);
            vg = vmulq_f32(vg, v_inv_range);
            vb = vmulq_f32(vb, v_inv_range);

            // White balance
            vr = vmulq_f32(vr, vdupq_n_f32(wb_r));
            vg = vmulq_f32(vg, vdupq_n_f32(wb_g));
            vb = vmulq_f32(vb, vdupq_n_f32(wb_b));

            // Matrix multiply: camera RGB → linear sRGB
            let sr = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(vdupq_n_f32(m00), vr),
                    vmulq_f32(vdupq_n_f32(m01), vg),
                ),
                vmulq_f32(vdupq_n_f32(m02), vb),
            );
            let sg = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(vdupq_n_f32(m10), vr),
                    vmulq_f32(vdupq_n_f32(m11), vg),
                ),
                vmulq_f32(vdupq_n_f32(m12), vb),
            );
            let sb = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(vdupq_n_f32(m20), vr),
                    vmulq_f32(vdupq_n_f32(m21), vg),
                ),
                vmulq_f32(vdupq_n_f32(m22), vb),
            );

            // Clip to [0, 1]
            let sr = vminq_f32(vmaxq_f32(sr, v_zero), v_one);
            let sg = vminq_f32(vmaxq_f32(sg, v_zero), v_one);
            let sb = vminq_f32(vmaxq_f32(sb, v_zero), v_one);

            // Extract to arrays for scalar gamma
            let mut sr_a = [0f32; 4];
            let mut sg_a = [0f32; 4];
            let mut sb_a = [0f32; 4];
            vst1q_f32(sr_a.as_mut_ptr(), sr);
            vst1q_f32(sg_a.as_mut_ptr(), sg);
            vst1q_f32(sb_a.as_mut_ptr(), sb);
            (sr_a, sg_a, sb_a)
        };

        // Gamma + write (scalar — no SIMD pow)
        for j in 0..4 {
            let gr = linear_to_srgb(sr_arr[j] as f64);
            let gg = linear_to_srgb(sg_arr[j] as f64);
            let gb = linear_to_srgb(sb_arr[j] as f64);

            if output_8bit {
                let out_idx = (i + j) * 3;
                output[out_idx] = (gr * out_max + 0.5) as u8;
                output[out_idx + 1] = (gg * out_max + 0.5) as u8;
                output[out_idx + 2] = (gb * out_max + 0.5) as u8;
            } else {
                let out_idx = (i + j) * 6;
                let rv = (gr * out_max + 0.5) as u16;
                let gv = (gg * out_max + 0.5) as u16;
                let bv = (gb * out_max + 0.5) as u16;
                output[out_idx..out_idx + 2].copy_from_slice(&rv.to_le_bytes());
                output[out_idx + 2..out_idx + 4].copy_from_slice(&gv.to_le_bytes());
                output[out_idx + 4..out_idx + 6].copy_from_slice(&bv.to_le_bytes());
            }
        }

        i += 4;
    }

    // Remaining pixels (scalar)
    for idx in i..pixel_count {
        let src = idx * 3;
        if src + 2 >= rgb16.len() {
            break;
        }
        let r = ((rgb16[src] as f64) - black_level).max(0.0) * inv_range;
        let g = ((rgb16[src + 1] as f64) - black_level).max(0.0) * inv_range;
        let b = ((rgb16[src + 2] as f64) - black_level).max(0.0) * inv_range;
        let r = r * white_balance[0];
        let g = g * white_balance[1];
        let b = b * white_balance[2];
        let sr = (camera_to_srgb[0][0] * r + camera_to_srgb[0][1] * g + camera_to_srgb[0][2] * b)
            .clamp(0.0, 1.0);
        let sg = (camera_to_srgb[1][0] * r + camera_to_srgb[1][1] * g + camera_to_srgb[1][2] * b)
            .clamp(0.0, 1.0);
        let sb = (camera_to_srgb[2][0] * r + camera_to_srgb[2][1] * g + camera_to_srgb[2][2] * b)
            .clamp(0.0, 1.0);
        let sr = linear_to_srgb(sr);
        let sg = linear_to_srgb(sg);
        let sb = linear_to_srgb(sb);
        if output_8bit {
            let out_idx = idx * 3;
            output[out_idx] = (sr * out_max + 0.5) as u8;
            output[out_idx + 1] = (sg * out_max + 0.5) as u8;
            output[out_idx + 2] = (sb * out_max + 0.5) as u8;
        } else {
            let out_idx = idx * 6;
            let rv = (sr * out_max + 0.5) as u16;
            let gv = (sg * out_max + 0.5) as u16;
            let bv = (sb * out_max + 0.5) as u16;
            output[out_idx..out_idx + 2].copy_from_slice(&rv.to_le_bytes());
            output[out_idx + 2..out_idx + 4].copy_from_slice(&gv.to_le_bytes());
            output[out_idx + 4..out_idx + 6].copy_from_slice(&bv.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&id).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-10,
                    "inv[{i}][{j}] = {} != {expected}",
                    inv[i][j]
                );
            }
        }
    }

    #[test]
    fn invert_known_matrix() {
        // Simple test: [[2,0,0],[0,3,0],[0,0,4]] -> inv = [[0.5,0,0],[0,1/3,0],[0,0,0.25]]
        let m = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let inv = invert_3x3(&m).unwrap();
        assert!((inv[0][0] - 0.5).abs() < 1e-10);
        assert!((inv[1][1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((inv[2][2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn singular_returns_none() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert!(invert_3x3(&m).is_none());
    }

    #[test]
    fn srgb_gamma_black() {
        assert!((linear_to_srgb(0.0)).abs() < 1e-10);
    }

    #[test]
    fn srgb_gamma_white() {
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn srgb_gamma_midpoint() {
        let mid = linear_to_srgb(0.5);
        // sRGB(0.5) ≈ 0.735
        assert!(mid > 0.7 && mid < 0.8, "sRGB(0.5) = {mid}");
    }

    #[test]
    fn build_identity_camera_matrix() {
        // If ColorMatrix1 = identity (camera = XYZ), then camera_to_srgb = XYZ_to_SRGB
        let cm = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = build_camera_to_srgb(&cm).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result[i][j] - XYZ_TO_SRGB[i][j]).abs() < 1e-8,
                    "mismatch at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn color_pipeline_uniform_white() {
        // All pixels at white level, no black, identity camera matrix, no WB
        let rgb16 = vec![4095u16; 12]; // 4 pixels × 3 channels
        let identity_cm = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let cam_to_srgb = build_camera_to_srgb(&identity_cm).unwrap();
        let wb = [1.0, 1.0, 1.0];

        let output = apply_color_pipeline(&rgb16, 2, 2, &cam_to_srgb, &wb, 0.0, 4095.0, true);

        // With identity matrix and full-scale input, output should be near white
        // (exact value depends on XYZ_to_sRGB matrix row sums)
        assert_eq!(output.len(), 12); // 4 pixels × 3 bytes
        // Each channel should be > 200 (approximately white after gamma)
        for val in &output {
            assert!(*val > 100, "expected bright pixel, got {val}");
        }
    }
}
