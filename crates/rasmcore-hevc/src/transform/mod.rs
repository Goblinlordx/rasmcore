//! HEVC Transform + Dequantization (ITU-T H.265 Sections 8.6.3, 8.6.4).
//!
//! Provides the full residual reconstruction pipeline:
//! 1. Dequantize CABAC-decoded coefficients using QP and optional scaling lists
//! 2. Apply inverse transform (DST for 4x4 intra luma, DCT for all others)
//! 3. Clip residuals to valid range

pub mod dequant;
pub mod scaling_list;

pub use dequant::{dequantize_block, dequantize_block_with_scaling_list, level_scale};
pub use scaling_list::{ScalingList, parse_scaling_list_data};

use crate::error::HevcError;

/// Reconstruct spatial residuals from quantized coefficients.
///
/// Full pipeline: dequantize → inverse transform → clip.
///
/// # Arguments
/// * `coeffs` — quantized coefficients from CABAC (row-major)
/// * `output` — spatial residual output (row-major, same size as coeffs)
/// * `log2_size` — log2 of the transform block size (2=4x4 through 5=32x32)
/// * `qp` — quantization parameter
/// * `bit_depth` — luma/chroma bit depth (8 or 10)
/// * `is_intra_4x4_luma` — use DST instead of DCT for 4x4 intra luma
/// * `scaling_list` — optional scaling list (None = flat default)
/// * `matrix_id` — scaling list matrix ID (0–5)
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_residual(
    coeffs: &[i16],
    output: &mut [i16],
    log2_size: u8,
    qp: i32,
    bit_depth: u8,
    is_intra_4x4_luma: bool,
    scaling_list: Option<&ScalingList>,
    matrix_id: u8,
) -> Result<(), HevcError> {
    let block_size = 1usize << (log2_size as usize * 2);

    if coeffs.len() < block_size || output.len() < block_size {
        return Err(HevcError::DecodeFailed(format!(
            "buffer too small for {}x{} transform",
            1 << log2_size,
            1 << log2_size
        )));
    }

    // Step 1: Dequantize (copy coefficients then dequant in-place)
    let mut dequant_buf = vec![0i16; block_size];
    dequant_buf[..block_size].copy_from_slice(&coeffs[..block_size]);

    if let Some(sl) = scaling_list {
        dequantize_block_with_scaling_list(
            &mut dequant_buf,
            log2_size,
            qp,
            bit_depth,
            sl,
            matrix_id,
        );
    } else {
        dequantize_block(&mut dequant_buf, log2_size, qp, bit_depth);
    }

    // Step 2: Inverse transform
    inverse_transform(&dequant_buf, output, log2_size, is_intra_4x4_luma)?;

    // Step 3: Clip residuals to valid range.
    // HEVC spec Section 8.6.2: residuals are clipped to [-(1<<(BitDepth+1)), (1<<(BitDepth+1))-1]
    // for standard precision. For 8-bit: [-512, 511].
    // NOT [-128, 127] — residuals can exceed the pixel range because they
    // represent the difference between prediction and reconstruction.
    // Ref: libde265 v1.0.18 transform.cc line 283 — bdShift = 20 - bit_depth
    // (the IDCT output range is determined by the transform shift, not pixel depth).
    let max_val = (1i16 << (bit_depth + 1)) - 1;
    let min_val = -(1i16 << (bit_depth + 1));
    for v in output[..block_size].iter_mut() {
        *v = (*v).clamp(min_val, max_val);
    }

    Ok(())
}

/// Apply the appropriate inverse transform for a given block size.
///
/// Selection rule (Section 8.6.4.2):
/// - 4x4 intra luma: inverse DST
/// - All other sizes and modes: inverse DCT
fn inverse_transform(
    input: &[i16],
    output: &mut [i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
) -> Result<(), HevcError> {
    match log2_size {
        2 => {
            let inp: &[i16; 16] = input[..16]
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("4x4 transform buffer error".into()))?;
            let out: &mut [i16; 16] = (&mut output[..16])
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("4x4 transform buffer error".into()))?;

            if is_intra_4x4_luma {
                rasmcore_dct::inverse_dst_4x4(inp, out);
            } else {
                rasmcore_dct::inverse_dct_4x4(inp, out);
            }
        }
        3 => {
            let inp: &[i16; 64] = input[..64]
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("8x8 transform buffer error".into()))?;
            let out: &mut [i16; 64] = (&mut output[..64])
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("8x8 transform buffer error".into()))?;
            rasmcore_dct::inverse_dct_8x8(inp, out);
        }
        4 => {
            let inp: &[i16; 256] = input[..256]
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("16x16 transform buffer error".into()))?;
            let out: &mut [i16; 256] = (&mut output[..256])
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("16x16 transform buffer error".into()))?;
            rasmcore_dct::inverse_dct_16x16(inp, out);
        }
        5 => {
            let inp: &[i16; 1024] = input[..1024]
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("32x32 transform buffer error".into()))?;
            let out: &mut [i16; 1024] = (&mut output[..1024])
                .try_into()
                .map_err(|_| HevcError::DecodeFailed("32x32 transform buffer error".into()))?;
            rasmcore_dct::inverse_dct_32x32(inp, out);
        }
        _ => {
            return Err(HevcError::DecodeFailed(format!(
                "unsupported transform size log2={log2_size}"
            )));
        }
    }

    Ok(())
}
