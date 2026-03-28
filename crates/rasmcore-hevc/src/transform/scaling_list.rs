//! HEVC Scaling List management (ITU-T H.265 Section 7.3.4, 7.4.5).

use crate::bitread::HevcBitReader;
use crate::error::HevcError;

/// Default 8x8 intra scaling list (Table 7-4).
const DEFAULT_8X8_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

/// Default 8x8 inter scaling list (Table 7-4).
const DEFAULT_8X8_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

/// Up-right diagonal scan order for 4x4 blocks (Table 6-5).
const DIAG_SCAN_4X4: [u8; 16] = [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15];

/// Up-right diagonal scan order for 8x8 blocks.
const DIAG_SCAN_8X8: [u8; 64] = [
    0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 32, 25, 18, 11, 4, 40, 33, 26, 19, 12, 5, 48, 41, 34, 27, 20,
    13, 6, 56, 49, 42, 35, 28, 21, 14, 7, 57, 50, 43, 36, 29, 22, 15, 58, 51, 44, 37, 30, 23, 59,
    52, 45, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63,
];

/// HEVC Scaling List data for all transform sizes.
///
/// Stores scaling matrices per size and matrix ID:
/// - matrixId 0–2: intra Y, Cb, Cr
/// - matrixId 3–5: inter Y, Cb, Cr
#[derive(Debug, Clone)]
pub struct ScalingList {
    /// 4x4 lists (sizeId 0): 6 matrices, 16 values each.
    pub list_4x4: [[u8; 16]; 6],
    /// 8x8 lists (sizeId 1): 6 matrices, 64 values each.
    pub list_8x8: [[u8; 64]; 6],
    /// 16x16 lists (sizeId 2): stored as 8x8, 6 matrices.
    pub list_16x16: [[u8; 64]; 6],
    /// 32x32 lists (sizeId 3): stored as 8x8, 2 matrices (intra Y, inter Y).
    pub list_32x32: [[u8; 64]; 2],
    /// DC coefficients for 16x16 (indices 0–5 = matrixId).
    pub dc_16x16: [u8; 6],
    /// DC coefficients for 32x32 (index 0 = intra Y, 1 = inter Y).
    pub dc_32x32: [u8; 2],
}

impl Default for ScalingList {
    /// Create default scaling lists per H.265 Table 7-4.
    fn default() -> Self {
        let flat_4x4 = [16u8; 16];
        let intra_8x8 = DEFAULT_8X8_INTRA;
        let inter_8x8 = DEFAULT_8X8_INTER;

        Self {
            list_4x4: [flat_4x4; 6],
            list_8x8: [
                intra_8x8, intra_8x8, intra_8x8, inter_8x8, inter_8x8, inter_8x8,
            ],
            list_16x16: [
                intra_8x8, intra_8x8, intra_8x8, inter_8x8, inter_8x8, inter_8x8,
            ],
            list_32x32: [intra_8x8, inter_8x8],
            dc_16x16: [16; 6],
            dc_32x32: [16; 2],
        }
    }
}

impl ScalingList {
    /// Get the scaling factor for a coefficient at (x, y) in a block.
    ///
    /// # Arguments
    /// * `log2_size` — log2 of the transform block size (2=4x4, 3=8x8, 4=16x16, 5=32x32)
    /// * `matrix_id` — scaling list matrix ID (0–5, see struct docs)
    /// * `x`, `y` — coefficient position in the block
    pub fn get_factor(&self, log2_size: u8, matrix_id: u8, x: usize, y: usize) -> u8 {
        match log2_size {
            2 => self.list_4x4[matrix_id as usize][y * 4 + x],
            3 => self.list_8x8[matrix_id as usize][y * 8 + x],
            4 => {
                if x == 0 && y == 0 {
                    self.dc_16x16[matrix_id as usize]
                } else {
                    self.list_16x16[matrix_id as usize][(y / 2) * 8 + (x / 2)]
                }
            }
            5 => {
                let idx = if matrix_id >= 3 { 1 } else { 0 };
                if x == 0 && y == 0 {
                    self.dc_32x32[idx]
                } else {
                    self.list_32x32[idx][(y / 4) * 8 + (x / 4)]
                }
            }
            _ => 16,
        }
    }
}

/// Parse scaling_list_data() from the bitstream (Section 7.3.4).
pub fn parse_scaling_list_data(r: &mut HevcBitReader) -> Result<ScalingList, HevcError> {
    let mut sl = ScalingList::default();

    for size_id in 0u8..4 {
        let matrix_count: u8 = if size_id == 3 { 2 } else { 6 };

        for matrix_id in 0..matrix_count {
            let pred_mode_flag = r.read_flag()?;

            if !pred_mode_flag {
                let delta = r.read_ue()? as u8;
                // Copy from reference matrix (matrix_id - delta)
                if delta == 0 {
                    // Use default scaling list
                    set_default_matrix(&mut sl, size_id, matrix_id);
                } else {
                    let ref_id = matrix_id - delta;
                    copy_matrix(&mut sl, size_id, matrix_id, ref_id);
                }
            } else {
                // Parse DPCM-coded scaling list
                let coeff_num = std::cmp::min(64u32, 1 << ((4 + (size_id as u32)) << 1));

                // DC coefficient for 16x16 and 32x32
                if size_id > 1 {
                    let dc_delta = r.read_se()?;
                    let dc = ((8i32 + dc_delta + 256) % 256) as u8;
                    match size_id {
                        2 => sl.dc_16x16[matrix_id as usize] = dc,
                        3 => sl.dc_32x32[matrix_id as usize] = dc,
                        _ => {}
                    }
                }

                // Parse coefficient deltas
                let mut next_coeff: i32 = 8;
                let scan = if size_id == 0 {
                    &DIAG_SCAN_4X4[..]
                } else {
                    &DIAG_SCAN_8X8[..]
                };
                let num = coeff_num as usize;

                for &scan_pos in scan.iter().take(num) {
                    let delta = r.read_se()?;
                    next_coeff = (next_coeff + delta + 256) % 256;
                    let pos = scan_pos as usize;

                    match size_id {
                        0 => sl.list_4x4[matrix_id as usize][pos] = next_coeff as u8,
                        1 => sl.list_8x8[matrix_id as usize][pos] = next_coeff as u8,
                        2 => sl.list_16x16[matrix_id as usize][pos] = next_coeff as u8,
                        3 => sl.list_32x32[matrix_id as usize][pos] = next_coeff as u8,
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(sl)
}

fn set_default_matrix(sl: &mut ScalingList, size_id: u8, matrix_id: u8) {
    let is_intra = matrix_id < 3;
    match size_id {
        0 => sl.list_4x4[matrix_id as usize] = [16; 16],
        1 => {
            sl.list_8x8[matrix_id as usize] = if is_intra {
                DEFAULT_8X8_INTRA
            } else {
                DEFAULT_8X8_INTER
            };
        }
        2 => {
            sl.list_16x16[matrix_id as usize] = if is_intra {
                DEFAULT_8X8_INTRA
            } else {
                DEFAULT_8X8_INTER
            };
            sl.dc_16x16[matrix_id as usize] = 16;
        }
        3 => {
            sl.list_32x32[matrix_id as usize] = if is_intra {
                DEFAULT_8X8_INTRA
            } else {
                DEFAULT_8X8_INTER
            };
            sl.dc_32x32[matrix_id as usize] = 16;
        }
        _ => {}
    }
}

fn copy_matrix(sl: &mut ScalingList, size_id: u8, dst: u8, src: u8) {
    match size_id {
        0 => {
            let src_list = sl.list_4x4[src as usize];
            sl.list_4x4[dst as usize] = src_list;
        }
        1 => {
            let src_list = sl.list_8x8[src as usize];
            sl.list_8x8[dst as usize] = src_list;
        }
        2 => {
            let src_list = sl.list_16x16[src as usize];
            sl.list_16x16[dst as usize] = src_list;
            sl.dc_16x16[dst as usize] = sl.dc_16x16[src as usize];
        }
        3 => {
            let src_list = sl.list_32x32[src as usize];
            sl.list_32x32[dst as usize] = src_list;
            sl.dc_32x32[dst as usize] = sl.dc_32x32[src as usize];
        }
        _ => {}
    }
}
