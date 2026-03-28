//! CABAC context models (ITU-T H.265 Section 9.3.2).

use super::tables::{TRANS_IDX_LPS, TRANS_IDX_MPS};

/// A single CABAC context model.
///
/// Each context tracks an adaptive probability via:
/// - `state`: 6-bit probability state index (0–62). State 0 = highest LPS probability
///   (~0.5), state 62 = lowest (~0.01).
/// - `mps`: most probable symbol (0 or 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextModel {
    pub state: u8,
    pub mps: u8,
}

impl ContextModel {
    /// Initialize a context model from an `init_value` and slice QP (Section 9.3.2.2).
    ///
    /// The `init_value` comes from Tables 9-5 through 9-37 and encodes a linear
    /// relationship between QP and initial probability:
    ///
    /// ```text
    /// m = (init_value >> 4) * 5 - 45
    /// n = ((init_value & 15) << 3) - 16
    /// preCtxState = Clip3(1, 126, ((m * Clip3(0, 51, slice_qp)) >> 4) + n)
    /// ```
    pub fn new(init_value: u8, slice_qp: i32) -> Self {
        let iv = init_value as i32;
        let m = (iv >> 4) * 5 - 45;
        let n = ((iv & 15) << 3) - 16;
        let qp = slice_qp.clamp(0, 51);
        let pre_ctx_state = ((m * qp) >> 4) + n;
        let pre_ctx_state = pre_ctx_state.clamp(1, 126);

        if pre_ctx_state <= 63 {
            Self {
                state: (63 - pre_ctx_state) as u8,
                mps: 0,
            }
        } else {
            Self {
                state: (pre_ctx_state - 64) as u8,
                mps: 1,
            }
        }
    }

    /// Update context after decoding an LPS (least probable symbol).
    #[inline]
    pub fn update_lps(&mut self) {
        if self.state == 0 {
            self.mps = 1 - self.mps;
        }
        self.state = TRANS_IDX_LPS[self.state as usize];
    }

    /// Update context after decoding an MPS (most probable symbol).
    #[inline]
    pub fn update_mps(&mut self) {
        self.state = TRANS_IDX_MPS[self.state as usize];
    }
}

/// Initialize a slice of context models from init values and slice QP.
pub fn init_contexts(init_values: &[u8], slice_qp: i32) -> Vec<ContextModel> {
    init_values
        .iter()
        .map(|&iv| ContextModel::new(iv, slice_qp))
        .collect()
}

/// Placeholder for "context not used" in init-value tables.
pub const CNU: u8 = 154;

// ---------------------------------------------------------------------------
// Context initialization values (ITU-T H.265 Tables 9-5 through 9-37).
//
// Each table is `[[u8; N]; 3]` indexed by `[init_type][ctx_offset]`:
//   init_type 0 = I-slice
//   init_type 1 = P-slice
//   init_type 2 = B-slice
// ---------------------------------------------------------------------------

/// Table 9-5: sao_merge_left_flag / sao_merge_up_flag (1 context).
pub const SAO_MERGE_FLAG_INIT: [[u8; 1]; 3] = [
    [153], // I
    [153], // P
    [153], // B
];

/// Table 9-6: sao_type_idx_luma / sao_type_idx_chroma (1 context).
pub const SAO_TYPE_IDX_INIT: [[u8; 1]; 3] = [
    [200], // I
    [185], // P
    [160], // B
];

/// Table 9-7: split_cu_flag (3 contexts, indexed by depth comparison).
pub const SPLIT_CU_FLAG_INIT: [[u8; 3]; 3] = [
    [139, 141, 157], // I
    [107, 139, 126], // P
    [107, 139, 126], // B
];

/// Table 9-8: cu_transquant_bypass_flag (1 context).
pub const CU_TRANSQUANT_BYPASS_FLAG_INIT: [[u8; 1]; 3] = [
    [154], // I
    [154], // P
    [154], // B
];

/// Table 9-9: cu_skip_flag (3 contexts). Not signaled in I-slices.
pub const CU_SKIP_FLAG_INIT: [[u8; 3]; 3] = [
    [CNU, CNU, CNU], // I
    [197, 185, 201], // P
    [197, 185, 201], // B
];

/// Table 9-10: pred_mode_flag (1 context). Not signaled in I-slices.
pub const PRED_MODE_FLAG_INIT: [[u8; 1]; 3] = [
    [CNU], // I
    [149], // P
    [134], // B
];

/// Table 9-11: part_mode (4 contexts).
pub const PART_MODE_INIT: [[u8; 4]; 3] = [
    [184, CNU, CNU, CNU], // I
    [154, 139, CNU, CNU], // P
    [154, 139, CNU, CNU], // B
];

/// Table 9-12: prev_intra_luma_pred_flag (1 context).
pub const PREV_INTRA_LUMA_PRED_FLAG_INIT: [[u8; 1]; 3] = [
    [184], // I
    [154], // P
    [183], // B
];

/// Table 9-13: intra_chroma_pred_mode (1 context).
pub const INTRA_CHROMA_PRED_MODE_INIT: [[u8; 1]; 3] = [
    [63],  // I
    [152], // P
    [152], // B
];

/// Table 9-14: rqt_root_cbf (1 context). Not signaled in I-slices.
pub const RQT_ROOT_CBF_INIT: [[u8; 1]; 3] = [
    [CNU], // I
    [79],  // P
    [79],  // B
];

/// Table 9-15: merge_flag (1 context). Not signaled in I-slices.
pub const MERGE_FLAG_INIT: [[u8; 1]; 3] = [
    [CNU], // I
    [110], // P
    [154], // B
];

/// Table 9-16: merge_idx (1 context). Not signaled in I-slices.
pub const MERGE_IDX_INIT: [[u8; 1]; 3] = [
    [CNU], // I
    [122], // P
    [137], // B
];

/// Table 9-17: inter_pred_idc (5 contexts). Not signaled in I-slices.
pub const INTER_PRED_IDC_INIT: [[u8; 5]; 3] = [
    [CNU, CNU, CNU, CNU, CNU], // I
    [95, 79, 63, 31, 31],      // P
    [95, 79, 63, 31, 31],      // B
];

/// Table 9-18: ref_idx_l0 / ref_idx_l1 (2 contexts). Not signaled in I-slices.
pub const REF_IDX_INIT: [[u8; 2]; 3] = [
    [CNU, CNU], // I
    [153, 153], // P
    [153, 153], // B
];

/// Table 9-19: mvp_l0_flag / mvp_l1_flag (1 context). Not signaled in I-slices.
pub const MVP_LX_FLAG_INIT: [[u8; 1]; 3] = [
    [CNU], // I
    [168], // P
    [168], // B
];

/// Table 9-20: split_transform_flag (3 contexts, indexed by 5 - log2TrafoSize).
pub const SPLIT_TRANSFORM_FLAG_INIT: [[u8; 3]; 3] = [
    [153, 138, 138], // I
    [124, 138, 94],  // P
    [224, 167, 122], // B
];

/// Table 9-21: cbf_luma (2 contexts).
pub const CBF_LUMA_INIT: [[u8; 2]; 3] = [
    [111, 141], // I (initType 0)
    [153, 111], // P (initType 1)
    [153, 111], // B (initType 2)
];

/// Table 9-22: cbf_cb / cbf_cr (5 contexts, indexed by trafoDepth).
pub const CBF_CHROMA_INIT: [[u8; 5]; 3] = [
    [94, 138, 182, 154, 154],  // I
    [149, 107, 167, 154, 154], // P
    [149, 107, 167, 154, 154], // B
];

/// Table 9-23: abs_mvd_greater0_flag (2 contexts). Not signaled in I-slices.
pub const ABS_MVD_GREATER0_INIT: [[u8; 2]; 3] = [
    [CNU, CNU], // I
    [140, 198], // P
    [140, 198], // B
];

/// Table 9-24: abs_mvd_greater1_flag (2 contexts). Not signaled in I-slices.
pub const ABS_MVD_GREATER1_INIT: [[u8; 2]; 3] = [
    [CNU, CNU], // I
    [140, 198], // P
    [140, 198], // B
];

/// Table 9-27: transform_skip_flag (2 contexts: luma, chroma).
pub const TRANSFORM_SKIP_FLAG_INIT: [[u8; 2]; 3] = [
    [139, 139], // I
    [139, 139], // P
    [139, 139], // B
];

/// Table 9-30: last_sig_coeff_x_prefix (18 contexts).
pub const LAST_SIG_COEFF_X_PREFIX_INIT: [[u8; 18]; 3] = [
    [
        110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    ], // I
    [
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    ], // P
    [
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    ], // B
];

/// Table 9-31: last_sig_coeff_y_prefix (18 contexts).
pub const LAST_SIG_COEFF_Y_PREFIX_INIT: [[u8; 18]; 3] = [
    [
        110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    ], // I
    [
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    ], // P
    [
        125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    ], // B
];

/// Table 9-32: coded_sub_block_flag (4 contexts: 2 luma + 2 chroma).
pub const CODED_SUB_BLOCK_FLAG_INIT: [[u8; 4]; 3] = [
    [91, 171, 134, 141], // I (initType 0)
    [121, 140, 61, 154], // P
    [121, 140, 61, 154], // B
];

/// Table 9-33: sig_coeff_flag (42 contexts: 27 luma + 15 chroma, Main profile).
pub const SIG_COEFF_FLAG_INIT: [[u8; 42]; 3] = [
    // I-slice (initType 0)
    [
        111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141,
        179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
        136, 139, 111, 136, 139, 111,
    ],
    // P-slice (initType 1)
    [
        155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
        151, 183, 140, 151, 183, 140,
    ],
    // B-slice (initType 2)
    [
        170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167,
        151, 183, 140, 151, 183, 140,
    ],
];

/// Table 9-34: coeff_abs_level_greater1_flag (24 contexts).
pub const COEFF_ABS_LEVEL_GREATER1_FLAG_INIT: [[u8; 24]; 3] = [
    // I-slice (initType 0)
    [
        140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
        182, 140, 227, 122, 197,
    ],
    // P-slice (initType 1)
    [
        154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194,
        166, 167, 154, 167, 137, 182,
    ],
    // B-slice (initType 2)
    [
        154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208,
        166, 167, 154, 152, 167, 182,
    ],
];

/// Table 9-35: coeff_abs_level_greater2_flag (6 contexts).
pub const COEFF_ABS_LEVEL_GREATER2_FLAG_INIT: [[u8; 6]; 3] = [
    [138, 153, 136, 167, 152, 152], // I (initType 0)
    [107, 167, 91, 122, 107, 167],  // P (initType 1)
    [107, 167, 91, 107, 107, 167],  // B (initType 2)
];

/// Table 9-36: cu_qp_delta_abs (2 contexts).
pub const CU_QP_DELTA_ABS_INIT: [[u8; 2]; 3] = [
    [154, 154], // I
    [154, 154], // P
    [154, 154], // B
];

/// Table 9-37: cu_chroma_qp_offset_flag / cu_chroma_qp_offset_idx (1 context).
pub const CU_CHROMA_QP_OFFSET_INIT: [[u8; 1]; 3] = [
    [154], // I
    [154], // P
    [154], // B
];

/// HEVC slice types for context initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    I = 0,
    P = 1,
    B = 2,
}

impl SliceType {
    /// Get the init_type index for context initialization.
    ///
    /// For P and B slices, `cabac_init_flag` can swap init types 1 and 2
    /// (Section 9.3.2.2). For I-slices this flag is always 0.
    pub fn init_type(self, cabac_init_flag: bool) -> usize {
        match self {
            SliceType::I => 0,
            SliceType::P => {
                if cabac_init_flag {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if cabac_init_flag {
                    1
                } else {
                    2
                }
            }
        }
    }
}
