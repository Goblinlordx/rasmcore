//! CABAC (Context-Adaptive Binary Arithmetic Coding) engine for HEVC.
//!
//! Implements the entropy decoding layer per ITU-T H.265 Section 9.3:
//! - Binary arithmetic decoder with 9-bit range
//! - Adaptive context models with LPS/MPS state transitions
//! - Binarization schemes: truncated Rice, Exp-Golomb, fixed-length, truncated unary

mod context;
mod decoder;
pub mod tables;

pub use context::{
    ABS_MVD_GREATER0_INIT,
    ABS_MVD_GREATER1_INIT,
    CBF_CHROMA_INIT,
    CBF_LUMA_INIT,
    CNU,
    CODED_SUB_BLOCK_FLAG_INIT,
    COEFF_ABS_LEVEL_GREATER1_FLAG_INIT,
    COEFF_ABS_LEVEL_GREATER2_FLAG_INIT,
    CU_CHROMA_QP_OFFSET_INIT,
    CU_QP_DELTA_ABS_INIT,
    CU_SKIP_FLAG_INIT,
    CU_TRANSQUANT_BYPASS_FLAG_INIT,
    ContextModel,
    INTER_PRED_IDC_INIT,
    INTRA_CHROMA_PRED_MODE_INIT,
    LAST_SIG_COEFF_X_PREFIX_INIT,
    LAST_SIG_COEFF_Y_PREFIX_INIT,
    MERGE_FLAG_INIT,
    MERGE_IDX_INIT,
    MVP_LX_FLAG_INIT,
    PART_MODE_INIT,
    PRED_MODE_FLAG_INIT,
    PREV_INTRA_LUMA_PRED_FLAG_INIT,
    REF_IDX_INIT,
    RQT_ROOT_CBF_INIT,
    // Init-value tables (Tables 9-5 through 9-37)
    SAO_MERGE_FLAG_INIT,
    SAO_TYPE_IDX_INIT,
    SIG_COEFF_FLAG_INIT,
    SPLIT_CU_FLAG_INIT,
    SPLIT_TRANSFORM_FLAG_INIT,
    SliceType,
    TRANSFORM_SKIP_FLAG_INIT,
    init_contexts,
};
pub use decoder::CabacDecoder;
