//! HEVC Parameter Set parsing — VPS, SPS, PPS.
//!
//! ITU-T H.265 Sections 7.3.2 (VPS), 7.3.2.2 (SPS), 7.3.2.3 (PPS).

use crate::bitread::HevcBitReader;
use crate::error::HevcError;

/// Video Parameter Set (VPS).
#[derive(Debug, Clone)]
pub struct Vps {
    pub vps_id: u8,
    pub max_layers: u8,
    pub max_sub_layers: u8,
    pub temporal_id_nesting: bool,
    pub profile: ProfileTierLevel,
}

/// Profile, Tier, Level information.
#[derive(Debug, Clone, Default)]
pub struct ProfileTierLevel {
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    pub general_profile_idc: u8,
    pub general_profile_compatibility_flags: u32,
    pub general_level_idc: u8,
}

/// Sequence Parameter Set (SPS).
#[derive(Debug, Clone)]
pub struct Sps {
    pub sps_id: u8,
    pub vps_id: u8,
    pub max_sub_layers: u8,
    pub profile: ProfileTierLevel,
    pub chroma_format_idc: u8,
    pub separate_colour_plane: bool,
    pub pic_width: u32,
    pub pic_height: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_pic_order_cnt_lsb: u8,
    pub log2_min_luma_coding_block_size: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list_enabled: bool,
    pub amp_enabled: bool,
    pub sample_adaptive_offset_enabled: bool,
    pub pcm_enabled: bool,
    pub num_short_term_ref_pic_sets: u8,
    pub long_term_ref_pics_present: bool,
    pub sps_temporal_mvp_enabled: bool,
    pub strong_intra_smoothing_enabled: bool,
}

/// Conformance window cropping offsets.
#[derive(Debug, Clone, Copy)]
pub struct ConformanceWindow {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

/// Picture Parameter Set (PPS).
#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default_active: u8,
    pub num_ref_idx_l1_default_active: u8,
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass_enabled: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    pub num_tile_columns: u16,
    pub num_tile_rows: u16,
    pub loop_filter_across_tiles_enabled: bool,
    pub loop_filter_across_slices_enabled: bool,
    pub deblocking_filter_control_present: bool,
    pub deblocking_filter_override_enabled: bool,
    pub deblocking_filter_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u8,
}

/// Decoder context holding active parameter sets.
#[derive(Debug, Clone, Default)]
pub struct DecoderContext {
    pub vps: Vec<Option<Vps>>,
    pub sps: Vec<Option<Sps>>,
    pub pps: Vec<Option<Pps>>,
}

impl DecoderContext {
    /// Create a new context with capacity for parameter sets.
    pub fn new() -> Self {
        Self {
            vps: vec![None; 16],
            sps: vec![None; 16],
            pps: vec![None; 64],
        }
    }

    /// Get SPS by ID.
    pub fn get_sps(&self, id: u8) -> Result<&Sps, HevcError> {
        self.sps
            .get(id as usize)
            .and_then(|s| s.as_ref())
            .ok_or(HevcError::InvalidParameterSet(format!(
                "SPS {id} not found"
            )))
    }

    /// Get PPS by ID.
    pub fn get_pps(&self, id: u8) -> Result<&Pps, HevcError> {
        self.pps
            .get(id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(HevcError::InvalidParameterSet(format!(
                "PPS {id} not found"
            )))
    }
}

/// Parse profile_tier_level() — shared by VPS and SPS.
pub fn parse_profile_tier_level(
    r: &mut HevcBitReader,
    max_sub_layers: u8,
) -> Result<ProfileTierLevel, HevcError> {
    let general_profile_space = r.read_u(2)? as u8;
    let general_tier_flag = r.read_flag()?;
    let general_profile_idc = r.read_u(5)? as u8;

    let mut compat = 0u32;
    for i in 0..32 {
        if r.read_flag()? {
            compat |= 1 << i;
        }
    }

    // general_progressive_source_flag through general_reserved_zero_43bits
    r.skip(48)?; // 48 constraint bits

    let general_level_idc = r.read_u(8)? as u8;

    // Sub-layer profile/level info (skip for simplicity — HEIC usually has 1 sub-layer)
    if max_sub_layers > 1 {
        let mut sub_layer_profile_present = [false; 7];
        let mut sub_layer_level_present = [false; 7];
        for i in 0..(max_sub_layers - 1) as usize {
            sub_layer_profile_present[i] = r.read_flag()?;
            sub_layer_level_present[i] = r.read_flag()?;
        }
        // Reserved bits for unused sub-layers
        if max_sub_layers < 8 {
            for _ in (max_sub_layers - 1)..7 {
                r.skip(2)?;
            }
        }
        // Skip sub-layer profile/level data
        for i in 0..(max_sub_layers - 1) as usize {
            if sub_layer_profile_present[i] {
                r.skip(88)?; // profile_space(2)+tier(1)+profile_idc(5)+compat(32)+constraints(48)
            }
            if sub_layer_level_present[i] {
                r.skip(8)?; // level_idc
            }
        }
    }

    Ok(ProfileTierLevel {
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags: compat,
        general_level_idc,
    })
}

/// Parse VPS from RBSP data.
pub fn parse_vps(rbsp: &[u8]) -> Result<Vps, HevcError> {
    let mut r = HevcBitReader::new(rbsp);

    let vps_id = r.read_u(4)? as u8;
    r.skip(2)?; // vps_reserved_three_2bits
    let max_layers = r.read_u(6)? as u8 + 1;
    let max_sub_layers = r.read_u(3)? as u8 + 1;
    let temporal_id_nesting = r.read_flag()?;
    r.skip(16)?; // vps_reserved_0xffff_16bits

    let profile = parse_profile_tier_level(&mut r, max_sub_layers)?;

    Ok(Vps {
        vps_id,
        max_layers,
        max_sub_layers,
        temporal_id_nesting,
        profile,
    })
}

/// Parse SPS from RBSP data.
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, HevcError> {
    let mut r = HevcBitReader::new(rbsp);

    let vps_id = r.read_u(4)? as u8;
    let max_sub_layers = r.read_u(3)? as u8 + 1;
    let _temporal_id_nesting = r.read_flag()?;

    let profile = parse_profile_tier_level(&mut r, max_sub_layers)?;

    let sps_id = r.read_ue()? as u8;
    let chroma_format_idc = r.read_ue()? as u8;

    let separate_colour_plane = if chroma_format_idc == 3 {
        r.read_flag()?
    } else {
        false
    };

    let pic_width = r.read_ue()?;
    let pic_height = r.read_ue()?;

    let conformance_window = if r.read_flag()? {
        Some(ConformanceWindow {
            left: r.read_ue()?,
            right: r.read_ue()?,
            top: r.read_ue()?,
            bottom: r.read_ue()?,
        })
    } else {
        None
    };

    let bit_depth_luma = r.read_ue()? as u8 + 8;
    let bit_depth_chroma = r.read_ue()? as u8 + 8;

    let log2_max_pic_order_cnt_lsb = r.read_ue()? as u8 + 4;

    // sub_layer_ordering_info
    let sub_layer_ordering_info_present = r.read_flag()?;
    let start = if sub_layer_ordering_info_present {
        0
    } else {
        max_sub_layers - 1
    };
    for _ in start..max_sub_layers {
        let _max_dec_pic_buffering = r.read_ue()?;
        let _max_num_reorder_pics = r.read_ue()?;
        let _max_latency_increase = r.read_ue()?;
    }

    let log2_min_luma_coding_block_size = r.read_ue()? as u8 + 3;
    let log2_diff_max_min_luma_coding_block_size = r.read_ue()? as u8;
    let log2_min_luma_transform_block_size = r.read_ue()? as u8 + 2;
    let log2_diff_max_min_luma_transform_block_size = r.read_ue()? as u8;
    let max_transform_hierarchy_depth_inter = r.read_ue()? as u8;
    let max_transform_hierarchy_depth_intra = r.read_ue()? as u8;

    let scaling_list_enabled = r.read_flag()?;
    if scaling_list_enabled && r.read_flag()? {
        // scaling_list_data() — skip for now (parsed in transform track)
        skip_scaling_list_data(&mut r)?;
    }

    let amp_enabled = r.read_flag()?;
    let sample_adaptive_offset_enabled = r.read_flag()?;

    let pcm_enabled = r.read_flag()?;
    if pcm_enabled {
        let _pcm_bit_depth_luma = r.read_u(4)?;
        let _pcm_bit_depth_chroma = r.read_u(4)?;
        let _log2_min_pcm_cb_size = r.read_ue()?;
        let _log2_diff_max_min_pcm_cb_size = r.read_ue()?;
        let _pcm_loop_filter_disabled = r.read_flag()?;
    }

    let num_short_term_ref_pic_sets = r.read_ue()? as u8;
    // Skip short-term ref pic sets (not needed for I-frame decode)
    for i in 0..num_short_term_ref_pic_sets {
        skip_short_term_ref_pic_set(&mut r, i, num_short_term_ref_pic_sets)?;
    }

    let long_term_ref_pics_present = r.read_flag()?;
    if long_term_ref_pics_present {
        let num_long_term_ref_pics = r.read_ue()?;
        for _ in 0..num_long_term_ref_pics {
            let _lt_ref_pic_poc_lsb = r.read_u(log2_max_pic_order_cnt_lsb)?;
            let _used_by_curr_pic = r.read_flag()?;
        }
    }

    let sps_temporal_mvp_enabled = r.read_flag()?;
    let strong_intra_smoothing_enabled = r.read_flag()?;

    Ok(Sps {
        sps_id,
        vps_id,
        max_sub_layers,
        profile,
        chroma_format_idc,
        separate_colour_plane,
        pic_width,
        pic_height,
        conformance_window,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_pic_order_cnt_lsb,
        log2_min_luma_coding_block_size,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled,
        amp_enabled,
        sample_adaptive_offset_enabled,
        pcm_enabled,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present,
        sps_temporal_mvp_enabled,
        strong_intra_smoothing_enabled,
    })
}

/// Parse PPS from RBSP data.
pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, HevcError> {
    let mut r = HevcBitReader::new(rbsp);

    let pps_id = r.read_ue()? as u8;
    let sps_id = r.read_ue()? as u8;
    let dependent_slice_segments_enabled = r.read_flag()?;
    let output_flag_present = r.read_flag()?;
    let num_extra_slice_header_bits = r.read_u(3)? as u8;
    let sign_data_hiding_enabled = r.read_flag()?;
    let cabac_init_present = r.read_flag()?;

    let num_ref_idx_l0_default_active = r.read_ue()? as u8 + 1;
    let num_ref_idx_l1_default_active = r.read_ue()? as u8 + 1;

    let init_qp = r.read_se()? + 26;

    let constrained_intra_pred = r.read_flag()?;
    let transform_skip_enabled = r.read_flag()?;

    let cu_qp_delta_enabled = r.read_flag()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled {
        r.read_ue()? as u8
    } else {
        0
    };

    let cb_qp_offset = r.read_se()?;
    let cr_qp_offset = r.read_se()?;
    let slice_chroma_qp_offsets_present = r.read_flag()?;
    let weighted_pred = r.read_flag()?;
    let weighted_bipred = r.read_flag()?;
    let transquant_bypass_enabled = r.read_flag()?;
    let tiles_enabled = r.read_flag()?;
    let entropy_coding_sync_enabled = r.read_flag()?;

    let mut num_tile_columns = 1u16;
    let mut num_tile_rows = 1u16;
    let mut loop_filter_across_tiles_enabled = true;

    if tiles_enabled {
        num_tile_columns = r.read_ue()? as u16 + 1;
        num_tile_rows = r.read_ue()? as u16 + 1;
        let uniform_spacing = r.read_flag()?;
        if !uniform_spacing {
            for _ in 0..num_tile_columns - 1 {
                let _column_width = r.read_ue()?;
            }
            for _ in 0..num_tile_rows - 1 {
                let _row_height = r.read_ue()?;
            }
        }
        loop_filter_across_tiles_enabled = r.read_flag()?;
    }

    let loop_filter_across_slices_enabled = r.read_flag()?;

    let deblocking_filter_control_present = r.read_flag()?;
    let mut deblocking_filter_override_enabled = false;
    let mut deblocking_filter_disabled = false;
    let mut beta_offset = 0i32;
    let mut tc_offset = 0i32;

    if deblocking_filter_control_present {
        deblocking_filter_override_enabled = r.read_flag()?;
        deblocking_filter_disabled = r.read_flag()?;
        if !deblocking_filter_disabled {
            beta_offset = r.read_se()? * 2;
            tc_offset = r.read_se()? * 2;
        }
    }

    // Skip remaining PPS fields that are less common
    let _scaling_list_data_present = r.read_flag()?;
    // If scaling list present, would need to parse — handled by transform track

    let lists_modification_present = r.read_flag()?;
    let log2_parallel_merge_level = r.read_ue()? as u8 + 2;

    Ok(Pps {
        pps_id,
        sps_id,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        cabac_init_present,
        num_ref_idx_l0_default_active,
        num_ref_idx_l1_default_active,
        init_qp,
        constrained_intra_pred,
        transform_skip_enabled,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        weighted_pred,
        weighted_bipred,
        transquant_bypass_enabled,
        tiles_enabled,
        entropy_coding_sync_enabled,
        num_tile_columns,
        num_tile_rows,
        loop_filter_across_tiles_enabled,
        loop_filter_across_slices_enabled,
        deblocking_filter_control_present,
        deblocking_filter_override_enabled,
        deblocking_filter_disabled,
        beta_offset,
        tc_offset,
        lists_modification_present,
        log2_parallel_merge_level,
    })
}

/// Parse parameter sets from hvcC NAL arrays into a DecoderContext.
pub fn parse_hvcc_params(
    nal_arrays: &[rasmcore_isobmff::NalArray],
) -> Result<DecoderContext, HevcError> {
    let mut ctx = DecoderContext::new();

    for array in nal_arrays {
        for nal_data in &array.nal_units {
            if nal_data.len() < 2 {
                continue;
            }
            let nal_type = (nal_data[0] >> 1) & 0x3F;
            let rbsp = &nal_data[2..]; // hvcC NALs don't have emulation prevention bytes

            match nal_type {
                32 => {
                    // VPS
                    let vps = parse_vps(rbsp)?;
                    let id = vps.vps_id as usize;
                    if id < ctx.vps.len() {
                        ctx.vps[id] = Some(vps);
                    }
                }
                33 => {
                    // SPS
                    let sps = parse_sps(rbsp)?;
                    let id = sps.sps_id as usize;
                    if id < ctx.sps.len() {
                        ctx.sps[id] = Some(sps);
                    }
                }
                34 => {
                    // PPS
                    let pps = parse_pps(rbsp)?;
                    let id = pps.pps_id as usize;
                    if id < ctx.pps.len() {
                        ctx.pps[id] = Some(pps);
                    }
                }
                _ => {} // Skip other NAL types (SEI, etc.)
            }
        }
    }

    Ok(ctx)
}

/// Skip scaling_list_data() in the bitstream.
fn skip_scaling_list_data(r: &mut HevcBitReader) -> Result<(), HevcError> {
    for size_id in 0..4 {
        let matrix_count = if size_id == 3 { 2 } else { 6 };
        for _ in 0..matrix_count {
            let pred_mode_flag = r.read_flag()?;
            if !pred_mode_flag {
                let _delta = r.read_ue()?;
            } else {
                let coeff_num = std::cmp::min(64, 1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    let _dc_coef = r.read_se()?;
                }
                for _ in 0..coeff_num {
                    let _delta = r.read_se()?;
                }
            }
        }
    }
    Ok(())
}

/// Skip short_term_ref_pic_set() in the bitstream.
fn skip_short_term_ref_pic_set(
    r: &mut HevcBitReader,
    idx: u8,
    num_sets: u8,
) -> Result<(), HevcError> {
    let inter_ref_pic_set_prediction = if idx != 0 { r.read_flag()? } else { false };

    if inter_ref_pic_set_prediction {
        if idx == num_sets {
            let _delta_idx = r.read_ue()?;
        }
        let _delta_rps_sign = r.read_flag()?;
        let _abs_delta_rps = r.read_ue()?;
        // We'd need to know the previous set's size to skip correctly.
        // For HEIC (I-frame only), short-term ref pic sets are typically empty.
        // Skip a reasonable approximation.
    } else {
        let num_negative_pics = r.read_ue()?;
        let num_positive_pics = r.read_ue()?;
        for _ in 0..num_negative_pics {
            let _delta_poc = r.read_ue()?;
            let _used = r.read_flag()?;
        }
        for _ in 0..num_positive_pics {
            let _delta_poc = r.read_ue()?;
            let _used = r.read_flag()?;
        }
    }
    Ok(())
}

impl Sps {
    /// Compute the CTU (Coding Tree Unit) size in pixels.
    pub fn ctu_size(&self) -> u32 {
        1 << (self.log2_min_luma_coding_block_size + self.log2_diff_max_min_luma_coding_block_size)
    }

    /// Compute the minimum CU size in pixels.
    pub fn min_cb_size(&self) -> u32 {
        1 << self.log2_min_luma_coding_block_size
    }

    /// Compute the number of CTUs in the horizontal direction.
    pub fn pic_width_in_ctus(&self) -> u32 {
        self.pic_width.div_ceil(self.ctu_size())
    }

    /// Compute the number of CTUs in the vertical direction.
    pub fn pic_height_in_ctus(&self) -> u32 {
        self.pic_height.div_ceil(self.ctu_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SPS RBSP for testing.
    /// Uses known values that produce a valid parse.
    fn build_test_sps_rbsp() -> Vec<u8> {
        // Encode field by field into a bit buffer
        // Helper to push bits
        fn push_bits(bits: &mut Vec<bool>, val: u32, n: u8) {
            for i in (0..n).rev() {
                bits.push((val >> i) & 1 != 0);
            }
        }
        fn push_ue(bits: &mut Vec<bool>, val: u32) {
            if val == 0 {
                bits.push(true);
                return;
            }
            let code = val + 1;
            let len = 32 - code.leading_zeros();
            for _ in 0..len - 1 {
                bits.push(false);
            }
            for i in (0..len).rev() {
                bits.push((code >> i) & 1 != 0);
            }
        }
        let mut b: Vec<bool> = Vec::new();

        // sps_video_parameter_set_id = 0 (4 bits)
        push_bits(&mut b, 0, 4);
        // sps_max_sub_layers_minus1 = 0 (3 bits)
        push_bits(&mut b, 0, 3);
        // sps_temporal_id_nesting_flag = 1 (1 bit)
        push_bits(&mut b, 1, 1);

        // profile_tier_level(maxNumSubLayersMinus1=0):
        push_bits(&mut b, 0, 2); // general_profile_space
        push_bits(&mut b, 0, 1); // general_tier_flag
        push_bits(&mut b, 1, 5); // general_profile_idc = 1 (Main)
        for _ in 0..32 {
            b.push(false); // general_profile_compatibility_flags
        }
        for _ in 0..48 {
            b.push(false); // constraint flags
        }
        push_bits(&mut b, 120, 8); // general_level_idc = 120 (Level 4.0)

        // sps_seq_parameter_set_id = 0
        push_ue(&mut b, 0);
        // chroma_format_idc = 1 (4:2:0)
        push_ue(&mut b, 1);
        // pic_width_in_luma_samples = 1920
        push_ue(&mut b, 1920);
        // pic_height_in_luma_samples = 1080
        push_ue(&mut b, 1080);

        // conformance_window_flag = 0
        b.push(false);

        // bit_depth_luma_minus8 = 0
        push_ue(&mut b, 0);
        // bit_depth_chroma_minus8 = 0
        push_ue(&mut b, 0);
        // log2_max_pic_order_cnt_lsb_minus4 = 0
        push_ue(&mut b, 0);

        // sps_sub_layer_ordering_info_present_flag = 0
        b.push(false);
        // For max_sub_layers (1 iteration): max_dec_pic_buffering=1, max_num_reorder=0, max_latency=0
        push_ue(&mut b, 1);
        push_ue(&mut b, 0);
        push_ue(&mut b, 0);

        // log2_min_luma_coding_block_size_minus3 = 0 -> min CB = 8
        push_ue(&mut b, 0);
        // log2_diff_max_min_luma_coding_block_size = 3 -> CTU = 64
        push_ue(&mut b, 3);
        // log2_min_luma_transform_block_size_minus2 = 0 -> min TU = 4
        push_ue(&mut b, 0);
        // log2_diff_max_min_luma_transform_block_size = 3 -> max TU = 32
        push_ue(&mut b, 3);
        // max_transform_hierarchy_depth_inter = 0
        push_ue(&mut b, 0);
        // max_transform_hierarchy_depth_intra = 0
        push_ue(&mut b, 0);

        // scaling_list_enabled_flag = 0
        b.push(false);
        // amp_enabled_flag = 0
        b.push(false);
        // sample_adaptive_offset_enabled_flag = 1
        b.push(true);
        // pcm_enabled_flag = 0
        b.push(false);

        // num_short_term_ref_pic_sets = 0
        push_ue(&mut b, 0);

        // long_term_ref_pics_present_flag = 0
        b.push(false);
        // sps_temporal_mvp_enabled_flag = 0
        b.push(false);
        // strong_intra_smoothing_enabled_flag = 1
        b.push(true);

        // Convert bits to bytes
        let mut bytes = Vec::new();
        for chunk in b.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << (7 - i);
                }
            }
            bytes.push(byte);
        }
        bytes
    }

    #[test]
    fn parse_test_sps() {
        let rbsp = build_test_sps_rbsp();
        let sps = parse_sps(&rbsp).unwrap();

        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.vps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1); // 4:2:0
        assert_eq!(sps.pic_width, 1920);
        assert_eq!(sps.pic_height, 1080);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        assert_eq!(sps.profile.general_profile_idc, 1); // Main
        assert_eq!(sps.profile.general_level_idc, 120); // Level 4.0
        assert_eq!(sps.ctu_size(), 64);
        assert_eq!(sps.min_cb_size(), 8);
        assert!(sps.sample_adaptive_offset_enabled);
        assert!(sps.strong_intra_smoothing_enabled);
    }

    #[test]
    fn sps_ctu_calculations() {
        let rbsp = build_test_sps_rbsp();
        let sps = parse_sps(&rbsp).unwrap();

        // 1920 / 64 = 30 CTUs wide
        assert_eq!(sps.pic_width_in_ctus(), 30);
        // 1080 / 64 = 16.875 -> 17 CTUs tall
        assert_eq!(sps.pic_height_in_ctus(), 17);
    }

    #[test]
    fn parse_minimal_pps() {
        // Build a minimal PPS RBSP
        let mut b: Vec<bool> = Vec::new();
        fn push_bits(bits: &mut Vec<bool>, val: u32, n: u8) {
            for i in (0..n).rev() {
                bits.push((val >> i) & 1 != 0);
            }
        }
        fn push_ue(bits: &mut Vec<bool>, val: u32) {
            if val == 0 {
                bits.push(true);
                return;
            }
            let code = val + 1;
            let len = 32 - code.leading_zeros();
            for _ in 0..len - 1 {
                bits.push(false);
            }
            for i in (0..len).rev() {
                bits.push((code >> i) & 1 != 0);
            }
        }
        fn push_se(bits: &mut Vec<bool>, val: i32) {
            let ue = if val > 0 {
                (val as u32) * 2 - 1
            } else if val < 0 {
                ((-val) as u32) * 2
            } else {
                0
            };
            push_ue(bits, ue);
        }

        push_ue(&mut b, 0); // pps_id = 0
        push_ue(&mut b, 0); // sps_id = 0
        b.push(false); // dependent_slice_segments_enabled
        b.push(false); // output_flag_present
        push_bits(&mut b, 0, 3); // num_extra_slice_header_bits
        b.push(false); // sign_data_hiding
        b.push(false); // cabac_init_present
        push_ue(&mut b, 0); // num_ref_idx_l0 - 1 = 0
        push_ue(&mut b, 0); // num_ref_idx_l1 - 1 = 0
        push_se(&mut b, 0); // init_qp_minus26 = 0
        b.push(false); // constrained_intra_pred
        b.push(false); // transform_skip_enabled
        b.push(false); // cu_qp_delta_enabled
        push_se(&mut b, 0); // cb_qp_offset
        push_se(&mut b, 0); // cr_qp_offset
        b.push(false); // slice_chroma_qp_offsets_present
        b.push(false); // weighted_pred
        b.push(false); // weighted_bipred
        b.push(false); // transquant_bypass
        b.push(false); // tiles_enabled
        b.push(false); // entropy_coding_sync
        b.push(false); // loop_filter_across_slices
        b.push(false); // deblocking_filter_control_present
        b.push(false); // scaling_list_data_present
        b.push(false); // lists_modification_present
        push_ue(&mut b, 0); // log2_parallel_merge_level - 2 = 0

        let mut bytes = Vec::new();
        for chunk in b.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << (7 - i);
                }
            }
            bytes.push(byte);
        }

        let pps = parse_pps(&bytes).unwrap();
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert_eq!(pps.init_qp, 26); // 0 + 26
        assert!(!pps.tiles_enabled);
        assert_eq!(pps.num_tile_columns, 1);
        assert_eq!(pps.num_tile_rows, 1);
    }

    #[test]
    fn decoder_context_lookup() {
        let mut ctx = DecoderContext::new();

        let rbsp = build_test_sps_rbsp();
        let sps = parse_sps(&rbsp).unwrap();
        ctx.sps[0] = Some(sps);

        let sps_ref = ctx.get_sps(0).unwrap();
        assert_eq!(sps_ref.pic_width, 1920);

        let err = ctx.get_sps(5);
        assert!(err.is_err());
    }
}
