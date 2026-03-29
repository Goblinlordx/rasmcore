//! VPS/SPS/PPS bitstream serialization — write parameter sets to RBSP.
//!
//! Direct counterpart to `params.rs` (parser). Each write function produces
//! RBSP bytes that can be parsed back by the corresponding parse function.
//!
//! Ref: x265 4.1 encoder/entropy.cpp — codeVPS, codeSPS, codePPS
//! Ref: ITU-T H.265 Sections 7.3.2.1 (VPS), 7.3.2.2 (SPS), 7.3.2.3 (PPS)

use super::bitwrite::BitstreamWriter;
use crate::params::{Pps, ProfileTierLevel, Sps, Vps};
#[cfg(test)]
use crate::params::ConformanceWindow;

/// Write profile_tier_level() to the bitstream.
///
/// Ref: ITU-T H.265 Section 7.3.3 — profile_tier_level()
/// Ref: x265 4.1 encoder/entropy.cpp — codeProfileTier
fn write_profile_tier_level(w: &mut BitstreamWriter, ptl: &ProfileTierLevel, max_sub_layers: u8) {
    w.write_bits(ptl.general_profile_space as u32, 2);
    w.write_flag(ptl.general_tier_flag);
    w.write_bits(ptl.general_profile_idc as u32, 5);

    // general_profile_compatibility_flags — 32 individual flag bits
    for i in 0..32 {
        w.write_flag((ptl.general_profile_compatibility_flags >> i) & 1 != 0);
    }

    // general constraint flags — 48 bits (6 bytes)
    // For Main Still Picture profile: progressive_source=1, interlaced_source=0,
    // non_packed_constraint=1, frame_only_constraint=1, rest=0
    // Simplified: write all zeros for the 48 constraint bits.
    // The exact values depend on the profile; for roundtrip testing we write
    // what x265 writes for Main Still Picture.
    w.write_flag(true); // general_progressive_source_flag
    w.write_flag(false); // general_interlaced_source_flag
    w.write_flag(true); // general_non_packed_constraint_flag
    w.write_flag(true); // general_frame_only_constraint_flag
    // remaining 44 constraint bits = 0
    w.write_bits(0, 32);
    w.write_bits(0, 12);

    w.write_bits(ptl.general_level_idc as u32, 8);

    // Sub-layer profile/level info
    if max_sub_layers > 1 {
        for _ in 0..(max_sub_layers - 1) {
            w.write_flag(false); // sub_layer_profile_present_flag
            w.write_flag(false); // sub_layer_level_present_flag
        }
        // Padding for unused sub-layers (up to 8)
        if max_sub_layers < 8 {
            for _ in (max_sub_layers - 1)..7 {
                w.write_bits(0, 2); // reserved_zero_2bits
            }
        }
    }
}

/// Write VPS (Video Parameter Set) to RBSP bytes.
///
/// Ref: ITU-T H.265 Section 7.3.2.1
/// Ref: x265 4.1 encoder/entropy.cpp — codeVPS
pub fn write_vps(vps: &Vps) -> Vec<u8> {
    let mut w = BitstreamWriter::with_capacity(64);

    w.write_bits(vps.vps_id as u32, 4);
    w.write_bits(3, 2); // vps_reserved_three_2bits
    w.write_bits((vps.max_layers - 1) as u32, 6);
    w.write_bits((vps.max_sub_layers - 1) as u32, 3);
    w.write_flag(vps.temporal_id_nesting);
    w.write_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits

    write_profile_tier_level(&mut w, &vps.profile, vps.max_sub_layers);

    // vps_sub_layer_ordering_info_present_flag
    w.write_flag(true);
    // For each sub-layer (just 1 for single-layer):
    w.write_ue(1); // vps_max_dec_pic_buffering_minus1
    w.write_ue(0); // vps_max_num_reorder_pics
    w.write_ue(0); // vps_max_latency_increase_plus1

    w.write_bits(0, 6); // vps_max_layer_id = 0
    w.write_ue(0); // vps_num_layer_sets_minus1 = 0

    w.write_flag(false); // vps_timing_info_present_flag
    w.write_flag(false); // vps_extension_flag

    w.write_rbsp_trailing_bits();
    w.finish()
}

/// Write SPS (Sequence Parameter Set) to RBSP bytes.
///
/// Ref: ITU-T H.265 Section 7.3.2.2
/// Ref: x265 4.1 encoder/entropy.cpp — codeSPS
pub fn write_sps(sps: &Sps) -> Vec<u8> {
    let mut w = BitstreamWriter::with_capacity(256);

    w.write_bits(sps.vps_id as u32, 4);
    w.write_bits((sps.max_sub_layers - 1) as u32, 3);
    w.write_flag(true); // sps_temporal_id_nesting_flag (always true for single-layer)

    write_profile_tier_level(&mut w, &sps.profile, sps.max_sub_layers);

    w.write_ue(sps.sps_id as u32);
    w.write_ue(sps.chroma_format_idc as u32);

    if sps.chroma_format_idc == 3 {
        w.write_flag(sps.separate_colour_plane);
    }

    w.write_ue(sps.pic_width);
    w.write_ue(sps.pic_height);

    // Conformance window
    let has_window = sps.conformance_window.is_some();
    w.write_flag(has_window);
    if let Some(cw) = &sps.conformance_window {
        w.write_ue(cw.left);
        w.write_ue(cw.right);
        w.write_ue(cw.top);
        w.write_ue(cw.bottom);
    }

    w.write_ue((sps.bit_depth_luma - 8) as u32);
    w.write_ue((sps.bit_depth_chroma - 8) as u32);
    w.write_ue((sps.log2_max_pic_order_cnt_lsb - 4) as u32);

    // sub_layer_ordering_info_present_flag — write for all sub-layers
    w.write_flag(true);
    for _ in 0..sps.max_sub_layers {
        w.write_ue(1); // sps_max_dec_pic_buffering_minus1
        w.write_ue(0); // sps_max_num_reorder_pics
        w.write_ue(0); // sps_max_latency_increase_plus1
    }

    w.write_ue((sps.log2_min_luma_coding_block_size - 3) as u32);
    w.write_ue(sps.log2_diff_max_min_luma_coding_block_size as u32);
    w.write_ue((sps.log2_min_luma_transform_block_size - 2) as u32);
    w.write_ue(sps.log2_diff_max_min_luma_transform_block_size as u32);
    w.write_ue(sps.max_transform_hierarchy_depth_inter as u32);
    w.write_ue(sps.max_transform_hierarchy_depth_intra as u32);

    w.write_flag(sps.scaling_list_enabled);
    if sps.scaling_list_enabled {
        w.write_flag(false); // scaling_list_data_present_flag = 0 (use default)
    }

    w.write_flag(sps.amp_enabled);
    w.write_flag(sps.sample_adaptive_offset_enabled);
    w.write_flag(sps.pcm_enabled);
    // Note: if pcm_enabled, more fields would follow — skipped for I-frame encoder

    w.write_ue(sps.num_short_term_ref_pic_sets as u32);
    // No short-term ref pic sets for I-frame-only

    w.write_flag(sps.long_term_ref_pics_present);
    // No long-term ref pics for I-frame-only

    w.write_flag(sps.sps_temporal_mvp_enabled);
    w.write_flag(sps.strong_intra_smoothing_enabled);

    w.write_flag(false); // vui_parameters_present_flag
    w.write_flag(false); // sps_extension_flag

    w.write_rbsp_trailing_bits();
    w.finish()
}

/// Write PPS (Picture Parameter Set) to RBSP bytes.
///
/// Ref: ITU-T H.265 Section 7.3.2.3
/// Ref: x265 4.1 encoder/entropy.cpp — codePPS
pub fn write_pps(pps: &Pps) -> Vec<u8> {
    let mut w = BitstreamWriter::with_capacity(128);

    w.write_ue(pps.pps_id as u32);
    w.write_ue(pps.sps_id as u32);
    w.write_flag(pps.dependent_slice_segments_enabled);
    w.write_flag(pps.output_flag_present);
    w.write_bits(pps.num_extra_slice_header_bits as u32, 3);
    w.write_flag(pps.sign_data_hiding_enabled);
    w.write_flag(pps.cabac_init_present);

    w.write_ue((pps.num_ref_idx_l0_default_active - 1) as u32);
    w.write_ue((pps.num_ref_idx_l1_default_active - 1) as u32);

    w.write_se(pps.init_qp - 26);

    w.write_flag(pps.constrained_intra_pred);
    w.write_flag(pps.transform_skip_enabled);
    w.write_flag(pps.cu_qp_delta_enabled);
    if pps.cu_qp_delta_enabled {
        w.write_ue(pps.diff_cu_qp_delta_depth as u32);
    }

    w.write_se(pps.cb_qp_offset);
    w.write_se(pps.cr_qp_offset);
    w.write_flag(pps.slice_chroma_qp_offsets_present);
    w.write_flag(pps.weighted_pred);
    w.write_flag(pps.weighted_bipred);
    w.write_flag(pps.transquant_bypass_enabled);
    w.write_flag(pps.tiles_enabled);
    w.write_flag(pps.entropy_coding_sync_enabled);

    if pps.tiles_enabled {
        w.write_ue((pps.num_tile_columns - 1) as u32);
        w.write_ue((pps.num_tile_rows - 1) as u32);
        w.write_flag(true); // uniform_spacing_flag
        w.write_flag(pps.loop_filter_across_tiles_enabled);
    }

    w.write_flag(pps.loop_filter_across_slices_enabled);

    w.write_flag(pps.deblocking_filter_control_present);
    if pps.deblocking_filter_control_present {
        w.write_flag(pps.deblocking_filter_override_enabled);
        w.write_flag(pps.deblocking_filter_disabled);
        if !pps.deblocking_filter_disabled {
            w.write_se(pps.beta_offset / 2);
            w.write_se(pps.tc_offset / 2);
        }
    }

    w.write_flag(false); // pps_scaling_list_data_present_flag
    w.write_flag(pps.lists_modification_present);
    w.write_ue(pps.log2_parallel_merge_level as u32);
    w.write_flag(false); // slice_segment_header_extension_present_flag
    w.write_flag(false); // pps_extension_flag

    w.write_rbsp_trailing_bits();
    w.finish()
}

/// Create default VPS for I-frame-only HEIC encoding.
pub fn default_vps() -> Vps {
    Vps {
        vps_id: 0,
        max_layers: 1,
        max_sub_layers: 1,
        temporal_id_nesting: true,
        profile: default_profile(),
    }
}

/// Create default SPS for I-frame-only HEIC encoding.
pub fn default_sps(width: u32, height: u32) -> Sps {
    Sps {
        sps_id: 0,
        vps_id: 0,
        max_sub_layers: 1,
        profile: default_profile(),
        chroma_format_idc: 1, // 4:2:0
        separate_colour_plane: false,
        pic_width: width,
        pic_height: height,
        conformance_window: None,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        log2_max_pic_order_cnt_lsb: 4,
        log2_min_luma_coding_block_size: 4, // min CU = 16
        log2_diff_max_min_luma_coding_block_size: 1, // max CU = 32
        log2_min_luma_transform_block_size: 2, // min TU = 4
        log2_diff_max_min_luma_transform_block_size: 3, // max TU = 32
        max_transform_hierarchy_depth_inter: 0,
        max_transform_hierarchy_depth_intra: 1,
        scaling_list_enabled: false,
        scaling_list: None,
        amp_enabled: false,
        sample_adaptive_offset_enabled: false,
        pcm_enabled: false,
        num_short_term_ref_pic_sets: 0,
        long_term_ref_pics_present: false,
        sps_temporal_mvp_enabled: false,
        strong_intra_smoothing_enabled: true,
    }
}

/// Create default PPS for I-frame-only HEIC encoding.
pub fn default_pps(qp: i32) -> Pps {
    Pps {
        pps_id: 0,
        sps_id: 0,
        dependent_slice_segments_enabled: false,
        output_flag_present: false,
        num_extra_slice_header_bits: 0,
        sign_data_hiding_enabled: true,
        cabac_init_present: false,
        num_ref_idx_l0_default_active: 1,
        num_ref_idx_l1_default_active: 1,
        init_qp: qp,
        constrained_intra_pred: false,
        transform_skip_enabled: false,
        cu_qp_delta_enabled: false,
        diff_cu_qp_delta_depth: 0,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        slice_chroma_qp_offsets_present: false,
        weighted_pred: false,
        weighted_bipred: false,
        transquant_bypass_enabled: false,
        tiles_enabled: false,
        entropy_coding_sync_enabled: false, // WPP disabled — encoder is single-threaded
        num_tile_columns: 1,
        num_tile_rows: 1,
        loop_filter_across_tiles_enabled: true,
        loop_filter_across_slices_enabled: true,
        deblocking_filter_control_present: true,
        deblocking_filter_override_enabled: false,
        deblocking_filter_disabled: false,
        beta_offset: 0,
        tc_offset: 0,
        lists_modification_present: false,
        log2_parallel_merge_level: 2,
    }
}

/// Default profile for Main Still Picture.
fn default_profile() -> ProfileTierLevel {
    ProfileTierLevel {
        general_profile_space: 0,
        general_tier_flag: false,
        general_profile_idc: 3, // Main Still Picture
        general_profile_compatibility_flags: 1 << 3, // bit 3 = Main Still Picture
        general_level_idc: 60, // Level 2.0 (sufficient for small stills)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params;

    #[test]
    fn vps_roundtrip() {
        let vps = default_vps();
        let rbsp = write_vps(&vps);
        let parsed = params::parse_vps(&rbsp).expect("VPS parse failed");

        assert_eq!(parsed.vps_id, vps.vps_id);
        assert_eq!(parsed.max_layers, vps.max_layers);
        assert_eq!(parsed.max_sub_layers, vps.max_sub_layers);
        assert_eq!(parsed.temporal_id_nesting, vps.temporal_id_nesting);
        assert_eq!(
            parsed.profile.general_profile_idc,
            vps.profile.general_profile_idc
        );
        assert_eq!(
            parsed.profile.general_level_idc,
            vps.profile.general_level_idc
        );
    }

    #[test]
    fn sps_roundtrip() {
        let sps = default_sps(256, 256);
        let rbsp = write_sps(&sps);
        let parsed = params::parse_sps(&rbsp).expect("SPS parse failed");

        assert_eq!(parsed.sps_id, sps.sps_id);
        assert_eq!(parsed.vps_id, sps.vps_id);
        assert_eq!(parsed.chroma_format_idc, sps.chroma_format_idc);
        assert_eq!(parsed.pic_width, sps.pic_width);
        assert_eq!(parsed.pic_height, sps.pic_height);
        assert_eq!(parsed.bit_depth_luma, sps.bit_depth_luma);
        assert_eq!(parsed.bit_depth_chroma, sps.bit_depth_chroma);
        assert_eq!(
            parsed.log2_min_luma_coding_block_size,
            sps.log2_min_luma_coding_block_size
        );
        assert_eq!(
            parsed.log2_diff_max_min_luma_coding_block_size,
            sps.log2_diff_max_min_luma_coding_block_size
        );
        assert_eq!(
            parsed.log2_min_luma_transform_block_size,
            sps.log2_min_luma_transform_block_size
        );
        assert_eq!(
            parsed.max_transform_hierarchy_depth_intra,
            sps.max_transform_hierarchy_depth_intra
        );
        assert_eq!(
            parsed.strong_intra_smoothing_enabled,
            sps.strong_intra_smoothing_enabled
        );
        assert_eq!(parsed.scaling_list_enabled, sps.scaling_list_enabled);
        assert_eq!(parsed.pcm_enabled, sps.pcm_enabled);
    }

    #[test]
    fn sps_roundtrip_various_sizes() {
        for (w, h) in [(64, 64), (128, 128), (256, 256), (1920, 1080), (3840, 2160)] {
            let sps = default_sps(w, h);
            let rbsp = write_sps(&sps);
            let parsed = params::parse_sps(&rbsp).unwrap();
            assert_eq!(parsed.pic_width, w, "width mismatch for {w}x{h}");
            assert_eq!(parsed.pic_height, h, "height mismatch for {w}x{h}");
        }
    }

    #[test]
    fn pps_roundtrip() {
        let pps = default_pps(26);
        let rbsp = write_pps(&pps);
        let parsed = params::parse_pps(&rbsp).expect("PPS parse failed");

        assert_eq!(parsed.pps_id, pps.pps_id);
        assert_eq!(parsed.sps_id, pps.sps_id);
        assert_eq!(parsed.init_qp, pps.init_qp);
        assert_eq!(
            parsed.sign_data_hiding_enabled,
            pps.sign_data_hiding_enabled
        );
        assert_eq!(
            parsed.entropy_coding_sync_enabled,
            pps.entropy_coding_sync_enabled
        );
        assert_eq!(
            parsed.deblocking_filter_disabled,
            pps.deblocking_filter_disabled
        );
        assert_eq!(parsed.beta_offset, pps.beta_offset);
        assert_eq!(parsed.tc_offset, pps.tc_offset);
    }

    #[test]
    fn pps_roundtrip_various_qp() {
        for qp in [22, 26, 30, 37, 51] {
            let pps = default_pps(qp);
            let rbsp = write_pps(&pps);
            let parsed = params::parse_pps(&rbsp).unwrap();
            assert_eq!(parsed.init_qp, qp, "QP mismatch for {qp}");
        }
    }

    #[test]
    fn sps_with_conformance_window() {
        let mut sps = default_sps(1920, 1080);
        sps.conformance_window = Some(ConformanceWindow {
            left: 0,
            right: 0,
            top: 0,
            bottom: 4, // 1080 is not CTU-aligned, need crop
        });
        let rbsp = write_sps(&sps);
        let parsed = params::parse_sps(&rbsp).unwrap();
        assert!(parsed.conformance_window.is_some());
        let cw = parsed.conformance_window.unwrap();
        assert_eq!(cw.bottom, 4);
    }

    #[test]
    fn full_parameter_set_nal_roundtrip() {
        use crate::encode::nal_write::assemble_annex_b;
        use crate::nal;
        use crate::types::NalUnitType;

        let vps = default_vps();
        let sps = default_sps(64, 64);
        let pps = default_pps(26);

        let vps_rbsp = write_vps(&vps);
        let sps_rbsp = write_sps(&sps);
        let pps_rbsp = write_pps(&pps);

        let stream = assemble_annex_b(&[
            (NalUnitType::VpsNut, vps_rbsp),
            (NalUnitType::SpsNut, sps_rbsp),
            (NalUnitType::PpsNut, pps_rbsp),
        ]);

        // Parse back with decoder infrastructure
        let mut found_vps = false;
        let mut found_sps = false;
        let mut found_pps = false;

        for nal_data in nal::NalIterator::new(&stream) {
            let nal_unit = nal::parse_nal_unit(nal_data).unwrap();
            match nal_unit.nal_type {
                NalUnitType::VpsNut => {
                    let parsed = params::parse_vps(&nal_unit.rbsp).unwrap();
                    assert_eq!(parsed.vps_id, 0);
                    found_vps = true;
                }
                NalUnitType::SpsNut => {
                    let parsed = params::parse_sps(&nal_unit.rbsp).unwrap();
                    assert_eq!(parsed.pic_width, 64);
                    assert_eq!(parsed.pic_height, 64);
                    found_sps = true;
                }
                NalUnitType::PpsNut => {
                    let parsed = params::parse_pps(&nal_unit.rbsp).unwrap();
                    assert_eq!(parsed.init_qp, 26);
                    found_pps = true;
                }
                _ => {}
            }
        }

        assert!(found_vps, "VPS not found in stream");
        assert!(found_sps, "SPS not found in stream");
        assert!(found_pps, "PPS not found in stream");
    }
}
