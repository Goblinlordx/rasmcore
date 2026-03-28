//! Unit tests for HEVC CABAC engine.

use rasmcore_hevc::cabac::tables::{RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};
use rasmcore_hevc::cabac::{CabacDecoder, ContextModel, SliceType, init_contexts};

// ============================================================================
// Context model tests
// ============================================================================

#[test]
fn context_init_midpoint_qp26() {
    // init_value=154 at QP=26 should give a neutral context.
    // m = (154 >> 4) * 5 - 45 = 9*5 - 45 = 0
    // n = ((154 & 15) << 3) - 16 = (10 << 3) - 16 = 64
    // preCtxState = clip(1,126, (0*26 >> 4) + 64) = 64
    // 64 > 63 → state = 64 - 64 = 0, mps = 1
    let ctx = ContextModel::new(154, 26);
    assert_eq!(ctx.state, 0);
    assert_eq!(ctx.mps, 1);
}

#[test]
fn context_init_low_qp0() {
    // init_value=107 at QP=0:
    // m = (107 >> 4) * 5 - 45 = 6*5 - 45 = -15
    // n = ((107 & 15) << 3) - 16 = (11 << 3) - 16 = 72
    // preCtxState = clip(1,126, (-15*0 >> 4) + 72) = 72
    // 72 > 63 → state = 72 - 64 = 8, mps = 1
    let ctx = ContextModel::new(107, 0);
    assert_eq!(ctx.state, 8);
    assert_eq!(ctx.mps, 1);
}

#[test]
fn context_init_high_qp51() {
    // init_value=107 at QP=51:
    // m = -15, n = 72
    // preCtxState = clip(1,126, (-15*51 >> 4) + 72) = clip(1,126, (-765 >> 4) + 72)
    //            = clip(1,126, -48 + 72) = 24  (arithmetic right shift)
    // 24 <= 63 → state = 63 - 24 = 39, mps = 0
    let ctx = ContextModel::new(107, 51);
    assert_eq!(ctx.state, 39);
    assert_eq!(ctx.mps, 0);
}

#[test]
fn context_init_clamps_prestate() {
    // init_value=0 at QP=51:
    // m = (0 >> 4) * 5 - 45 = -45
    // n = ((0 & 15) << 3) - 16 = -16
    // preCtxState = clip(1,126, (-45*51 >> 4) + (-16)) = clip(1,126, -143 + (-16))
    //            = clip(1,126, -159) = 1
    // 1 <= 63 → state = 63 - 1 = 62, mps = 0
    let ctx = ContextModel::new(0, 51);
    assert_eq!(ctx.state, 62);
    assert_eq!(ctx.mps, 0);
}

#[test]
fn context_init_upper_clamp() {
    // init_value=255 at QP=0:
    // m = (255 >> 4) * 5 - 45 = 15*5 - 45 = 30
    // n = ((255 & 15) << 3) - 16 = (15 << 3) - 16 = 104
    // preCtxState = clip(1,126, (30*0 >> 4) + 104) = 104
    // 104 > 63 → state = 104 - 64 = 40, mps = 1
    let ctx = ContextModel::new(255, 0);
    assert_eq!(ctx.state, 40);
    assert_eq!(ctx.mps, 1);
}

#[test]
fn context_mps_transition() {
    let mut ctx = ContextModel { state: 0, mps: 1 };
    // MPS at state 0 → state 1
    ctx.update_mps();
    assert_eq!(ctx.state, TRANS_IDX_MPS[0]);
    assert_eq!(ctx.state, 1);

    // MPS at state 1 → state 2
    ctx.update_mps();
    assert_eq!(ctx.state, 2);

    // MPS at state 62 → state 62 (saturates)
    ctx.state = 62;
    ctx.update_mps();
    assert_eq!(ctx.state, 62);
}

#[test]
fn context_lps_transition() {
    let mut ctx = ContextModel { state: 10, mps: 1 };
    // LPS at state 10 → state 8
    ctx.update_lps();
    assert_eq!(ctx.state, TRANS_IDX_LPS[10]);
    assert_eq!(ctx.state, 8);
    assert_eq!(ctx.mps, 1); // mps unchanged (state != 0)
}

#[test]
fn context_lps_at_state0_flips_mps() {
    let mut ctx = ContextModel { state: 0, mps: 1 };
    ctx.update_lps();
    // State 0 → state 0 (from table), and MPS flips
    assert_eq!(ctx.state, 0);
    assert_eq!(ctx.mps, 0);

    // Flip again
    ctx.update_lps();
    assert_eq!(ctx.mps, 1);
}

#[test]
fn context_state63_lps_stays() {
    // State 63 LPS → state 63 (from table: TRANS_IDX_LPS[63] = 63)
    let mut ctx = ContextModel { state: 63, mps: 0 };
    ctx.update_lps();
    assert_eq!(ctx.state, 63);
}

#[test]
fn init_contexts_batch() {
    let init_values = &[139, 141, 157]; // split_cu_flag I-slice
    let ctxs = init_contexts(init_values, 26);
    assert_eq!(ctxs.len(), 3);

    // Verify each matches individual init
    for (i, &iv) in init_values.iter().enumerate() {
        let single = ContextModel::new(iv, 26);
        assert_eq!(ctxs[i].state, single.state);
        assert_eq!(ctxs[i].mps, single.mps);
    }
}

// ============================================================================
// Table sanity checks
// ============================================================================

#[test]
fn range_tab_lps_dimensions() {
    assert_eq!(RANGE_TAB_LPS.len(), 64);
    for row in &RANGE_TAB_LPS {
        assert_eq!(row.len(), 4);
    }
}

#[test]
fn range_tab_lps_state0_monotonic() {
    // For state 0 (highest LPS probability), LPS range should increase with qRangeIdx
    let row = &RANGE_TAB_LPS[0];
    assert!(row[0] <= row[1]);
    assert!(row[1] <= row[2]);
    assert!(row[2] <= row[3]);
}

#[test]
fn range_tab_lps_column_decreasing_with_state() {
    // For each qRangeIdx, LPS range should decrease (or stay equal) as state increases
    for col in 0..4 {
        for state in 1..64 {
            assert!(
                RANGE_TAB_LPS[state][col] <= RANGE_TAB_LPS[state - 1][col],
                "LPS range increased at state={state} col={col}: {} > {}",
                RANGE_TAB_LPS[state][col],
                RANGE_TAB_LPS[state - 1][col]
            );
        }
    }
}

#[test]
fn trans_idx_lps_decreasing() {
    // LPS transition should always go to same or lower state
    for state in 0..64u8 {
        assert!(TRANS_IDX_LPS[state as usize] <= state);
    }
}

#[test]
fn trans_idx_mps_increasing() {
    // MPS transition should always go to same or higher state
    for state in 0..64u8 {
        assert!(TRANS_IDX_MPS[state as usize] >= state);
    }
}

#[test]
fn trans_idx_mps_saturates_at_62() {
    assert_eq!(TRANS_IDX_MPS[62], 62);
    assert_eq!(TRANS_IDX_MPS[63], 63);
}

// ============================================================================
// Arithmetic decoder tests
// ============================================================================

#[test]
fn decoder_init_all_zeros() {
    // All-zero data: offset initialized to 0
    let data = vec![0x00; 16];
    let dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 0);
    assert_eq!(dec.range(), 510);
}

#[test]
fn decoder_init_offset_value() {
    // First 9 bits are 0b100000000 = 256
    // Byte 0: 0b10000000 = 0x80 (first 8 bits: 10000000)
    // Byte 1: 0b0xxxxxxx (bit 8: 0)
    // 9 bits: 100000000 = 256
    let data = vec![0x80, 0x00, 0x00, 0x00];
    let dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 256);
    assert_eq!(dec.range(), 510);
}

#[test]
fn decoder_init_max_valid_offset() {
    // Max valid offset < range(510): offset = 509 = 0b111111101
    // bits: 1 1 1 1 1 1 1 0  1
    // byte 0: 0b11111110 = 0xFE
    // byte 1: bit 8 = 1 → 0b1xxxxxxx = 0x80
    let data = vec![0xFE, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 509);
}

#[test]
fn decoder_terminate_at_end() {
    // With offset very close to range, terminate should return 1.
    // offset=508, range=510: range-=2 → 508, offset(508) >= 508 → terminate=1
    // 508 = 0b111111100, bits: 11111110 0
    let data = vec![0xFE, 0x00, 0x00, 0x00];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 508);
    let term = dec.decode_terminate().unwrap();
    assert_eq!(term, 1);
}

#[test]
fn decoder_terminate_not_at_end() {
    // With offset=0, range=510: range-=2 → 508, offset(0) < 508 → not terminated
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 0);
    let term = dec.decode_terminate().unwrap();
    assert_eq!(term, 0);
    // After renormalization, range should be >= 256
    assert!(dec.range() >= 256);
}

#[test]
fn decoder_bypass_all_zeros() {
    // All-zero data → offset stays 0 → all bypass bins are 0
    let data = vec![0x00; 32];
    let mut dec = CabacDecoder::new(&data).unwrap();
    for _ in 0..50 {
        assert_eq!(dec.decode_bypass().unwrap(), 0);
    }
}

#[test]
fn decoder_bypass_all_ones() {
    // All 1-bits: offset gets large quickly → bypass bins should be 1
    // After init: offset = 0b111111111 = 511 ≥ range(510)... hmm, 511 > 510
    // That's technically an invalid state. Let's use 0xFF, 0x7F pattern:
    // 9 bits: 11111111 0 = 510... still >= 510? No: 0b111111110 = 510, which equals range.
    // Actually offset = 510 >= range(510) is invalid too.
    //
    // Let's use 0xFE 0xFF... : 9 bits = 11111110 1 = 0b111111101 = 509
    let mut data = vec![0xFE];
    data.extend(vec![0xFF; 31]);
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 509);

    // First bypass: offset = 509*2 + 1 = 1019, 1019 >= 510 → bin=1, offset=509
    assert_eq!(dec.decode_bypass().unwrap(), 1);
    // Second bypass: offset = 509*2 + 1 = 1019 >= 510 → bin=1, offset=509
    assert_eq!(dec.decode_bypass().unwrap(), 1);
    // Pattern continues: all-1 data keeps offset at 509 after each bypass
    for _ in 0..20 {
        assert_eq!(dec.decode_bypass().unwrap(), 1);
    }
}

#[test]
fn decoder_bypass_alternating() {
    // Carefully construct data so bypass alternates 1, 0, 1, 0, ...
    // This is harder to construct analytically, so we just test that
    // the decoder produces consistent results with known data.
    let data = vec![
        0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
        0xFF,
    ];
    let mut dec = CabacDecoder::new(&data).unwrap();
    // With offset=0, first bypass reads bit 0 → offset=0, result=0
    // Then reads more 0-bits → result stays 0 for a while
    // Once 1-bits start appearing, offset grows and results flip
    // Just verify no panics and results are 0 or 1
    for _ in 0..30 {
        let bin = dec.decode_bypass().unwrap();
        assert!(bin <= 1);
    }
}

#[test]
fn decoder_context_coded_with_known_state() {
    // Test context-coded decoding with a context at state 0 (highest LPS prob).
    // At state 0 qRangeIdx=3, LPS range = 240 (from table).
    // range=510, qRangeIdx = (510 >> 6) & 3 = 7 & 3 = 3
    // lps_range = RANGE_TAB_LPS[0][3] = 240
    // range -= 240 → range = 270
    // With offset=0: offset(0) < range(270) → MPS path
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    let mut ctx = ContextModel { state: 0, mps: 0 };

    let bin = dec.decode_bin(&mut ctx).unwrap();
    assert_eq!(bin, 0); // MPS=0, offset < range → decoded MPS
    assert_eq!(ctx.state, 1); // MPS transition: 0 → 1
    assert_eq!(ctx.mps, 0); // MPS unchanged
}

#[test]
fn decoder_context_coded_lps_path() {
    // Force LPS path: need offset >= range after LPS subtraction.
    // Setup: context at state 62 (very low LPS prob).
    // range=510, qRangeIdx=(510>>6)&3 = 3
    // lps_range = RANGE_TAB_LPS[62][3] = 10
    // range -= 10 → 500
    // Need offset >= 500. offset = 509 works (from 0xFE 0xFF...).
    let mut data = vec![0xFE];
    data.extend(vec![0xFF; 15]);
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 509);

    let mut ctx = ContextModel { state: 62, mps: 1 };

    let bin = dec.decode_bin(&mut ctx).unwrap();
    // offset(509) >= range(500) → LPS path
    // bin = 1 - mps(1) = 0
    assert_eq!(bin, 0);
    // LPS transition: state 62 → TRANS_IDX_LPS[62] = 38
    assert_eq!(ctx.state, 38);
    assert_eq!(ctx.mps, 1); // state != 0, no flip
}

#[test]
fn decoder_multiple_context_bins() {
    // Decode several bins with context adaptation
    let data = vec![0x00; 32];
    let mut dec = CabacDecoder::new(&data).unwrap();
    let mut ctx = ContextModel::new(153, 26);
    let initial_state = ctx.state;

    let mut bins = Vec::new();
    for _ in 0..10 {
        bins.push(dec.decode_bin(&mut ctx).unwrap());
    }

    // All bins should be 0 or 1
    for &b in &bins {
        assert!(b <= 1);
    }
    // Context state should have changed from repeated MPS hits
    // (with all-zero data and initial offset 0, we expect MPS path)
    assert!(ctx.state > initial_state || ctx.state == initial_state);
}

#[test]
fn decoder_truncated_input() {
    // Too short for 9-bit init
    let data = vec![0xFF];
    let result = CabacDecoder::new(&data);
    assert!(result.is_err());
}

// ============================================================================
// Binarization tests
// ============================================================================

#[test]
fn binarization_fl_bypass_zeros() {
    // All-zero data → FL(4) bypass = 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_fl_bypass(4).unwrap(), 0);
    assert_eq!(dec.decode_fl_bypass(8).unwrap(), 0);
}

#[test]
fn binarization_fl_bypass_ones() {
    // All-one data (offset=509) → FL bypass produces high values
    let mut data = vec![0xFE];
    data.extend(vec![0xFF; 15]);
    let mut dec = CabacDecoder::new(&data).unwrap();

    // With all-1 bypass bits, FL(4) should give 0b1111 = 15
    assert_eq!(dec.decode_fl_bypass(4).unwrap(), 15);
}

#[test]
fn binarization_tu_bypass_zero() {
    // All-zero bypass → TU immediately hits 0, returns 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_tu_bypass(10).unwrap(), 0);
}

#[test]
fn binarization_tu_bypass_max() {
    // All-one bypass → TU never hits 0, returns c_max
    let mut data = vec![0xFE];
    data.extend(vec![0xFF; 15]);
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_tu_bypass(5).unwrap(), 5);
}

#[test]
fn binarization_egk_zero() {
    // EGk(0) with first bypass bin = 0 → value = 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_egk(0).unwrap(), 0);
}

#[test]
fn binarization_egk_with_k3() {
    // EGk(3) with first bypass bin = 0 → prefix_len=0, suffix_len=3
    // suffix = 3 bypass bits, all zero → suffix=0
    // value = ((1<<0)-1)<<3 + 0 = 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_egk(3).unwrap(), 0);
}

#[test]
fn binarization_egk_with_prefix() {
    // All-one bypass → EGk(0):
    // prefix: 1,1,1,... until we count many 1-bits, then 0
    // But all-1 data means we never get 0 → prefix grows indefinitely
    // Let's use known data instead
    // For EGk(0), a value of 0 is: prefix 0-bit = 0 (first bypass = 0)
    // A value of 1 is: prefix 1, stop 0, suffix 1 bit
    // A value of 2 is: prefix 1, stop 0, suffix 1 bit (value = 1 + suffix)
    // A value of 3 is: prefix 1,1, stop 0, suffix 2 bits

    // Construct data so EGk(0) decodes value=1:
    // Init offset=255 from 9 bits: 011111111 → [0x7F, 0x80...]
    // First bypass: bit=0 from byte 1 → offset=255*2+0=510 >= 510 → bin=1, offset=0
    // Second bypass: bit=0 → offset=0 → bin=0 (prefix stops, len=1)
    // Suffix: 1 bit, bin=0 → suffix=0
    // value = ((1<<1)-1) + 0 = 1
    let data = vec![0x7F, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.offset(), 255);
    let val = dec.decode_egk(0).unwrap();
    assert_eq!(val, 1);
}

#[test]
fn binarization_tr_all_zero_bypass() {
    // TR with c_max=7, c_rice_param=0 (equivalent to TU(7))
    // All-zero bypass → first bin is 0 → value = 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_tr(7, 0, None).unwrap(), 0);
}

#[test]
fn binarization_tr_with_rice_param() {
    // TR with c_max=15, c_rice_param=2, bypass mode
    // All zero → prefix=0 (first bin 0), suffix=2 bits=00
    // value = (0 << 2) | 0 = 0
    let data = vec![0x00; 16];
    let mut dec = CabacDecoder::new(&data).unwrap();
    assert_eq!(dec.decode_tr(15, 2, None).unwrap(), 0);
}

#[test]
fn binarization_fl_ctx() {
    // Test FL with context-coded bins
    let data = vec![0x00; 32];
    let mut dec = CabacDecoder::new(&data).unwrap();
    let mut ctxs = vec![ContextModel::new(154, 26); 4];

    let val = dec.decode_fl_ctx(4, &mut ctxs).unwrap();
    assert!(val <= 15); // 4-bit value
}

#[test]
fn binarization_tu_ctx() {
    // Test TU with context-coded bins
    let data = vec![0x00; 32];
    let mut dec = CabacDecoder::new(&data).unwrap();
    let mut ctxs = vec![ContextModel::new(154, 26); 5];

    let val = dec.decode_tu_ctx(4, &mut ctxs).unwrap();
    assert!(val <= 4);
}

// ============================================================================
// SliceType tests
// ============================================================================

#[test]
fn slice_type_init_type() {
    assert_eq!(SliceType::I.init_type(false), 0);
    assert_eq!(SliceType::I.init_type(true), 0); // cabac_init_flag ignored for I

    assert_eq!(SliceType::P.init_type(false), 1);
    assert_eq!(SliceType::P.init_type(true), 2);

    assert_eq!(SliceType::B.init_type(false), 2);
    assert_eq!(SliceType::B.init_type(true), 1);
}

// ============================================================================
// Integration: init → decode sequence
// ============================================================================

#[test]
fn integration_i_slice_init_and_decode() {
    // Simulate: initialize contexts for I-slice at QP 26, decode several bins
    use rasmcore_hevc::cabac::{CBF_LUMA_INIT, SPLIT_CU_FLAG_INIT};

    let qp = 26;
    let init_type = SliceType::I.init_type(false);

    let split_cu_ctxs = init_contexts(&SPLIT_CU_FLAG_INIT[init_type], qp);
    assert_eq!(split_cu_ctxs.len(), 3);

    let cbf_luma_ctxs = init_contexts(&CBF_LUMA_INIT[init_type], qp);
    assert_eq!(cbf_luma_ctxs.len(), 2);

    // Decode with a known bitstream
    let data = vec![0x40; 32]; // 0100_0000 pattern
    let mut dec = CabacDecoder::new(&data).unwrap();
    let mut ctxs = split_cu_ctxs;

    // Decode 5 bins using split_cu_flag context 0
    for _ in 0..5 {
        let bin = dec.decode_bin(&mut ctxs[0]).unwrap();
        assert!(bin <= 1);
    }

    // Context should have adapted (state changed)
    // We don't assert exact values since they depend on decode path,
    // but state should be different from initial
    let fresh = ContextModel::new(SPLIT_CU_FLAG_INIT[init_type][0], qp);
    // After 5 bins, state likely advanced via MPS transitions
    assert!(ctxs[0].state != fresh.state || ctxs[0].mps != fresh.mps);
}
