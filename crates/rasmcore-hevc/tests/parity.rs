//! HEVC parity tests — compare our decoder output against ffmpeg/libde265 reference.
//!
//! These tests decode the same HEVC bitstreams that were encoded by x265 and
//! decoded by ffmpeg (the golden reference), then compare pixel output.
//!
//! Tests skip gracefully if fixtures are not generated.
//! Run `tests/fixtures/hevc/generate.sh` to generate fixtures.
//!
//! ## Test categories:
//! - **Metadata tests**: verify SPS parsing, NAL structure, fixture sizes
//! - **Pixel parity tests**: full decode + pixel comparison against ffmpeg reference

use rasmcore_hevc::testutil::{
    TEST_CASES, compare_pixels, fixtures_available, load_fixture, load_reference_rgb,
    load_reference_yuv,
};

/// Helper to decode a test case and return the result.
fn decode_test_case(case: &str) -> Result<rasmcore_hevc::DecodedFrame, rasmcore_hevc::HevcError> {
    let hevc_data = load_fixture(case, "hevc").expect("fixture not found");
    rasmcore_hevc::decode(&hevc_data, &[])
}

// ─── Pixel Parity Tests ────────────────────────────────────────────────────

#[test]

fn parity_flat_64x64_q22_pixels() {
    if !fixtures_available() {
        return;
    }

    let frame =
        decode_test_case("flat_64x64_q22").expect("flat_64x64_q22 should decode successfully");

    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.pixels.len(), 64 * 64 * 3);

    let ref_rgb = load_reference_rgb("flat_64x64_q22").unwrap();
    let cmp = compare_pixels(&frame.pixels, &ref_rgb, 0);

    eprintln!(
        "flat_64x64_q22: PSNR={:.1}dB, max_diff={}, exact={}",
        cmp.psnr,
        cmp.max_diff,
        cmp.is_exact()
    );

    assert!(
        cmp.passes_psnr(30.0),
        "PSNR too low: {:.1}dB (expected > 30dB)",
        cmp.psnr
    );
}

#[test]

fn parity_flat_64x64_q37_pixels() {
    if !fixtures_available() {
        return;
    }

    let frame =
        decode_test_case("flat_64x64_q37").expect("flat_64x64_q37 should decode successfully");

    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);

    let ref_rgb = load_reference_rgb("flat_64x64_q37").unwrap();
    let cmp = compare_pixels(&frame.pixels, &ref_rgb, 0);
    eprintln!(
        "flat_64x64_q37: PSNR={:.1}dB, max_diff={}, exact={}",
        cmp.psnr,
        cmp.max_diff,
        cmp.is_exact()
    );
    assert!(cmp.passes_psnr(30.0));
}

#[test]

fn parity_gradient_128x128_q22_pixels() {
    if !fixtures_available() {
        return;
    }

    let frame = decode_test_case("gradient_128x128_q22")
        .expect("gradient_128x128_q22 should decode successfully");

    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);

    let ref_rgb = load_reference_rgb("gradient_128x128_q22").unwrap();
    let cmp = compare_pixels(&frame.pixels, &ref_rgb, 0);
    eprintln!(
        "gradient_128x128_q22: PSNR={:.1}dB, max_diff={}, mismatches={}/{}",
        cmp.psnr, cmp.max_diff, cmp.mismatches, cmp.total_pixels
    );
    assert!(cmp.passes_psnr(25.0));
}

#[test]

fn parity_checker_256x256_q22_pixels() {
    if !fixtures_available() {
        return;
    }

    let frame = decode_test_case("checker_256x256_q22")
        .expect("checker_256x256_q22 should decode successfully");

    assert_eq!(frame.width, 256);
    assert_eq!(frame.height, 256);

    let ref_rgb = load_reference_rgb("checker_256x256_q22").unwrap();
    let cmp = compare_pixels(&frame.pixels, &ref_rgb, 0);
    eprintln!(
        "checker_256x256_q22: PSNR={:.1}dB, max_diff={}, mismatches={}/{}",
        cmp.psnr, cmp.max_diff, cmp.mismatches, cmp.total_pixels
    );
    assert!(cmp.passes_psnr(25.0));
}

// ─── YUV Parity Tests — Byte-exact Y plane comparison ─────────────────────

#[test]
fn yuv_parity_flat_64x64_q22() {
    if !fixtures_available() {
        return;
    }
    let frame = decode_test_case("flat_64x64_q22").unwrap();
    let ref_yuv = load_reference_yuv("flat_64x64_q22").unwrap();
    let y_size = (frame.width * frame.height) as usize;
    let ref_y = &ref_yuv[..y_size];

    let diffs: Vec<_> = frame
        .y_plane
        .iter()
        .zip(ref_y.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(20)
        .map(|(i, (&a, &b))| {
            let x = i % frame.width as usize;
            let y = i / frame.width as usize;
            (x, y, a as i16 - b as i16)
        })
        .collect();

    let total_diffs = frame
        .y_plane
        .iter()
        .zip(ref_y.iter())
        .filter(|(a, b)| a != b)
        .count();

    eprintln!(
        "flat_64x64_q22 Y parity: {}/{} bytes differ",
        total_diffs, y_size
    );
    if !diffs.is_empty() {
        eprintln!("  First diffs: {:?}", diffs);
    }
    assert_eq!(
        total_diffs, 0,
        "Y plane not byte-exact: {} bytes differ",
        total_diffs
    );
}

#[test]
fn yuv_parity_flat_64x64_q37() {
    if !fixtures_available() {
        return;
    }
    let frame = decode_test_case("flat_64x64_q37").unwrap();
    let ref_yuv = load_reference_yuv("flat_64x64_q37").unwrap();
    let y_size = (frame.width * frame.height) as usize;
    let ref_y = &ref_yuv[..y_size];

    let total_diffs = frame
        .y_plane
        .iter()
        .zip(ref_y.iter())
        .filter(|(a, b)| a != b)
        .count();

    eprintln!(
        "flat_64x64_q37 Y parity: {}/{} bytes differ",
        total_diffs, y_size
    );
    assert_eq!(total_diffs, 0, "Y plane not byte-exact");
}

#[test]
fn yuv_parity_gradient_128x128_q22() {
    if !fixtures_available() {
        return;
    }
    let frame = decode_test_case("gradient_128x128_q22").unwrap();
    let ref_yuv = load_reference_yuv("gradient_128x128_q22").unwrap();
    let y_size = (frame.width * frame.height) as usize;
    let ref_y = &ref_yuv[..y_size];

    let diffs: Vec<_> = frame
        .y_plane
        .iter()
        .zip(ref_y.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(20)
        .map(|(i, (&a, &b))| {
            let x = i % frame.width as usize;
            let y = i / frame.width as usize;
            (x, y, a, b, a as i16 - b as i16)
        })
        .collect();

    let total_diffs = frame
        .y_plane
        .iter()
        .zip(ref_y.iter())
        .filter(|(a, b)| a != b)
        .count();

    eprintln!(
        "gradient_128x128_q22 Y parity: {}/{} bytes differ",
        total_diffs, y_size
    );
    if !diffs.is_empty() {
        eprintln!("  First diffs (x, y, ours, ref, delta): {:?}", diffs);
    }
    assert_eq!(
        total_diffs, 0,
        "Y plane not byte-exact: {} bytes differ",
        total_diffs
    );
}

#[test]

fn parity_all_cases_decode() {
    if !fixtures_available() {
        return;
    }

    for &(case, expected_w, expected_h) in TEST_CASES {
        let frame = decode_test_case(case).unwrap_or_else(|e| panic!("{case}: decode failed: {e}"));

        assert_eq!(frame.width, expected_w, "{case}: width mismatch");
        assert_eq!(frame.height, expected_h, "{case}: height mismatch");
        assert_eq!(
            frame.pixels.len(),
            (expected_w * expected_h * 3) as usize,
            "{case}: pixel buffer size mismatch"
        );
        eprintln!(
            "{case}: OK ({expected_w}x{expected_h}, {} bytes)",
            frame.pixels.len()
        );
    }
}

// ─── Metadata Tests (always run — no CABAC decode needed) ──────────────────

#[test]
fn parity_metadata_sps_matches_reference() {
    if !fixtures_available() {
        eprintln!("SKIPPED: fixtures not generated");
        return;
    }

    for &(case, expected_w, expected_h) in TEST_CASES {
        let hevc_data = load_fixture(case, "hevc").unwrap();

        let nals: Vec<&[u8]> = rasmcore_hevc::NalIterator::new(&hevc_data).collect();
        assert!(
            nals.len() >= 3,
            "{case}: expected >= 3 NALs (VPS+SPS+PPS), got {}",
            nals.len()
        );

        let mut found_sps = false;
        for nal_data in &nals {
            let nal = rasmcore_hevc::parse_nal_unit(nal_data).unwrap();
            if nal.nal_type == rasmcore_hevc::NalUnitType::SpsNut {
                let sps = rasmcore_hevc::params::parse_sps(&nal.rbsp).unwrap();
                assert_eq!(sps.pic_width, expected_w, "{case}: SPS width mismatch");
                assert_eq!(sps.pic_height, expected_h, "{case}: SPS height mismatch");
                assert_eq!(sps.chroma_format_idc, 1, "{case}: should be 4:2:0");
                assert_eq!(sps.bit_depth_luma, 8, "{case}: should be 8-bit");
                eprintln!(
                    "{case}: SPS OK ({}x{}, 4:2:0, 8-bit, CTU={})",
                    sps.pic_width,
                    sps.pic_height,
                    sps.ctu_size()
                );
                found_sps = true;
                break;
            }
        }
        assert!(found_sps, "{case}: SPS NAL not found");
    }
}

#[test]
fn parity_yuv_reference_sizes() {
    if !fixtures_available() {
        eprintln!("SKIPPED: fixtures not generated");
        return;
    }

    for &(case, w, h) in TEST_CASES {
        let yuv = load_reference_yuv(case);
        assert!(yuv.is_some(), "{case}: YUV reference file missing");
        let yuv_data = yuv.unwrap();
        let expected_size = (w * h) as usize + 2 * ((w / 2) * (h / 2)) as usize;
        assert_eq!(
            yuv_data.len(),
            expected_size,
            "{case}: YUV size mismatch (expected {expected_size}, got {})",
            yuv_data.len()
        );
    }
}

#[test]
fn parity_rgb_reference_sizes() {
    if !fixtures_available() {
        eprintln!("SKIPPED: fixtures not generated");
        return;
    }

    for &(case, w, h) in TEST_CASES {
        let rgb = load_reference_rgb(case);
        assert!(rgb.is_some(), "{case}: RGB reference file missing");
        let rgb_data = rgb.unwrap();
        let expected_size = (w * h * 3) as usize;
        assert_eq!(rgb_data.len(), expected_size, "{case}: RGB size mismatch");
    }
}

#[test]
fn parity_nal_structure_valid() {
    if !fixtures_available() {
        eprintln!("SKIPPED: fixtures not generated");
        return;
    }

    for &(case, _, _) in TEST_CASES {
        let hevc_data = load_fixture(case, "hevc").unwrap();
        let nals: Vec<&[u8]> = rasmcore_hevc::NalIterator::new(&hevc_data).collect();

        // Every bitstream should have VPS + SPS + PPS + at least one VCL NAL
        let has_vps = nals.iter().any(|n| {
            rasmcore_hevc::parse_nal_unit(n)
                .map(|u| u.nal_type == rasmcore_hevc::NalUnitType::VpsNut)
                .unwrap_or(false)
        });
        let has_sps = nals.iter().any(|n| {
            rasmcore_hevc::parse_nal_unit(n)
                .map(|u| u.nal_type == rasmcore_hevc::NalUnitType::SpsNut)
                .unwrap_or(false)
        });
        let has_pps = nals.iter().any(|n| {
            rasmcore_hevc::parse_nal_unit(n)
                .map(|u| u.nal_type == rasmcore_hevc::NalUnitType::PpsNut)
                .unwrap_or(false)
        });
        let has_vcl = nals.iter().any(|n| {
            rasmcore_hevc::parse_nal_unit(n)
                .map(|u| u.nal_type.is_vcl())
                .unwrap_or(false)
        });

        assert!(has_vps, "{case}: missing VPS");
        assert!(has_sps, "{case}: missing SPS");
        assert!(has_pps, "{case}: missing PPS");
        assert!(has_vcl, "{case}: missing VCL NAL");
        eprintln!("{case}: NAL structure OK ({} NALs)", nals.len());
    }
}
