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
fn yuv_diag_checker_256x256_q22() {
    if !fixtures_available() {
        return;
    }
    let frame = decode_test_case("checker_256x256_q22").unwrap();
    let ref_yuv = load_reference_yuv("checker_256x256_q22").unwrap();
    let width = frame.width as usize;
    let height = frame.height as usize;
    let ref_y = &ref_yuv[..width * height];
    let our_y = &frame.y_plane;

    // Per CTU row analysis (CTU = 64 for 256x256)
    let ctu_size = 64usize;
    let ctu_rows = height / ctu_size;
    let ctu_cols = width / ctu_size;
    for ctu_row in 0..ctu_rows {
        let start = ctu_row * ctu_size * width;
        let end = start + ctu_size * width;
        let mut diffs = 0u32;
        let mut max_diff = 0i32;
        let mut sum_sq = 0f64;
        for i in start..end {
            let d = our_y[i] as i32 - ref_y[i] as i32;
            if d != 0 {
                diffs += 1;
            }
            max_diff = max_diff.max(d.abs());
            sum_sq += (d as f64) * (d as f64);
        }
        let mse = sum_sq / (ctu_size * width) as f64;
        let psnr = if mse < 0.001 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };
        eprintln!(
            "CTU row {}: diffs={}/{}, max_diff={}, PSNR={:.1}dB",
            ctu_row,
            diffs,
            ctu_size * width,
            max_diff,
            psnr
        );
    }

    // Per CTU analysis (within each row)
    for ctu_row in 0..ctu_rows {
        for ctu_col in 0..ctu_cols {
            let mut diffs = 0u32;
            let mut max_diff = 0i32;
            for row in 0..ctu_size {
                for col in 0..ctu_size {
                    let y = ctu_row * ctu_size + row;
                    let x = ctu_col * ctu_size + col;
                    let i = y * width + x;
                    let d = (our_y[i] as i32 - ref_y[i] as i32).abs();
                    if d != 0 {
                        diffs += 1;
                    }
                    max_diff = max_diff.max(d);
                }
            }
            if diffs > 0 {
                eprintln!(
                    "  CTU({},{}): diffs={}/{}, max_diff={}",
                    ctu_col,
                    ctu_row,
                    diffs,
                    ctu_size * ctu_size,
                    max_diff
                );
            }
        }
    }

    // Per 8x8 block analysis within CTU(0,0) to pinpoint where errors start
    eprintln!("\nCTU(0,0) per-8x8 block diffs:");
    for by in 0..4 {
        for bx in 0..4 {
            let mut diffs = 0u32;
            for r in 0..8 {
                for c in 0..8 {
                    let y = by * 8 + r;
                    let x = bx * 8 + c;
                    let i = y * width + x;
                    if our_y[i] != ref_y[i] {
                        diffs += 1;
                    }
                }
            }
            if diffs > 0 {
                let i0 = (by * 8) * width + bx * 8;
                eprint!(
                    "  block({},{}):{}/64 ours={} ref={} ",
                    bx, by, diffs, our_y[i0], ref_y[i0]
                );
            }
        }
    }
    eprintln!();

    // Show specific row 0 values for first 64 pixels
    eprintln!("\nRow 0, first 64 pixels:");
    eprint!("  REF: ");
    for x in 0..64 { eprint!("{:3} ", ref_y[x]); }
    eprintln!();
    eprint!("  OUR: ");
    for x in 0..64 { eprint!("{:3} ", our_y[x]); }
    eprintln!();
    eprint!("  DIF: ");
    for x in 0..64 { eprint!("{:3} ", our_y[x] as i16 - ref_y[x] as i16); }
    eprintln!();

    // Show first 20 differing pixels
    let first_diffs: Vec<_> = our_y
        .iter()
        .zip(ref_y.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(20)
        .map(|(i, (&a, &b))| {
            let x = i % width;
            let y = i / width;
            (x, y, a, b, a as i16 - b as i16)
        })
        .collect();
    if !first_diffs.is_empty() {
        eprintln!("First diffs (x, y, ours, ref, delta): {:?}", first_diffs);
    }
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
