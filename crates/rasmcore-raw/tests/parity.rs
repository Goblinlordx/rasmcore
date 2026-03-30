//! DNG decode parity and validation tests.
//!
//! Since dcraw/rawpy are not available in CI, we use synthetic DNG files
//! built programmatically with known sensor data, CFA patterns, and color
//! matrices. We validate the full pipeline against mathematically expected
//! results.

/// Build a minimal uncompressed DNG file in memory.
/// Returns the bytes of a valid DNG with the specified parameters.
fn build_synthetic_dng(
    width: u32,
    height: u32,
    bps: u16,
    cfa: [u8; 4],
    color_matrix: [f64; 9],
    as_shot_neutral: [f64; 3],
    raw_pixels: &[u16],
) -> Vec<u8> {
    // Build a minimal little-endian TIFF/DNG file
    let mut data = Vec::new();

    // Header: II, 42, IFD offset
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    // IFD offset — we'll fill this after building tag values
    let ifd_offset_pos = data.len();
    data.extend_from_slice(&0u32.to_le_bytes()); // placeholder

    // Prepare tag values that don't fit inline (>4 bytes)
    // We'll put them after the IFD entries.

    // Build raw pixel data first, so we know the offset
    let mut raw_bytes = Vec::new();
    match bps {
        8 => {
            for &v in raw_pixels {
                raw_bytes.push(v as u8);
            }
        }
        16 => {
            for &v in raw_pixels {
                raw_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        _ => {
            for &v in raw_pixels {
                raw_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    // We'll place:
    // 1. Raw data at offset 8 (right after header)
    // 2. Extended tag values after raw data
    // 3. IFD after extended tag values

    let raw_data_offset = 8u32;
    data.resize(8 + raw_bytes.len(), 0);
    data[8..8 + raw_bytes.len()].copy_from_slice(&raw_bytes);

    let mut ext_values_start = data.len();

    // ColorMatrix1: SRATIONAL × 9 = 72 bytes
    let cm_offset = ext_values_start as u32;
    for &v in &color_matrix {
        // Convert f64 to SRATIONAL (numerator/denominator as i32)
        let (num, den) = float_to_srational(v);
        data.extend_from_slice(&num.to_le_bytes());
        data.extend_from_slice(&den.to_le_bytes());
    }
    ext_values_start = data.len();

    // AsShotNeutral: RATIONAL × 3 = 24 bytes
    let asn_offset = ext_values_start as u32;
    for &v in &as_shot_neutral {
        let (num, den) = float_to_rational(v);
        data.extend_from_slice(&num.to_le_bytes());
        data.extend_from_slice(&den.to_le_bytes());
    }

    // StripOffsets: LONG × 1 — fits inline
    // StripByteCounts: LONG × 1 — fits inline

    // Now build the IFD
    let ifd_offset = data.len() as u32;
    // Write the IFD offset back in the header
    data[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

    // Tags we need:
    // 254 NewSubFileType = 0
    // 256 ImageWidth
    // 257 ImageLength
    // 258 BitsPerSample
    // 259 Compression = 1 (none)
    // 262 PhotometricInterpretation = 32803 (CFA)
    // 273 StripOffsets
    // 277 SamplesPerPixel = 1
    // 278 RowsPerStrip = height
    // 279 StripByteCounts
    // 33421 CFARepeatPatternDim = [2, 2]
    // 33422 CFAPattern = [r, g, g, b]
    // 50706 DNGVersion = [1, 4, 0, 0]
    // 50721 ColorMatrix1
    // 50727 AsShotNeutral

    let tags: Vec<(u16, u16, u32, u32)> = vec![
        // (tag, type, count, value_or_offset)
        (254, 4, 1, 0),                      // NewSubFileType = 0
        (256, 4, 1, width),                  // ImageWidth
        (257, 4, 1, height),                 // ImageLength
        (258, 3, 1, bps as u32),             // BitsPerSample
        (259, 3, 1, 1),                      // Compression = none
        (262, 3, 1, 32803),                  // PhotometricInterpretation = CFA
        (273, 4, 1, raw_data_offset),        // StripOffsets
        (277, 3, 1, 1),                      // SamplesPerPixel = 1
        (278, 4, 1, height),                 // RowsPerStrip = height
        (279, 4, 1, raw_bytes.len() as u32), // StripByteCounts
    ];

    // CFARepeatPatternDim (33421): SHORT×2 — fits in 4 bytes inline
    let cfa_dim_val = 2u16.to_le_bytes();
    let cfa_dim_inline = u32::from_le_bytes([
        cfa_dim_val[0],
        cfa_dim_val[1],
        cfa_dim_val[0],
        cfa_dim_val[1],
    ]);

    // CFAPattern (33422): BYTE×4 — fits in 4 bytes inline
    let cfa_pattern_inline = u32::from_le_bytes(cfa);

    // DNGVersion (50706): BYTE×4 — fits in 4 bytes inline
    let dng_version_inline = u32::from_le_bytes([1, 4, 0, 0]);

    let mut all_tags: Vec<(u16, u16, u32, u32)> = tags;
    all_tags.push((33421, 3, 2, cfa_dim_inline)); // CFARepeatPatternDim
    all_tags.push((33422, 1, 4, cfa_pattern_inline)); // CFAPattern
    all_tags.push((50706, 1, 4, dng_version_inline)); // DNGVersion
    all_tags.push((50721, 10, 9, cm_offset)); // ColorMatrix1 (SRATIONAL)
    all_tags.push((50727, 5, 3, asn_offset)); // AsShotNeutral (RATIONAL)

    // Sort tags by tag number (TIFF requirement)
    all_tags.sort_by_key(|t| t.0);

    // Write IFD
    let num_entries = all_tags.len() as u16;
    data.extend_from_slice(&num_entries.to_le_bytes());
    for (tag, typ, count, val) in &all_tags {
        data.extend_from_slice(&tag.to_le_bytes());
        data.extend_from_slice(&typ.to_le_bytes());
        data.extend_from_slice(&count.to_le_bytes());
        data.extend_from_slice(&val.to_le_bytes());
    }
    // Next IFD offset = 0 (no more IFDs)
    data.extend_from_slice(&0u32.to_le_bytes());

    data
}

fn float_to_srational(v: f64) -> (i32, i32) {
    let den = 10000i32;
    let num = (v * den as f64).round() as i32;
    (num, den)
}

fn float_to_rational(v: f64) -> (u32, u32) {
    let den = 10000u32;
    let num = (v * den as f64).round() as u32;
    (num, den)
}

#[test]
fn detect_synthetic_dng() {
    // A synthetic DNG should be detected as DNG, not TIFF
    let raw = vec![1000u16; 16]; // 4×4
    let dng = build_synthetic_dng(
        4,
        4,
        16,
        [0, 1, 1, 2],                                  // RGGB
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // identity
        [1.0, 1.0, 1.0],
        &raw,
    );

    assert!(rasmcore_raw::is_dng(&dng), "should detect as DNG");
}

#[test]
fn detect_non_dng_tiff() {
    // A plain TIFF (without DNGVersion) should not be detected as DNG
    let mut data = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    // IFD with 1 entry: ImageWidth = 100
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&256u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&100u16.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    assert!(!rasmcore_raw::is_dng(&data), "plain TIFF should not be DNG");
}

#[test]
fn decode_uniform_grey_dng() {
    // All sensor pixels at the same value → output should be uniform grey
    let w = 8u32;
    let h = 8u32;
    let val = 2048u16;
    let raw = vec![val; (w * h) as usize];
    let dng = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],                                  // RGGB
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // identity matrix
        [1.0, 1.0, 1.0],                               // neutral WB
        &raw,
    );

    let result = rasmcore_raw::decode(&dng).expect("decode should succeed");
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    assert_eq!(result.pixels.len(), (w * h * 3) as usize);

    // With identity matrix, neutral WB, and uniform input at ~50% of white level,
    // all pixels should have roughly the same R=G=B value.
    // The exact value depends on the sRGB gamma curve:
    // normalized = 2048/4095 ≈ 0.5, sRGB(0.5) ≈ 0.735, output ≈ 187
    // But with identity ColorMatrix (camera=XYZ) and XYZ_to_sRGB, the matrix
    // rows sum to different values. Still, for a grey input R=G=B in camera space,
    // the output should still be a shade of grey since the matrix preserves neutrals.

    // Check interior pixels (avoid edges where demosaic has boundary effects)
    for row in 2..6 {
        for col in 2..6 {
            let idx = ((row * w + col) * 3) as usize;
            let r = result.pixels[idx];
            let g = result.pixels[idx + 1];
            let b = result.pixels[idx + 2];
            // R, G, B should be similar (within tolerance for matrix effects)
            let max_ch = r.max(g).max(b);
            let min_ch = r.min(g).min(b);
            assert!(
                max_ch - min_ch < 30,
                "pixel ({row},{col}) not grey: R={r} G={g} B={b}"
            );
        }
    }
}

#[test]
fn decode_saturated_red_dng() {
    // Set red CFA pixels to max, green and blue to zero
    // With RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
    let w = 8u32;
    let h = 8u32;
    let mut raw = vec![0u16; (w * h) as usize];

    // Set red pixels to max value
    for row in 0..h {
        for col in 0..w {
            if row % 2 == 0 && col % 2 == 0 {
                raw[(row * w + col) as usize] = 4095;
            }
        }
    }

    let dng = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],                                  // RGGB
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // identity
        [1.0, 1.0, 1.0],                               // neutral WB
        &raw,
    );

    let result = rasmcore_raw::decode(&dng).expect("decode should succeed");

    // At interior red pixel positions, R should be significantly higher than G and B
    // (0,0), (0,2), (2,0), etc. are red positions
    // Check pixel at (2,2) — a red pixel in the interior
    let idx = ((2 * w + 2) * 3) as usize;
    let r = result.pixels[idx];
    let g = result.pixels[idx + 1];
    let b = result.pixels[idx + 2];
    assert!(
        r > g && r > b,
        "red pixel should dominate: R={r} G={g} B={b}"
    );
}

#[test]
fn decode_8bit_dng() {
    // Test 8-bit sensor data
    let w = 8u32;
    let h = 8u32;
    let raw = vec![128u16; (w * h) as usize]; // mid-grey in 8-bit

    let dng = build_synthetic_dng(
        w,
        h,
        8,
        [0, 1, 1, 2],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        &raw,
    );

    let result = rasmcore_raw::decode(&dng).expect("decode 8-bit should succeed");
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    assert_eq!(result.bits_per_sample, 8);
}

#[test]
fn all_cfa_patterns_decode() {
    // All 4 CFA patterns should decode without error
    let patterns: [(&str, [u8; 4]); 4] = [
        ("RGGB", [0, 1, 1, 2]),
        ("BGGR", [2, 1, 1, 0]),
        ("GRBG", [1, 0, 2, 1]),
        ("GBRG", [1, 2, 0, 1]),
    ];

    let w = 8u32;
    let h = 8u32;
    let raw = vec![2000u16; (w * h) as usize];

    for (name, cfa) in &patterns {
        let dng = build_synthetic_dng(
            w,
            h,
            16,
            *cfa,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            &raw,
        );

        let result = rasmcore_raw::decode(&dng);
        assert!(
            result.is_ok(),
            "CFA pattern {name} failed: {:?}",
            result.err()
        );
    }
}

#[test]
fn decode_16bit_output() {
    let w = 4u32;
    let h = 4u32;
    let raw = vec![2048u16; (w * h) as usize];

    let dng = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        &raw,
    );

    let result = rasmcore_raw::decode_16bit(&dng).expect("16-bit decode should succeed");
    assert!(result.is_16bit);
    // 16-bit output: 6 bytes per pixel (RGB16 LE)
    assert_eq!(result.pixels.len(), (w * h * 6) as usize);
}

#[test]
fn color_matrix_affects_output() {
    // With different color matrices, the same input should produce different output
    let w = 8u32;
    let h = 8u32;
    let raw = vec![2048u16; (w * h) as usize];

    let dng_identity = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        &raw,
    );

    // A different color matrix (scaled)
    let dng_scaled = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],
        [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
        [1.0, 1.0, 1.0],
        &raw,
    );

    let r1 = rasmcore_raw::decode(&dng_identity).unwrap();
    let r2 = rasmcore_raw::decode(&dng_scaled).unwrap();

    // The outputs should differ (different color matrices)
    assert_ne!(
        r1.pixels, r2.pixels,
        "different color matrices should produce different output"
    );
}

#[test]
fn white_balance_affects_output() {
    let w = 8u32;
    let h = 8u32;
    let raw = vec![2048u16; (w * h) as usize];

    let dng_neutral = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], // neutral
        &raw,
    );

    let dng_warm = build_synthetic_dng(
        w,
        h,
        16,
        [0, 1, 1, 2],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.5, 1.0, 0.8], // warm (boost red, reduce blue)
        &raw,
    );

    let r1 = rasmcore_raw::decode(&dng_neutral).unwrap();
    let r2 = rasmcore_raw::decode(&dng_warm).unwrap();

    assert_ne!(
        r1.pixels, r2.pixels,
        "different WB should produce different output"
    );
}

#[test]
fn backward_compat_tiff_not_affected() {
    // A regular TIFF file should still not be detected as DNG
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes());
    // Minimal IFD
    tiff.extend_from_slice(&0u16.to_le_bytes()); // 0 entries
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

    assert!(!rasmcore_raw::is_dng(&tiff));
    // Attempting to decode as DNG should fail
    assert!(rasmcore_raw::decode(&tiff).is_err());
}
