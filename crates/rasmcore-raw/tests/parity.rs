//! DNG decode parity and validation tests.
//!
//! Two test categories:
//! 1. Synthetic tests — programmatically built DNG files with known values,
//!    validated against mathematically expected results.
//! 2. Reference parity tests — our decode vs dcraw and rawpy on the same
//!    real DNG input, with Mean Absolute Error (MAE) threshold.

use std::path::Path;

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

// ─── Reference Parity Tests ─────────────────────────────────────────────────
//
// Compare our DNG decode against dcraw and rawpy on the same input.
// The DNG fixture is generated by tests/fixtures/scripts/generate_dng_fixture.py
// and reference outputs by dcraw and rawpy are pre-generated alongside it.

/// Locate the project root (parent of crates/).
fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

/// Compute Mean Absolute Error between two RGB8 buffers.
fn mae_rgb8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffer lengths must match");
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// Read a TIFF file and extract raw RGB8 pixel data.
/// Uses a minimal TIFF reader — only handles the uncompressed RGB8/RGB16
/// output that dcraw produces.
fn read_tiff_rgb8(path: &Path) -> (Vec<u8>, u32, u32) {
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    // Parse TIFF header
    let order = match &data[0..2] {
        b"II" => "le",
        b"MM" => "be",
        _ => panic!("not a TIFF"),
    };

    let read_u16 = |off: usize| -> u16 {
        if order == "le" {
            u16::from_le_bytes([data[off], data[off + 1]])
        } else {
            u16::from_be_bytes([data[off], data[off + 1]])
        }
    };
    let read_u32 = |off: usize| -> u32 {
        if order == "le" {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
    };

    let ifd_offset = read_u32(4) as usize;
    let num_entries = read_u16(ifd_offset) as usize;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut bps = 8u16;
    let mut strip_offset = 0u32;
    let mut strip_bytes = 0u32;
    let mut spp = 3u16;

    for i in 0..num_entries {
        let base = ifd_offset + 2 + i * 12;
        let tag = read_u16(base);
        let typ = read_u16(base + 2);
        let count = read_u32(base + 4);
        match tag {
            256 => {
                width = if typ == 3 {
                    read_u16(base + 8) as u32
                } else {
                    read_u32(base + 8)
                }
            }
            257 => {
                height = if typ == 3 {
                    read_u16(base + 8) as u32
                } else {
                    read_u32(base + 8)
                }
            }
            258 => {
                // BitsPerSample: may be count=1 (inline) or count=3 (offset to 3 shorts)
                if count == 1 {
                    bps = read_u16(base + 8);
                } else {
                    // count > 1: value field is offset to array of shorts
                    let off = read_u32(base + 8) as usize;
                    bps = read_u16(off); // read first channel's BPS
                }
            }
            273 => {
                strip_offset = if typ == 3 {
                    read_u16(base + 8) as u32
                } else {
                    read_u32(base + 8)
                }
            }
            277 => spp = read_u16(base + 8),
            279 => {
                strip_bytes = if typ == 3 {
                    read_u16(base + 8) as u32
                } else {
                    read_u32(base + 8)
                }
            }
            _ => {}
        }
    }

    let pixel_data = &data[strip_offset as usize..(strip_offset + strip_bytes) as usize];
    let pixel_count = (width * height) as usize;

    if bps == 16 && spp == 3 {
        // Convert 16-bit to 8-bit (dcraw -4 output)
        let mut rgb8 = Vec::with_capacity(pixel_count * 3);
        for i in 0..pixel_count * 3 {
            let off = i * 2;
            let val16 = read_u16(strip_offset as usize + off);
            rgb8.push((val16 >> 8) as u8); // simple 16→8 bit shift
        }
        (rgb8, width, height)
    } else if bps == 8 && spp == 3 {
        (pixel_data.to_vec(), width, height)
    } else {
        panic!("unexpected TIFF format: bps={bps}, spp={spp}");
    }
}

#[test]
fn reference_parity_dcraw() {
    let root = project_root();
    let dng_path = root.join("tests/fixtures/generated/inputs/gradient_64x64.dng");
    // Use dcraw bilinear (-q 0) reference to match our demosaic algorithm
    let dcraw_ref = root.join("tests/fixtures/generated/reference/dng_dcraw_bilinear.tiff");

    if !dng_path.exists() || !dcraw_ref.exists() {
        eprintln!(
            "SKIP: DNG fixture not found. Run: python3 tests/fixtures/scripts/generate_dng_fixture.py && dcraw -T -o 1 -W <dng>"
        );
        return;
    }

    // Our decode
    let dng_data = std::fs::read(&dng_path).unwrap();
    let our_result = rasmcore_raw::decode(&dng_data).expect("our DNG decode failed");

    // dcraw reference
    let (dcraw_rgb, dcraw_w, dcraw_h) = read_tiff_rgb8(&dcraw_ref);
    assert_eq!(our_result.width, dcraw_w, "width mismatch vs dcraw");
    assert_eq!(our_result.height, dcraw_h, "height mismatch vs dcraw");

    let mae = mae_rgb8(&our_result.pixels, &dcraw_rgb);
    eprintln!("MAE vs dcraw: {mae:.2}");
    assert!(mae < 5.0, "MAE vs dcraw = {mae:.2}, exceeds threshold 5.0");
}

#[test]
fn reference_parity_rawpy() {
    let root = project_root();
    let dng_path = root.join("tests/fixtures/generated/inputs/gradient_64x64.dng");
    let rawpy_ref = root.join("tests/fixtures/generated/reference/dng_rawpy_bilinear_rgb8.bin");

    if !dng_path.exists() || !rawpy_ref.exists() {
        eprintln!(
            "SKIP: DNG/rawpy fixture not found. Run: python3 tests/fixtures/scripts/generate_dng_fixture.py"
        );
        return;
    }

    // Our decode
    let dng_data = std::fs::read(&dng_path).unwrap();
    let our_result = rasmcore_raw::decode(&dng_data).expect("our DNG decode failed");

    // rawpy reference (flat RGB8 bytes, 64×64×3)
    let rawpy_rgb = std::fs::read(&rawpy_ref).unwrap();
    assert_eq!(
        our_result.pixels.len(),
        rawpy_rgb.len(),
        "pixel count mismatch vs rawpy"
    );

    let mae = mae_rgb8(&our_result.pixels, &rawpy_rgb);
    eprintln!("MAE vs rawpy: {mae:.2}");
    // rawpy (LibRaw) uses a different WB/exposure normalization than dcraw even
    // with matching flags. dcraw vs rawpy baseline MAE is ~59 for this fixture.
    // We follow dcraw's math, so our-vs-rawpy matches dcraw-vs-rawpy.
    assert!(
        mae < 65.0,
        "MAE vs rawpy = {mae:.2}, exceeds threshold 65.0 (rawpy uses different pipeline)"
    );
}

#[test]
fn three_way_validation() {
    let root = project_root();
    let dng_path = root.join("tests/fixtures/generated/inputs/gradient_64x64.dng");
    let dcraw_ref = root.join("tests/fixtures/generated/reference/dng_dcraw_bilinear.tiff");
    let rawpy_ref = root.join("tests/fixtures/generated/reference/dng_rawpy_bilinear_rgb8.bin");

    if !dng_path.exists() || !dcraw_ref.exists() || !rawpy_ref.exists() {
        eprintln!("SKIP: fixtures not found for three-way validation");
        return;
    }

    // All three decoders
    let dng_data = std::fs::read(&dng_path).unwrap();
    let our_result = rasmcore_raw::decode(&dng_data).expect("our decode failed");
    let (dcraw_rgb, _, _) = read_tiff_rgb8(&dcraw_ref);
    let rawpy_rgb = std::fs::read(&rawpy_ref).unwrap();

    let mae_ours_dcraw = mae_rgb8(&our_result.pixels, &dcraw_rgb);
    let mae_ours_rawpy = mae_rgb8(&our_result.pixels, &rawpy_rgb);
    let mae_dcraw_rawpy = mae_rgb8(&dcraw_rgb, &rawpy_rgb);

    eprintln!("Three-way MAE:");
    eprintln!("  ours vs dcraw:  {mae_ours_dcraw:.2}");
    eprintln!("  ours vs rawpy:  {mae_ours_rawpy:.2}");
    eprintln!("  dcraw vs rawpy: {mae_dcraw_rawpy:.2}");

    // Our decode should be within 5.0 MAE of dcraw (same algorithm)
    assert!(
        mae_ours_dcraw < 5.0,
        "ours vs dcraw MAE = {mae_ours_dcraw:.2} > 5.0"
    );
    // rawpy uses different pipeline; our MAE vs rawpy should match dcraw vs rawpy
    assert!(
        (mae_ours_rawpy - mae_dcraw_rawpy).abs() < 10.0,
        "ours-vs-rawpy ({mae_ours_rawpy:.2}) deviates from dcraw-vs-rawpy ({mae_dcraw_rawpy:.2}) by more than 10.0"
    );
}
