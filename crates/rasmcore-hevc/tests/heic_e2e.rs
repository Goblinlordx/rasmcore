//! End-to-end HEIC decode tests.
//!
//! Tests the full pipeline: HEIC file (ISOBMFF container) → parse → HEVC decode → pixels.
//! Exercises rasmcore-isobmff + rasmcore-hevc together via rasmcore-image's decode_heif path.

use rasmcore_hevc::testutil::{fixtures_available, load_fixture};

/// Build a minimal HEIF container wrapping a raw HEVC Annex B bitstream.
///
/// Creates: ftyp(heic) + meta(pitm + iinf/infe(hvc1) + iloc + iprp(ispe)) + mdat(bitstream)
fn wrap_in_heif(hevc_bitstream: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut file = Vec::new();

    // ftyp box
    let ftyp_content = b"heic\x00\x00\x00\x00heicmif1";
    let ftyp_size = (8 + ftyp_content.len()) as u32;
    file.extend_from_slice(&ftyp_size.to_be_bytes());
    file.extend_from_slice(b"ftyp");
    file.extend_from_slice(ftyp_content);

    // Build meta children first to compute mdat offset

    // pitm (primary item = 1)
    let pitm = make_full_box(b"pitm", 0, 0, &[0x00, 0x01]);

    // infe (item 1, type hvc1)
    let mut infe_content = Vec::new();
    infe_content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
    infe_content.extend_from_slice(&[0x00, 0x00]); // protection = 0
    infe_content.extend_from_slice(b"hvc1");
    infe_content.push(0x00); // name
    let infe = make_full_box(b"infe", 2, 0, &infe_content);

    // iinf
    let mut iinf_content = Vec::new();
    iinf_content.extend_from_slice(&[0x00, 0x01]); // count = 1
    iinf_content.extend(&infe);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_content);

    // ispe property
    let mut ispe_content = Vec::new();
    ispe_content.extend_from_slice(&width.to_be_bytes());
    ispe_content.extend_from_slice(&height.to_be_bytes());
    let ispe = make_full_box(b"ispe", 0, 0, &ispe_content);

    // ipco (property container)
    let ipco = make_box(b"ipco", &ispe);

    // ipma (associate prop 1 to item 1)
    let mut ipma_content = Vec::new();
    ipma_content.extend_from_slice(&1u32.to_be_bytes()); // 1 entry
    ipma_content.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
    ipma_content.push(1); // 1 association
    ipma_content.push(0x81); // essential, prop 1
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_content);

    // iprp
    let mut iprp_content = Vec::new();
    iprp_content.extend(&ipco);
    iprp_content.extend(&ipma);
    let iprp = make_box(b"iprp", &iprp_content);

    // iloc — will point into mdat
    // offset_size=4, length_size=4, base_offset_size=4
    let mut iloc_content = Vec::new();
    iloc_content.push(0x44); // offset_size=4, length_size=4
    iloc_content.push(0x40); // base_offset_size=4
    iloc_content.extend_from_slice(&[0x00, 0x01]); // 1 item
    iloc_content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
    iloc_content.extend_from_slice(&[0x00, 0x00]); // data_ref = 0
    iloc_content.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // base_offset placeholder
    iloc_content.extend_from_slice(&[0x00, 0x01]); // extent_count = 1
    iloc_content.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // extent_offset = 0
    iloc_content.extend_from_slice(&(hevc_bitstream.len() as u32).to_be_bytes());
    let iloc = make_full_box(b"iloc", 0, 0, &iloc_content);

    // meta box
    let mut meta_content = Vec::new();
    meta_content.extend(&pitm);
    meta_content.extend(&iinf);
    meta_content.extend(&iloc);
    meta_content.extend(&iprp);
    let meta = make_full_box(b"meta", 0, 0, &meta_content);

    // mdat box
    let mdat = make_box(b"mdat", hevc_bitstream);

    // Compute mdat content offset (ftyp + meta + 8 bytes mdat header)
    let mdat_content_offset = (ftyp_size as usize + meta.len() + 8) as u32;

    // Assemble file
    file.extend(&meta);
    file.extend(&mdat);

    // Patch iloc base_offset to point to mdat content
    // iloc is inside meta. Find base_offset field.
    let iloc_start_in_meta = 12 + pitm.len() + iinf.len(); // meta full-box header + pitm + iinf
    let base_offset_pos = ftyp_size as usize + iloc_start_in_meta + 12 + 8; // iloc full-box + fields before base_offset
    let offset_bytes = mdat_content_offset.to_be_bytes();
    file[base_offset_pos] = offset_bytes[0];
    file[base_offset_pos + 1] = offset_bytes[1];
    file[base_offset_pos + 2] = offset_bytes[2];
    file[base_offset_pos + 3] = offset_bytes[3];

    file
}

fn make_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = (8 + content.len()) as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(fourcc);
    buf.extend_from_slice(content);
    buf
}

fn make_full_box(fourcc: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let size = (12 + content.len()) as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(fourcc);
    buf.push(version);
    let fb = flags.to_be_bytes();
    buf.extend_from_slice(&fb[1..4]);
    buf.extend_from_slice(content);
    buf
}

#[test]
fn heif_container_parse_roundtrip() {
    if !fixtures_available() {
        eprintln!("SKIPPED: fixtures not generated");
        return;
    }

    let hevc_data = load_fixture("flat_64x64_q22", "hevc").unwrap();
    let heic_file = wrap_in_heif(&hevc_data, 64, 64);

    // Verify ISOBMFF detection
    let brand = rasmcore_isobmff::detect(&heic_file);
    assert!(brand.is_some(), "should detect as HEIF");
    assert!(brand.unwrap().is_heic(), "should detect as HEIC brand");

    // Verify container parse
    let parsed = rasmcore_isobmff::parse(&heic_file).unwrap();
    assert_eq!(parsed.primary_image.width, 64);
    assert_eq!(parsed.primary_image.height, 64);
    assert_eq!(
        parsed.primary_image.codec,
        rasmcore_isobmff::CodecType::Hevc
    );
    assert_eq!(
        parsed.primary_image.bitstream.len(),
        hevc_data.len(),
        "bitstream should be extracted intact"
    );
    assert_eq!(
        parsed.primary_image.bitstream, hevc_data,
        "bitstream should match original"
    );

    eprintln!(
        "HEIF roundtrip OK: {}x{}, codec=HEVC, bitstream={} bytes",
        parsed.primary_image.width,
        parsed.primary_image.height,
        parsed.primary_image.bitstream.len()
    );
}

#[test]
fn heif_ftyp_detection_all_brands() {
    // Test that various HEIF brands are detected
    for brand_str in [b"heic", b"heix", b"avif", b"mif1"] {
        let mut ftyp_content = Vec::new();
        ftyp_content.extend_from_slice(brand_str);
        ftyp_content.extend_from_slice(&0u32.to_be_bytes());
        ftyp_content.extend_from_slice(brand_str);
        let ftyp = make_box(b"ftyp", &ftyp_content);

        let brand = rasmcore_isobmff::detect(&ftyp);
        assert!(
            brand.is_some(),
            "brand {:?} should be detected",
            std::str::from_utf8(brand_str)
        );
    }

    // Non-HEIF brand should NOT be detected
    let mut ftyp_content = Vec::new();
    ftyp_content.extend_from_slice(b"isom");
    ftyp_content.extend_from_slice(&0u32.to_be_bytes());
    ftyp_content.extend_from_slice(b"isom");
    let ftyp = make_box(b"ftyp", &ftyp_content);
    assert!(rasmcore_isobmff::detect(&ftyp).is_none());
}

#[test]

fn heif_end_to_end_decode() {
    if !fixtures_available() {
        return;
    }

    let hevc_data = load_fixture("flat_64x64_q22", "hevc").unwrap();
    let heic_file = wrap_in_heif(&hevc_data, 64, 64);

    // Parse container
    let parsed = rasmcore_isobmff::parse(&heic_file).unwrap();

    // Decode HEVC bitstream
    let frame = rasmcore_hevc::decode_frame(&parsed.primary_image.bitstream, None)
        .expect("should decode flat 64x64");

    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.pixels.len(), 64 * 64 * 3);

    // Verify pixels are reasonable for a flat gray image
    let avg: f64 = frame.pixels.iter().map(|&v| v as f64).sum::<f64>() / frame.pixels.len() as f64;
    assert!(
        (avg - 128.0).abs() < 30.0,
        "flat gray image average should be near 128, got {avg:.1}"
    );
}
