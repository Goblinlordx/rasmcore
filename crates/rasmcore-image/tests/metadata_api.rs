//! Integration tests for the metadata query API.

use rasmcore_image::domain::metadata;
use rasmcore_image::domain::metadata_query;
use rasmcore_image::domain::metadata_set::MetadataSet;
use std::collections::BTreeMap;

/// Build a minimal JPEG with EXIF containing Make, Model, Software fields.
/// Uses write_exif which produces valid EXIF that read_exif can parse.
fn build_jpeg_with_exif() -> Vec<u8> {
    use rasmcore_image::domain::metadata::{ExifMetadata, ExifOrientation, write_exif};

    let exif_meta = ExifMetadata {
        orientation: Some(ExifOrientation::Normal),
        width: Some(64),
        height: Some(64),
        camera_make: Some("TestCamera".to_string()),
        camera_model: Some("TestModel v2".to_string()),
        date_time: Some("2026:03:30 12:00:00".to_string()),
        software: Some("rasmcore-test".to_string()),
    };

    let exif_bytes = write_exif(&exif_meta).unwrap();

    // Build minimal JPEG: SOI + APP1(EXIF) + minimal scan + EOI
    let mut jpeg = vec![0xFF, 0xD8]; // SOI

    // APP1 with EXIF
    jpeg.push(0xFF);
    jpeg.push(0xE1);
    let app1_payload = exif_bytes; // write_exif returns the APP1 payload ("Exif\0\0" + TIFF)
    let seg_len = (app1_payload.len() + 2) as u16;
    jpeg.extend_from_slice(&seg_len.to_be_bytes());
    jpeg.extend_from_slice(&app1_payload);

    // Minimal SOF0 (start of frame)
    jpeg.extend_from_slice(&[
        0xFF, 0xC0, // SOF0 marker
        0x00, 0x0B, // length = 11
        0x08, // precision = 8 bits
        0x00, 0x40, // height = 64
        0x00, 0x40, // width = 64
        0x01, // num_components = 1 (grayscale)
        0x01, 0x11, 0x00, // component 1: id=1, sampling=1x1, quant=0
    ]);

    // Minimal DHT (empty Huffman table — enough for parser)
    jpeg.extend_from_slice(&[
        0xFF, 0xC4, // DHT marker
        0x00, 0x03, // length = 3 (minimal)
        0x00, // DC table 0
    ]);

    // SOS (start of scan)
    jpeg.extend_from_slice(&[
        0xFF, 0xDA, // SOS marker
        0x00, 0x08, // length = 8
        0x01, // num_components = 1
        0x01, 0x00, // component 1, DC=0, AC=0
        0x00, 0x3F, 0x00, // spectral selection: 0-63
    ]);

    // Scan data (empty)
    jpeg.push(0x00);

    // EOI
    jpeg.push(0xFF);
    jpeg.push(0xD9);

    jpeg
}

#[test]
fn dump_jpeg_with_exif() {
    let jpeg = build_jpeg_with_exif();

    let ms = metadata::read_metadata(&jpeg).unwrap();
    assert!(ms.exif.is_some(), "should have EXIF bytes");

    let dump = metadata_query::metadata_dump_from_bytes(&jpeg, &ms);
    eprintln!("Dump: {dump:?}");

    // Should have EXIF container
    assert!(dump.contains_key("exif"), "missing exif container");

    let exif = &dump["exif"];
    assert_eq!(exif.get("Make").map(|s| s.as_str()), Some("TestCamera"));
    assert_eq!(exif.get("Model").map(|s| s.as_str()), Some("TestModel v2"));
    assert_eq!(
        exif.get("Software").map(|s| s.as_str()),
        Some("rasmcore-test")
    );
}

#[test]
fn read_exif_field() {
    let jpeg = build_jpeg_with_exif();
    let ms = metadata::read_metadata(&jpeg).unwrap();

    assert_eq!(
        metadata_query::metadata_read_from_bytes(&jpeg, &ms, "exif.Make").unwrap(),
        Some("TestCamera".to_string())
    );
    assert_eq!(
        metadata_query::metadata_read_from_bytes(&jpeg, &ms, "exif.Model").unwrap(),
        Some("TestModel v2".to_string())
    );
    // Valid path, absent field
    assert_eq!(
        metadata_query::metadata_read_from_bytes(&jpeg, &ms, "xmp.Creator").unwrap(),
        None
    );
}

#[test]
fn read_invalid_paths() {
    let ms = MetadataSet::new();

    // Invalid container
    let err = metadata_query::metadata_read(&ms, "fake.Field").unwrap_err();
    assert!(err.to_string().contains("not a valid metadata container"));

    // Invalid field
    let err = metadata_query::metadata_read(&ms, "exif.NotATag").unwrap_err();
    assert!(err.to_string().contains("not a valid exif field"));

    // Malformed
    let err = metadata_query::metadata_read(&ms, "noperiod").unwrap_err();
    assert!(err.to_string().contains("Expected format"));
}

#[test]
fn dump_json_has_correct_structure() {
    let jpeg = build_jpeg_with_exif();
    let ms = metadata::read_metadata(&jpeg).unwrap();

    let json = metadata_query::metadata_dump_json_from_bytes(&jpeg, &ms);
    eprintln!("JSON: {json}");

    // Should be valid JSON-ish structure
    assert!(json.starts_with('{'));
    assert!(json.ends_with('}'));
    assert!(json.contains("\"exif\""));
    assert!(json.contains("\"Make\""));
    assert!(json.contains("TestCamera"));
}

#[test]
fn validate_metadata_object_accepts_valid() {
    let mut obj = BTreeMap::new();
    let mut exif = BTreeMap::new();
    exif.insert("Artist".into(), "Test".into());
    exif.insert("Make".into(), "Camera".into());
    obj.insert("exif".into(), exif);

    let mut xmp = BTreeMap::new();
    xmp.insert("Creator".into(), "Test".into());
    obj.insert("xmp".into(), xmp);

    let entries = metadata_query::validate_metadata_object(&obj).unwrap();
    assert_eq!(entries.len(), 3);
}

#[test]
fn validate_metadata_object_rejects_bad_container() {
    let mut obj = BTreeMap::new();
    let mut bad = BTreeMap::new();
    bad.insert("Foo".into(), "bar".into());
    obj.insert("notreal".into(), bad);

    let err = metadata_query::validate_metadata_object(&obj).unwrap_err();
    assert!(err.to_string().contains("not a valid metadata container"));
}

#[test]
fn validate_metadata_object_rejects_bad_field() {
    let mut obj = BTreeMap::new();
    let mut exif = BTreeMap::new();
    exif.insert("NotARealTag".into(), "bar".into());
    obj.insert("exif".into(), exif);

    let err = metadata_query::validate_metadata_object(&obj).unwrap_err();
    assert!(err.to_string().contains("not a valid exif field"));
}

#[test]
fn default_output_has_no_metadata() {
    // An empty MetadataSet (default) should produce empty dump
    let ms = MetadataSet::new();
    let dump = metadata_query::metadata_dump(&ms);
    assert!(dump.is_empty());
    assert_eq!(metadata_query::metadata_dump_json(&ms), "{}");
}
