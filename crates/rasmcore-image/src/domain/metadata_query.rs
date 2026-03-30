//! Metadata query API — dump, read, validate dot-paths.
//!
//! Provides structured access to image metadata via dot-path strings
//! (e.g., "exif.Artist", "xmp.Creator", "iptc.Headline", "icc.ProfileName").

use super::error::ImageError;
use super::metadata::{ExifMetadata, read_exif};
use super::metadata_iptc::parse_iptc;
use super::metadata_set::MetadataSet;
use super::metadata_xmp::parse_xmp;
use std::collections::BTreeMap;

// ─── Field Registries ────────────────────────────────────────────────────

/// Valid metadata containers.
pub const VALID_CONTAINERS: &[&str] = &["exif", "xmp", "iptc", "icc"];

/// Valid EXIF field names (PascalCase, matching EXIF tag names).
pub const EXIF_FIELDS: &[&str] = &[
    "Artist",
    "Copyright",
    "DateTime",
    "DateTimeDigitized",
    "DateTimeOriginal",
    "ExposureBiasValue",
    "ExposureTime",
    "FNumber",
    "Flash",
    "FocalLength",
    "GPSAltitude",
    "GPSLatitude",
    "GPSLongitude",
    "ISOSpeedRatings",
    "ImageDescription",
    "LensModel",
    "Make",
    "MeteringMode",
    "Model",
    "Orientation",
    "PixelXDimension",
    "PixelYDimension",
    "Software",
    "WhiteBalance",
];

/// Valid XMP field names.
pub const XMP_FIELDS: &[&str] = &[
    "CreateDate",
    "Creator",
    "CreatorTool",
    "Description",
    "ModifyDate",
    "Rating",
    "Rights",
    "Title",
];

/// Valid IPTC field names.
pub const IPTC_FIELDS: &[&str] = &[
    "Byline",
    "Caption",
    "Category",
    "Copyright",
    "Headline",
    "Keywords",
    "Title",
    "Urgency",
];

/// Valid ICC field names.
pub const ICC_FIELDS: &[&str] = &["ColorSpace", "ProfileName"];

/// Get the valid fields for a container.
fn fields_for_container(container: &str) -> Option<&'static [&'static str]> {
    match container {
        "exif" => Some(EXIF_FIELDS),
        "xmp" => Some(XMP_FIELDS),
        "iptc" => Some(IPTC_FIELDS),
        "icc" => Some(ICC_FIELDS),
        _ => None,
    }
}

// ─── Path Validation ─────────────────────────────────────────────────────

/// Validate a dot-path and return (container, field).
pub fn validate_path(path: &str) -> Result<(&str, &str), ImageError> {
    let parts: Vec<&str> = path.splitn(2, '.').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(ImageError::InvalidParameters(format!(
            "Invalid metadata path \"{path}\". Expected format: container.Field (e.g., exif.Artist)"
        )));
    }
    let container = parts[0];
    let field = parts[1];

    let valid_fields = fields_for_container(container).ok_or_else(|| {
        ImageError::InvalidParameters(format!(
            "\"{container}\" is not a valid metadata container. Valid containers: {}",
            VALID_CONTAINERS.join(", ")
        ))
    })?;

    if !valid_fields.contains(&field) {
        return Err(ImageError::InvalidParameters(format!(
            "\"{field}\" is not a valid {container} field. Valid fields: {}",
            valid_fields.join(", ")
        )));
    }

    Ok((container, field))
}

// ─── Dump ────────────────────────────────────────────────────────────────

/// Dump all metadata from raw encoded image bytes.
///
/// Returns: `{ "exif": { "Artist": "...", ... }, "xmp": { ... }, ... }`
/// Only containers and fields that are present are included.
/// Use this with the original encoded bytes (not MetadataSet) for EXIF.
pub fn metadata_dump_from_bytes(
    data: &[u8],
    ms: &MetadataSet,
) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut result = BTreeMap::new();

    // EXIF — parse from original bytes for best compatibility
    if ms.exif.is_some()
        && let Ok(exif) = read_exif(data) {
            let fields = exif_to_fields(&exif);
            if !fields.is_empty() {
                result.insert("exif".into(), fields);
            }
        }

    add_xmp_fields(&mut result, ms);
    add_iptc_fields(&mut result, ms);
    add_icc_fields(&mut result, ms);

    result
}

/// Dump all metadata from a MetadataSet only (no original bytes needed).
/// EXIF is parsed from the raw APP1 payload via a reconstructed JPEG wrapper.
pub fn metadata_dump(ms: &MetadataSet) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut result = BTreeMap::new();

    // EXIF
    if let Some(ref exif_bytes) = ms.exif
        && let Ok(exif) = parse_exif_from_raw(exif_bytes) {
            let fields = exif_to_fields(&exif);
            if !fields.is_empty() {
                result.insert("exif".into(), fields);
            }
        }

    add_xmp_fields(&mut result, ms);
    add_iptc_fields(&mut result, ms);
    add_icc_fields(&mut result, ms);

    result
}

fn exif_to_fields(exif: &ExifMetadata) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    if let Some(ref v) = exif.orientation {
        fields.insert("Orientation".into(), format!("{}", v.to_tag()));
    }
    if let Some(v) = exif.width {
        fields.insert("PixelXDimension".into(), v.to_string());
    }
    if let Some(v) = exif.height {
        fields.insert("PixelYDimension".into(), v.to_string());
    }
    if let Some(ref v) = exif.camera_make {
        fields.insert("Make".into(), v.clone());
    }
    if let Some(ref v) = exif.camera_model {
        fields.insert("Model".into(), v.clone());
    }
    if let Some(ref v) = exif.date_time {
        fields.insert("DateTime".into(), v.clone());
    }
    if let Some(ref v) = exif.software {
        fields.insert("Software".into(), v.clone());
    }
    fields
}

fn add_xmp_fields(result: &mut BTreeMap<String, BTreeMap<String, String>>, ms: &MetadataSet) {
    if let Some(ref xmp_bytes) = ms.xmp
        && let Ok(xmp) = parse_xmp(xmp_bytes) {
            let mut fields = BTreeMap::new();
            if let Some(ref v) = xmp.title {
                fields.insert("Title".into(), v.clone());
            }
            if let Some(ref v) = xmp.description {
                fields.insert("Description".into(), v.clone());
            }
            if let Some(ref v) = xmp.creator {
                fields.insert("Creator".into(), v.clone());
            }
            if let Some(ref v) = xmp.rights {
                fields.insert("Rights".into(), v.clone());
            }
            if let Some(ref v) = xmp.create_date {
                fields.insert("CreateDate".into(), v.clone());
            }
            if let Some(ref v) = xmp.modify_date {
                fields.insert("ModifyDate".into(), v.clone());
            }
            if let Some(ref v) = xmp.creator_tool {
                fields.insert("CreatorTool".into(), v.clone());
            }
            if !fields.is_empty() {
                result.insert("xmp".into(), fields);
            }
        }
}

fn add_iptc_fields(result: &mut BTreeMap<String, BTreeMap<String, String>>, ms: &MetadataSet) {
    if let Some(ref iptc_bytes) = ms.iptc
        && let Ok(iptc) = parse_iptc(iptc_bytes) {
            let mut fields = BTreeMap::new();
            if let Some(ref v) = iptc.title {
                fields.insert("Title".into(), v.clone());
            }
            if let Some(ref v) = iptc.caption {
                fields.insert("Caption".into(), v.clone());
            }
            if !iptc.keywords.is_empty() {
                fields.insert("Keywords".into(), iptc.keywords.join(", "));
            }
            if let Some(ref v) = iptc.byline {
                fields.insert("Byline".into(), v.clone());
            }
            if let Some(ref v) = iptc.copyright {
                fields.insert("Copyright".into(), v.clone());
            }
            if let Some(ref v) = iptc.category {
                fields.insert("Category".into(), v.clone());
            }
            if let Some(v) = iptc.urgency {
                fields.insert("Urgency".into(), v.to_string());
            }
            if !fields.is_empty() {
                result.insert("iptc".into(), fields);
            }
        }
}

fn add_icc_fields(result: &mut BTreeMap<String, BTreeMap<String, String>>, ms: &MetadataSet) {
    if let Some(ref icc_bytes) = ms.icc_profile {
        let mut fields = BTreeMap::new();
        if icc_bytes.len() >= 128 {
            let cs_bytes = &icc_bytes[16..20];
            let cs = std::str::from_utf8(cs_bytes)
                .unwrap_or("unknown")
                .trim()
                .to_string();
            fields.insert("ColorSpace".into(), cs);
            if let Some(name) = extract_icc_description(icc_bytes) {
                fields.insert("ProfileName".into(), name);
            }
        }
        if !fields.is_empty() {
            result.insert("icc".into(), fields);
        }
    }
}

/// Read a single metadata value by dot-path.
///
/// Returns `None` for valid paths where the field is not present in the data.
/// Returns `Err` for invalid paths (bad container, bad field, malformed).
pub fn metadata_read(ms: &MetadataSet, path: &str) -> Result<Option<String>, ImageError> {
    let (container, field) = validate_path(path)?;

    let dump = metadata_dump(ms);
    Ok(dump
        .get(container)
        .and_then(|fields| fields.get(field))
        .cloned())
}

/// Dump metadata as JSON string.
/// Read a single metadata value using original bytes for EXIF parsing.
pub fn metadata_read_from_bytes(
    data: &[u8],
    ms: &MetadataSet,
    path: &str,
) -> Result<Option<String>, ImageError> {
    let (container, field) = validate_path(path)?;
    let dump = metadata_dump_from_bytes(data, ms);
    Ok(dump
        .get(container)
        .and_then(|fields| fields.get(field))
        .cloned())
}

/// Dump metadata as JSON string using original bytes for EXIF.
pub fn metadata_dump_json_from_bytes(data: &[u8], ms: &MetadataSet) -> String {
    let dump = metadata_dump_from_bytes(data, ms);
    dump_to_json(&dump)
}

pub fn metadata_dump_json(ms: &MetadataSet) -> String {
    let dump = metadata_dump(ms);
    dump_to_json(&dump)
}

fn dump_to_json(dump: &BTreeMap<String, BTreeMap<String, String>>) -> String {
    // Simple JSON serialization without serde dependency
    let mut json = String::from("{");
    let mut first_container = true;
    for (container, fields) in dump {
        if !first_container {
            json.push_str(", ");
        }
        first_container = false;
        json.push('"');
        json.push_str(container);
        json.push_str("\": {");
        let mut first_field = true;
        for (key, value) in fields {
            if !first_field {
                json.push_str(", ");
            }
            first_field = false;
            json.push('"');
            json.push_str(key);
            json.push_str("\": ");
            json.push('"');
            // Escape JSON special chars in value
            for ch in value.chars() {
                match ch {
                    '"' => json.push_str("\\\""),
                    '\\' => json.push_str("\\\\"),
                    '\n' => json.push_str("\\n"),
                    '\r' => json.push_str("\\r"),
                    '\t' => json.push_str("\\t"),
                    c => json.push(c),
                }
            }
            json.push('"');
        }
        json.push('}');
    }
    json.push('}');
    json
}

// ─── Load Metadata (bulk set from JSON-like structure) ───────────────────

/// Validate a dump()-shaped object: { container: { field: value } }.
/// Returns all validated (container, field, value) triples, or the first error.
pub fn validate_metadata_object(
    obj: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<Vec<(String, String, String)>, ImageError> {
    let mut entries = Vec::new();
    for (container, fields) in obj {
        let valid_fields = fields_for_container(container).ok_or_else(|| {
            ImageError::InvalidParameters(format!(
                "\"{container}\" is not a valid metadata container. Valid containers: {}",
                VALID_CONTAINERS.join(", ")
            ))
        })?;
        for (field, value) in fields {
            if !valid_fields.contains(&field.as_str()) {
                return Err(ImageError::InvalidParameters(format!(
                    "\"{field}\" is not a valid {container} field. Valid fields: {}",
                    valid_fields.join(", ")
                )));
            }
            entries.push((container.clone(), field.clone(), value.clone()));
        }
    }
    Ok(entries)
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Parse EXIF from raw EXIF bytes (may include APP1 header or raw TIFF).
fn parse_exif_from_raw(exif_bytes: &[u8]) -> Result<ExifMetadata, ImageError> {
    // Try parsing as a JPEG with APP1 marker
    // The raw bytes in MetadataSet are the APP1 payload (after marker + length)
    // which starts with "Exif\0\0" followed by TIFF header.
    // The exif crate's Reader expects a full container (JPEG/TIFF/WebP).
    // Wrap in a minimal JPEG for parsing.
    let mut jpeg = vec![0xFF, 0xD8]; // SOI
    // APP1 marker
    jpeg.push(0xFF);
    jpeg.push(0xE1);
    let len = (exif_bytes.len() + 2) as u16;
    jpeg.extend_from_slice(&len.to_be_bytes());
    jpeg.extend_from_slice(exif_bytes);
    // EOI
    jpeg.push(0xFF);
    jpeg.push(0xD9);

    read_exif(&jpeg)
}

/// Extract ICC profile description from the 'desc' tag.
fn extract_icc_description(icc: &[u8]) -> Option<String> {
    if icc.len() < 132 {
        return None;
    }
    // ICC profile tag table starts at offset 128
    let tag_count = u32::from_be_bytes([icc[128], icc[129], icc[130], icc[131]]) as usize;
    let mut pos = 132;
    for _ in 0..tag_count {
        if pos + 12 > icc.len() {
            break;
        }
        let sig = &icc[pos..pos + 4];
        let offset =
            u32::from_be_bytes([icc[pos + 4], icc[pos + 5], icc[pos + 6], icc[pos + 7]]) as usize;
        let size =
            u32::from_be_bytes([icc[pos + 8], icc[pos + 9], icc[pos + 10], icc[pos + 11]]) as usize;

        if sig == b"desc" && offset + size <= icc.len() && size > 12 {
            // textDescriptionType: sig(4) + reserved(4) + length(4) + ASCII string
            let desc_data = &icc[offset..offset + size];
            let type_sig = &desc_data[0..4];
            if type_sig == b"desc" && desc_data.len() > 12 {
                let str_len =
                    u32::from_be_bytes([desc_data[8], desc_data[9], desc_data[10], desc_data[11]])
                        as usize;
                if str_len > 0 && 12 + str_len <= desc_data.len() {
                    let s = &desc_data[12..12 + str_len];
                    return Some(
                        std::str::from_utf8(s)
                            .unwrap_or("")
                            .trim_end_matches('\0')
                            .to_string(),
                    );
                }
            }
            // mluc type (multi-localized unicode)
            if type_sig == b"mluc" && desc_data.len() > 20 {
                let record_count =
                    u32::from_be_bytes([desc_data[8], desc_data[9], desc_data[10], desc_data[11]])
                        as usize;
                if record_count > 0 && desc_data.len() > 28 {
                    let str_offset = u32::from_be_bytes([
                        desc_data[24],
                        desc_data[25],
                        desc_data[26],
                        desc_data[27],
                    ]) as usize;
                    let str_len = u32::from_be_bytes([
                        desc_data[20],
                        desc_data[21],
                        desc_data[22],
                        desc_data[23],
                    ]) as usize;
                    if str_offset + str_len <= desc_data.len() {
                        // UTF-16BE encoded
                        let utf16: Vec<u16> = desc_data[str_offset..str_offset + str_len]
                            .chunks_exact(2)
                            .map(|c| u16::from_be_bytes([c[0], c[1]]))
                            .collect();
                        return Some(
                            String::from_utf16_lossy(&utf16)
                                .trim_end_matches('\0')
                                .to_string(),
                        );
                    }
                }
            }
        }
        pos += 12;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_path_valid() {
        assert!(validate_path("exif.Artist").is_ok());
        assert!(validate_path("xmp.Creator").is_ok());
        assert!(validate_path("iptc.Headline").is_ok());
        assert!(validate_path("icc.ProfileName").is_ok());
    }

    #[test]
    fn validate_path_invalid_container() {
        let err = validate_path("asdf.Foo").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a valid metadata container"));
        assert!(msg.contains("exif, xmp, iptc, icc"));
    }

    #[test]
    fn validate_path_invalid_field() {
        let err = validate_path("exif.FakeField").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a valid exif field"));
        assert!(msg.contains("Artist"));
    }

    #[test]
    fn validate_path_malformed() {
        let err = validate_path("noperiod").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Expected format: container.Field"));
    }

    #[test]
    fn dump_empty_metadata_set() {
        let ms = MetadataSet::new();
        let dump = metadata_dump(&ms);
        assert!(dump.is_empty());
    }

    #[test]
    fn dump_json_empty() {
        let ms = MetadataSet::new();
        assert_eq!(metadata_dump_json(&ms), "{}");
    }

    #[test]
    fn read_from_empty() {
        let ms = MetadataSet::new();
        // Valid path but no data → None
        assert_eq!(metadata_read(&ms, "exif.Artist").unwrap(), None);
    }

    #[test]
    fn read_invalid_path() {
        let ms = MetadataSet::new();
        assert!(metadata_read(&ms, "bad").is_err());
        assert!(metadata_read(&ms, "exif.FakeField").is_err());
    }

    #[test]
    fn validate_metadata_object_valid() {
        let mut obj = BTreeMap::new();
        let mut exif = BTreeMap::new();
        exif.insert("Artist".into(), "test".into());
        exif.insert("Make".into(), "Camera".into());
        obj.insert("exif".into(), exif);
        let entries = validate_metadata_object(&obj).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn validate_metadata_object_bad_container() {
        let mut obj = BTreeMap::new();
        obj.insert("fake".into(), BTreeMap::new());
        assert!(validate_metadata_object(&obj).is_err());
    }

    #[test]
    fn validate_metadata_object_bad_field() {
        let mut obj = BTreeMap::new();
        let mut exif = BTreeMap::new();
        exif.insert("NotReal".into(), "test".into());
        obj.insert("exif".into(), exif);
        assert!(validate_metadata_object(&obj).is_err());
    }
}
