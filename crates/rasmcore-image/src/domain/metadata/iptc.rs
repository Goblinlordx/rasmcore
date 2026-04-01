//! IPTC-IIM metadata parser and serializer.
//!
//! IPTC-IIM (Information Interchange Model) is a binary metadata format
//! stored in JPEG APP13 markers. Each record is: 0x1C + record_number +
//! dataset_number + 2-byte length + data.

use crate::domain::error::ImageError;

/// Parsed IPTC metadata — common fields.
#[derive(Debug, Clone, Default)]
pub struct IptcMetadata {
    pub title: Option<String>,
    pub caption: Option<String>,
    pub keywords: Vec<String>,
    pub byline: Option<String>,
    pub copyright: Option<String>,
    pub category: Option<String>,
    pub urgency: Option<u8>,
}

/// Parse IPTC-IIM binary data.
///
/// Scans for 0x1C tag markers and extracts record 2 (application) datasets.
/// If the data starts with "Photoshop 3.0\0", the IPTC block is located
/// inside the Photoshop IRB structure (8BIM resource blocks).
pub fn parse_iptc(data: &[u8]) -> Result<IptcMetadata, ImageError> {
    let mut meta = IptcMetadata::default();

    // If wrapped in Photoshop 3.0 APP13, find the IPTC resource (8BIM type 0x0404)
    let iptc_data = if data.starts_with(b"Photoshop 3.0\x00") {
        find_iptc_in_photoshop_irb(data).unwrap_or(data)
    } else {
        data
    };

    let mut pos = 0;
    while pos + 5 <= iptc_data.len() {
        // Each IPTC record: 0x1C + record + dataset + length(2) + data
        if iptc_data[pos] != 0x1C {
            pos += 1;
            continue;
        }

        let record = iptc_data[pos + 1];
        let dataset = iptc_data[pos + 2];
        let length = u16::from_be_bytes([iptc_data[pos + 3], iptc_data[pos + 4]]) as usize;
        let data_start = pos + 5;
        let data_end = data_start + length;

        if data_end > iptc_data.len() {
            break;
        }

        let value = &iptc_data[data_start..data_end];

        // We only care about record 2 (application record)
        if record == 2 {
            let text = String::from_utf8_lossy(value).to_string();
            match dataset {
                5 => meta.title = Some(text),           // Object Name
                10 => meta.urgency = text.parse().ok(), // Urgency
                15 => meta.category = Some(text),       // Category
                25 => meta.keywords.push(text),         // Keywords (repeatable)
                80 => meta.byline = Some(text),         // By-line
                116 => meta.copyright = Some(text),     // Copyright Notice
                120 => meta.caption = Some(text),       // Caption/Abstract
                _ => {}
            }
        }

        pos = data_end;
    }

    Ok(meta)
}

/// Serialize IPTC metadata to binary IPTC-IIM format.
pub fn serialize_iptc(meta: &IptcMetadata) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();

    if let Some(ref title) = meta.title {
        write_iptc_record(&mut buf, 2, 5, title.as_bytes());
    }
    if let Some(urgency) = meta.urgency {
        write_iptc_record(&mut buf, 2, 10, urgency.to_string().as_bytes());
    }
    if let Some(ref category) = meta.category {
        write_iptc_record(&mut buf, 2, 15, category.as_bytes());
    }
    for keyword in &meta.keywords {
        write_iptc_record(&mut buf, 2, 25, keyword.as_bytes());
    }
    if let Some(ref byline) = meta.byline {
        write_iptc_record(&mut buf, 2, 80, byline.as_bytes());
    }
    if let Some(ref copyright) = meta.copyright {
        write_iptc_record(&mut buf, 2, 116, copyright.as_bytes());
    }
    if let Some(ref caption) = meta.caption {
        write_iptc_record(&mut buf, 2, 120, caption.as_bytes());
    }

    Ok(buf)
}

fn write_iptc_record(buf: &mut Vec<u8>, record: u8, dataset: u8, data: &[u8]) {
    buf.push(0x1C);
    buf.push(record);
    buf.push(dataset);
    buf.extend_from_slice(&(data.len() as u16).to_be_bytes());
    buf.extend_from_slice(data);
}

/// Find IPTC data inside Photoshop IRB (Image Resource Block) structure.
/// Photoshop APP13 contains "Photoshop 3.0\0" + "8BIM" resource blocks.
/// IPTC data is in resource type 0x0404.
fn find_iptc_in_photoshop_irb(data: &[u8]) -> Option<&[u8]> {
    let header = b"Photoshop 3.0\x008BIM";
    let mut pos = if data.starts_with(b"Photoshop 3.0\x00") {
        14 // skip "Photoshop 3.0\0"
    } else {
        return None;
    };

    while pos + 8 <= data.len() {
        // Each 8BIM block: "8BIM" + type(2) + pascal_string + padding + size(4) + data
        if &data[pos..pos + 4] != b"8BIM" {
            break;
        }

        let resource_type = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
        pos += 6;

        // Skip Pascal string (length byte + string + padding to even)
        if pos >= data.len() {
            break;
        }
        let pascal_len = data[pos] as usize;
        let padded_pascal = if (pascal_len + 1).is_multiple_of(2) {
            pascal_len + 1
        } else {
            pascal_len + 2
        };
        pos += padded_pascal;

        if pos + 4 > data.len() {
            break;
        }

        let block_size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if resource_type == 0x0404 && pos + block_size <= data.len() {
            return Some(&data[pos..pos + block_size]);
        }

        // Skip to next block (padded to even)
        pos += (block_size + 1) & !1;
    }

    let _ = header; // suppress unused warning
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_and_parse_roundtrip() {
        let meta = IptcMetadata {
            title: Some("My Photo".to_string()),
            caption: Some("A beautiful sunset".to_string()),
            keywords: vec!["sunset".to_string(), "nature".to_string()],
            byline: Some("John Doe".to_string()),
            copyright: Some("2026 Acme".to_string()),
            ..Default::default()
        };

        let bytes = serialize_iptc(&meta).unwrap();
        let parsed = parse_iptc(&bytes).unwrap();

        assert_eq!(parsed.title, Some("My Photo".to_string()));
        assert_eq!(parsed.caption, Some("A beautiful sunset".to_string()));
        assert_eq!(parsed.keywords, vec!["sunset", "nature"]);
        assert_eq!(parsed.byline, Some("John Doe".to_string()));
        assert_eq!(parsed.copyright, Some("2026 Acme".to_string()));
    }

    #[test]
    fn parse_empty_data_returns_default() {
        let meta = parse_iptc(&[]).unwrap();
        assert!(meta.title.is_none());
        assert!(meta.keywords.is_empty());
    }

    #[test]
    fn parse_single_keyword() {
        let mut data = Vec::new();
        write_iptc_record(&mut data, 2, 25, b"landscape");
        let meta = parse_iptc(&data).unwrap();
        assert_eq!(meta.keywords, vec!["landscape"]);
    }

    #[test]
    fn serialize_empty_produces_empty() {
        let meta = IptcMetadata::default();
        let bytes = serialize_iptc(&meta).unwrap();
        assert!(bytes.is_empty());
    }
}
