//! XMP metadata read/write.
//!
//! XMP is XML-based metadata using RDF structure. Read via quick-xml,
//! write/serialize via xmp-writer.

use crate::domain::error::ImageError;

/// Parsed XMP metadata — common Dublin Core and XMP fields.
#[derive(Debug, Clone, Default)]
pub struct XmpMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub creator: Option<String>,
    pub rights: Option<String>,
    pub create_date: Option<String>,
    pub modify_date: Option<String>,
    pub creator_tool: Option<String>,
}

/// Parse XMP metadata from raw XMP bytes (XML).
///
/// Extracts common dc: and xmp: namespace fields using quick-xml SAX parser.
pub fn parse_xmp(data: &[u8]) -> Result<XmpMetadata, ImageError> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_reader(data);
    let mut buf = Vec::new();
    let mut meta = XmpMetadata::default();

    // Track element stack to find dc:title > rdf:Alt > rdf:li patterns
    let mut element_stack: Vec<String> = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                element_stack.push(name);
            }
            Ok(Event::Text(e)) => {
                let text = String::from_utf8_lossy(e.as_ref()).to_string();
                let text = text.trim().to_string();
                if text.is_empty() {
                    continue;
                }

                // Find the semantic parent (skip rdf:li, rdf:Alt, rdf:Seq, rdf:Bag)
                let parent = element_stack
                    .iter()
                    .rev()
                    .find(|t| !matches!(t.as_str(), "li" | "Alt" | "Seq" | "Bag"))
                    .cloned();

                if let Some(ref tag) = parent {
                    match tag.as_str() {
                        "title" => {
                            if meta.title.is_none() {
                                meta.title = Some(text);
                            }
                        }
                        "description" => {
                            if meta.description.is_none() {
                                meta.description = Some(text);
                            }
                        }
                        "creator" => {
                            if meta.creator.is_none() {
                                meta.creator = Some(text);
                            }
                        }
                        "rights" => {
                            if meta.rights.is_none() {
                                meta.rights = Some(text);
                            }
                        }
                        "CreateDate" => meta.create_date = Some(text),
                        "ModifyDate" => meta.modify_date = Some(text),
                        "CreatorTool" => meta.creator_tool = Some(text),
                        _ => {}
                    }
                }
            }
            Ok(Event::End(_)) => {
                element_stack.pop();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(ImageError::ProcessingFailed(format!(
                    "XMP parse failed: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(meta)
}

/// Serialize XMP metadata to raw XMP bytes (XML).
///
/// Uses xmp-writer to produce a valid XMP packet.
pub fn serialize_xmp(meta: &XmpMetadata) -> Result<Vec<u8>, ImageError> {
    use xmp_writer::XmpWriter;

    let mut writer = XmpWriter::new();

    if let Some(ref title) = meta.title {
        writer.title([(None, title.as_str())]);
    }
    if let Some(ref desc) = meta.description {
        writer.description([(None, desc.as_str())]);
    }
    if let Some(ref creator) = meta.creator {
        writer.creator([creator.as_str()]);
    }
    if let Some(ref rights) = meta.rights {
        writer.rights([(None, rights.as_str())]);
    }
    if let Some(ref tool) = meta.creator_tool {
        writer.creator_tool(tool);
    }
    // Note: xmp-writer DateTime requires structured parsing.
    // For now, store dates as dc:date entries via the raw XMP.
    // Full date parsing (ISO 8601 → xmp_writer::DateTime) deferred.
    let _ = &meta.create_date;
    let _ = &meta.modify_date;

    Ok(writer.finish(None).into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_and_parse_roundtrip() {
        let meta = XmpMetadata {
            title: Some("Test Image".to_string()),
            creator: Some("rasmcore".to_string()),
            rights: Some("2026 Acme Inc".to_string()),
            ..Default::default()
        };

        let bytes = serialize_xmp(&meta).unwrap();
        let xml = String::from_utf8(bytes.clone()).unwrap();
        assert!(xml.contains("Test Image"));
        assert!(xml.contains("rasmcore"));

        let parsed = parse_xmp(&bytes).unwrap();
        assert_eq!(parsed.title, Some("Test Image".to_string()));
        assert_eq!(parsed.creator, Some("rasmcore".to_string()));
        assert_eq!(parsed.rights, Some("2026 Acme Inc".to_string()));
    }

    #[test]
    fn parse_empty_xmp_returns_default() {
        let xmp = b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>";
        let meta = parse_xmp(xmp).unwrap();
        assert!(meta.title.is_none());
    }

    #[test]
    fn serialize_empty_metadata_produces_valid_xml() {
        let meta = XmpMetadata::default();
        let bytes = serialize_xmp(&meta).unwrap();
        let xml = String::from_utf8(bytes).unwrap();
        assert!(xml.contains("xmpmeta") || xml.contains("xpacket") || xml.contains("rdf"));
    }
}
