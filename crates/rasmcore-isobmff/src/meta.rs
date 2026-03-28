//! Meta box parser — parses pitm, iinf, iloc, and iref sub-boxes.

use crate::boxreader::{BoxIterator, read_full_box_header};
use crate::error::IsobmffError;
use crate::types::{Extent, ItemInfo, ItemLocation, ItemReference, MetaBox, ReferenceType};

/// Parse the meta box and all its children from the given data.
///
/// `data` is the full file data. `meta_offset` is the byte offset of the meta box.
pub fn parse_meta(data: &[u8], meta_offset: usize) -> Result<MetaBox, IsobmffError> {
    let full = read_full_box_header(data, meta_offset)?;
    let content_start = full.box_header.content_offset;
    let content_end = match full.box_header.content_size {
        Some(s) => content_start + s as usize,
        None => data.len(),
    };

    let mut primary_item_id: Option<u32> = None;
    let mut items = Vec::new();
    let mut locations = Vec::new();
    let mut references = Vec::new();

    for box_result in BoxIterator::new(data, content_start, content_end) {
        let header = box_result?;
        match &header.box_type {
            b"pitm" => {
                primary_item_id = Some(parse_pitm(
                    data,
                    header.content_offset - header.header_size as usize,
                )?);
            }
            b"iinf" => {
                items = parse_iinf(data, header.content_offset - header.header_size as usize)?;
            }
            b"iloc" => {
                locations = parse_iloc(data, header.content_offset - header.header_size as usize)?;
            }
            b"iref" => {
                references = parse_iref(data, header.content_offset - header.header_size as usize)?;
            }
            _ => {
                // Skip unknown boxes (iprp handled by properties track)
            }
        }
    }

    Ok(MetaBox {
        primary_item_id: primary_item_id.unwrap_or(0),
        items,
        locations,
        references,
    })
}

/// Parse pitm (primary item ID) box.
fn parse_pitm(data: &[u8], offset: usize) -> Result<u32, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let avail = data.len().saturating_sub(pos);

    if full.version == 0 {
        if avail < 2 {
            return Err(IsobmffError::Truncated {
                expected: 2,
                available: avail,
            });
        }
        Ok(u16::from_be_bytes([data[pos], data[pos + 1]]) as u32)
    } else {
        if avail < 4 {
            return Err(IsobmffError::Truncated {
                expected: 4,
                available: avail,
            });
        }
        Ok(u32::from_be_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]))
    }
}

/// Parse iinf (item info) box — contains a list of infe entries.
fn parse_iinf(data: &[u8], offset: usize) -> Result<Vec<ItemInfo>, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let avail = data.len().saturating_sub(pos);

    let (entry_count, entries_start) = if full.version == 0 {
        if avail < 2 {
            return Err(IsobmffError::Truncated {
                expected: 2,
                available: avail,
            });
        }
        (
            u16::from_be_bytes([data[pos], data[pos + 1]]) as u32,
            pos + 2,
        )
    } else {
        if avail < 4 {
            return Err(IsobmffError::Truncated {
                expected: 4,
                available: avail,
            });
        }
        (
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]),
            pos + 4,
        )
    };

    let content_end = match full.box_header.content_size {
        Some(s) => full.box_header.content_offset + s as usize,
        None => data.len(),
    };

    let mut items = Vec::with_capacity(entry_count as usize);
    let mut count = 0u32;

    for box_result in BoxIterator::new(data, entries_start, content_end) {
        let header = box_result?;
        if &header.box_type == b"infe" {
            items.push(parse_infe(
                data,
                header.content_offset - header.header_size as usize,
            )?);
            count += 1;
            if count >= entry_count {
                break;
            }
        }
    }

    Ok(items)
}

/// Parse a single infe (item info entry) box.
fn parse_infe(data: &[u8], offset: usize) -> Result<ItemInfo, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let content_end = match full.box_header.content_size {
        Some(s) => pos + s as usize,
        None => data.len(),
    };
    let avail = content_end.saturating_sub(pos);

    if full.version >= 2 {
        // Version 2: item_id(u16), protection_index(u16), item_type(4), item_name(null-term)
        // Version 3: item_id(u32), protection_index(u16), item_type(4), item_name(null-term)
        let (item_id, after_id) = if full.version == 2 {
            if avail < 2 {
                return Err(IsobmffError::Truncated {
                    expected: 2,
                    available: avail,
                });
            }
            (
                u16::from_be_bytes([data[pos], data[pos + 1]]) as u32,
                pos + 2,
            )
        } else {
            // version 3
            if avail < 4 {
                return Err(IsobmffError::Truncated {
                    expected: 4,
                    available: avail,
                });
            }
            (
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]),
                pos + 4,
            )
        };

        let remaining = content_end.saturating_sub(after_id);
        if remaining < 6 {
            return Err(IsobmffError::Truncated {
                expected: 6,
                available: remaining,
            });
        }

        let protection_index = u16::from_be_bytes([data[after_id], data[after_id + 1]]);

        let mut item_type = [0u8; 4];
        item_type.copy_from_slice(&data[after_id + 2..after_id + 6]);

        // item_name is a null-terminated string after item_type
        let name_start = after_id + 6;
        let item_name = read_null_terminated_string(data, name_start, content_end);

        Ok(ItemInfo {
            item_id,
            item_type,
            item_name,
            protection_index,
        })
    } else {
        // Version 0/1: item_id(u16), protection_index(u16) — no item_type
        if avail < 4 {
            return Err(IsobmffError::Truncated {
                expected: 4,
                available: avail,
            });
        }
        let item_id = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
        let protection_index = u16::from_be_bytes([data[pos + 2], data[pos + 3]]);

        // item_name follows as null-terminated string
        let item_name = read_null_terminated_string(data, pos + 4, content_end);

        Ok(ItemInfo {
            item_id,
            item_type: [0; 4], // Not available in v0/v1
            item_name,
            protection_index,
        })
    }
}

/// Parse iloc (item location) box.
fn parse_iloc(data: &[u8], offset: usize) -> Result<Vec<ItemLocation>, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let content_end = match full.box_header.content_size {
        Some(s) => pos + s as usize,
        None => data.len(),
    };
    let avail = content_end.saturating_sub(pos);

    if avail < 2 {
        return Err(IsobmffError::Truncated {
            expected: 2,
            available: avail,
        });
    }

    // First two bytes encode field sizes as nibbles
    let offset_size = (data[pos] >> 4) & 0x0F;
    let length_size = data[pos] & 0x0F;
    let base_offset_size = (data[pos + 1] >> 4) & 0x0F;
    let index_size = if full.version >= 1 {
        data[pos + 1] & 0x0F
    } else {
        0
    };

    let mut cursor = pos + 2;

    // Item count
    let item_count = if full.version < 2 {
        if cursor + 2 > content_end {
            return Err(IsobmffError::Truncated {
                expected: 2,
                available: content_end.saturating_sub(cursor),
            });
        }
        let c = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as u32;
        cursor += 2;
        c
    } else {
        if cursor + 4 > content_end {
            return Err(IsobmffError::Truncated {
                expected: 4,
                available: content_end.saturating_sub(cursor),
            });
        }
        let c = u32::from_be_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        cursor += 4;
        c
    };

    let mut locations = Vec::with_capacity(item_count as usize);

    for _ in 0..item_count {
        // item_id
        let item_id = if full.version < 2 {
            let v = read_uint(data, cursor, 2, content_end)? as u32;
            cursor += 2;
            v
        } else {
            let v = read_uint(data, cursor, 4, content_end)? as u32;
            cursor += 4;
            v
        };

        // construction_method (version >= 1)
        let construction_method = if full.version >= 1 {
            if cursor + 2 > content_end {
                return Err(IsobmffError::Truncated {
                    expected: 2,
                    available: content_end.saturating_sub(cursor),
                });
            }
            let cm = u16::from_be_bytes([data[cursor], data[cursor + 1]]) & 0x0F;
            cursor += 2;
            cm as u8
        } else {
            0
        };

        // data_reference_index
        if cursor + 2 > content_end {
            return Err(IsobmffError::Truncated {
                expected: 2,
                available: content_end.saturating_sub(cursor),
            });
        }
        let data_reference_index = u16::from_be_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;

        // base_offset
        let base_offset = read_uint(data, cursor, base_offset_size as usize, content_end)?;
        cursor += base_offset_size as usize;

        // extent_count
        if cursor + 2 > content_end {
            return Err(IsobmffError::Truncated {
                expected: 2,
                available: content_end.saturating_sub(cursor),
            });
        }
        let extent_count = u16::from_be_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;

        let mut extents = Vec::with_capacity(extent_count as usize);
        for _ in 0..extent_count {
            // extent_index (version >= 1, skip if present)
            if full.version >= 1 && index_size > 0 {
                cursor += index_size as usize;
            }

            let ext_offset = read_uint(data, cursor, offset_size as usize, content_end)?;
            cursor += offset_size as usize;

            let ext_length = read_uint(data, cursor, length_size as usize, content_end)?;
            cursor += length_size as usize;

            extents.push(Extent {
                offset: ext_offset,
                length: ext_length,
            });
        }

        locations.push(ItemLocation {
            item_id,
            construction_method,
            data_reference_index,
            base_offset,
            extents,
        });
    }

    Ok(locations)
}

/// Parse iref (item reference) box.
fn parse_iref(data: &[u8], offset: usize) -> Result<Vec<ItemReference>, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let content_start = full.box_header.content_offset;
    let content_end = match full.box_header.content_size {
        Some(s) => content_start + s as usize,
        None => data.len(),
    };

    let large_ids = full.version >= 1; // version 1 uses 32-bit item IDs
    let id_size: usize = if large_ids { 4 } else { 2 };

    let mut references = Vec::new();
    // iref contains a sequence of SingleItemTypeReference boxes
    for box_result in BoxIterator::new(data, content_start, content_end) {
        let header = box_result?;
        let ref_type = ReferenceType::from_fourcc(header.box_type);

        let pos = header.content_offset;
        let box_end = match header.content_size {
            Some(s) => pos + s as usize,
            None => content_end,
        };

        if box_end.saturating_sub(pos) < id_size + 2 {
            continue; // malformed, skip
        }

        let from_item_id = read_uint(data, pos, id_size, box_end)? as u32;
        let count_pos = pos + id_size;
        let ref_count = u16::from_be_bytes([data[count_pos], data[count_pos + 1]]);

        let mut to_item_ids = Vec::with_capacity(ref_count as usize);
        let mut cursor = count_pos + 2;
        for _ in 0..ref_count {
            let to_id = read_uint(data, cursor, id_size, box_end)? as u32;
            to_item_ids.push(to_id);
            cursor += id_size;
        }

        references.push(ItemReference {
            ref_type,
            from_item_id,
            to_item_ids,
        });
    }

    Ok(references)
}

/// Read an unsigned integer of `size` bytes (0, 2, 4, or 8) at `pos`.
fn read_uint(data: &[u8], pos: usize, size: usize, end: usize) -> Result<u64, IsobmffError> {
    if size == 0 {
        return Ok(0);
    }
    if pos + size > end {
        return Err(IsobmffError::Truncated {
            expected: size,
            available: end.saturating_sub(pos),
        });
    }
    match size {
        2 => Ok(u16::from_be_bytes([data[pos], data[pos + 1]]) as u64),
        4 => {
            Ok(u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64)
        }
        8 => Ok(u64::from_be_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ])),
        _ => Ok(0), // unsupported size, treat as zero
    }
}

/// Read a null-terminated UTF-8 string from data.
fn read_null_terminated_string(data: &[u8], start: usize, end: usize) -> String {
    let end = end.min(data.len());
    if start >= end {
        return String::new();
    }
    let slice = &data[start..end];
    let len = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..len]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ReferenceType;

    /// Helper to build a box with a given FourCC and content.
    fn make_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size = (8 + content.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(fourcc);
        buf.extend_from_slice(content);
        buf
    }

    /// Helper to build a full-box with version and flags.
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
    fn pitm_v0() {
        let data = make_full_box(b"pitm", 0, 0, &[0x00, 0x05]); // item_id = 5
        let id = parse_pitm(&data, 0).unwrap();
        assert_eq!(id, 5);
    }

    #[test]
    fn pitm_v1() {
        let data = make_full_box(b"pitm", 1, 0, &[0x00, 0x00, 0x00, 0x0A]); // item_id = 10
        let id = parse_pitm(&data, 0).unwrap();
        assert_eq!(id, 10);
    }

    #[test]
    fn iinf_with_two_infe_v2() {
        // Build 2 infe entries
        let mut infe1_content = Vec::new();
        infe1_content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
        infe1_content.extend_from_slice(&[0x00, 0x00]); // protection_index = 0
        infe1_content.extend_from_slice(b"hvc1"); // item_type
        infe1_content.push(0x00); // item_name (empty, null-terminated)
        let infe1 = make_full_box(b"infe", 2, 0, &infe1_content);

        let mut infe2_content = Vec::new();
        infe2_content.extend_from_slice(&[0x00, 0x02]); // item_id = 2
        infe2_content.extend_from_slice(&[0x00, 0x00]); // protection_index = 0
        infe2_content.extend_from_slice(b"Exif"); // item_type
        infe2_content.push(0x00); // item_name
        let infe2 = make_full_box(b"infe", 2, 0, &infe2_content);

        // iinf box: version=0, entry_count=2 (u16), then infe boxes
        let mut iinf_content = Vec::new();
        iinf_content.extend_from_slice(&[0x00, 0x02]); // entry_count = 2
        iinf_content.extend(&infe1);
        iinf_content.extend(&infe2);
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_content);

        let items = parse_iinf(&iinf, 0).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].item_id, 1);
        assert_eq!(&items[0].item_type, b"hvc1");
        assert_eq!(items[1].item_id, 2);
        assert_eq!(&items[1].item_type, b"Exif");
    }

    #[test]
    fn infe_v2_with_name() {
        let mut content = Vec::new();
        content.extend_from_slice(&[0x00, 0x03]); // item_id = 3
        content.extend_from_slice(&[0x00, 0x01]); // protection_index = 1
        content.extend_from_slice(b"jpeg"); // item_type
        content.extend_from_slice(b"thumb"); // item_name
        content.push(0x00); // null terminator
        let data = make_full_box(b"infe", 2, 0, &content);

        let info = parse_infe(&data, 0).unwrap();
        assert_eq!(info.item_id, 3);
        assert_eq!(info.protection_index, 1);
        assert_eq!(&info.item_type, b"jpeg");
        assert_eq!(info.item_name, "thumb");
    }

    #[test]
    fn iloc_v0_single_item() {
        // offset_size=4, length_size=4, base_offset_size=0, index_size=0
        let mut content = Vec::new();
        content.push(0x44); // offset_size=4, length_size=4
        content.push(0x00); // base_offset_size=0, index_size=0
        content.extend_from_slice(&[0x00, 0x01]); // item_count = 1

        // Item: id=1, data_ref_index=0, base_offset=0 (size 0), extent_count=1
        content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
        content.extend_from_slice(&[0x00, 0x00]); // data_reference_index = 0
        // base_offset: size 0, nothing to write
        content.extend_from_slice(&[0x00, 0x01]); // extent_count = 1
        content.extend_from_slice(&[0x00, 0x00, 0x01, 0x00]); // extent_offset = 256
        content.extend_from_slice(&[0x00, 0x00, 0x10, 0x00]); // extent_length = 4096

        let data = make_full_box(b"iloc", 0, 0, &content);
        let locs = parse_iloc(&data, 0).unwrap();
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].item_id, 1);
        assert_eq!(locs[0].construction_method, 0);
        assert_eq!(locs[0].extents.len(), 1);
        assert_eq!(locs[0].extents[0].offset, 256);
        assert_eq!(locs[0].extents[0].length, 4096);
    }

    #[test]
    fn iloc_v1_with_construction_method() {
        let mut content = Vec::new();
        content.push(0x44); // offset_size=4, length_size=4
        content.push(0x00); // base_offset_size=0, index_size=0
        content.extend_from_slice(&[0x00, 0x01]); // item_count = 1

        // Item with construction_method
        content.extend_from_slice(&[0x00, 0x02]); // item_id = 2
        content.extend_from_slice(&[0x00, 0x01]); // construction_method = 1 (idat)
        content.extend_from_slice(&[0x00, 0x00]); // data_reference_index = 0
        content.extend_from_slice(&[0x00, 0x01]); // extent_count = 1
        content.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // extent_offset = 0
        content.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]); // extent_length = 32

        let data = make_full_box(b"iloc", 1, 0, &content);
        let locs = parse_iloc(&data, 0).unwrap();
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].item_id, 2);
        assert_eq!(locs[0].construction_method, 1);
        assert_eq!(locs[0].extents[0].length, 32);
    }

    #[test]
    fn iloc_multiple_items_and_extents() {
        let mut content = Vec::new();
        content.push(0x44); // offset_size=4, length_size=4
        content.push(0x00); // base_offset_size=0
        content.extend_from_slice(&[0x00, 0x02]); // item_count = 2

        // Item 1: 2 extents
        content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
        content.extend_from_slice(&[0x00, 0x00]); // data_ref_index = 0
        content.extend_from_slice(&[0x00, 0x02]); // extent_count = 2
        content.extend_from_slice(&[0x00, 0x00, 0x01, 0x00]); // ext1 offset = 256
        content.extend_from_slice(&[0x00, 0x00, 0x08, 0x00]); // ext1 length = 2048
        content.extend_from_slice(&[0x00, 0x00, 0x09, 0x00]); // ext2 offset = 2304
        content.extend_from_slice(&[0x00, 0x00, 0x04, 0x00]); // ext2 length = 1024

        // Item 2: 1 extent
        content.extend_from_slice(&[0x00, 0x02]); // item_id = 2
        content.extend_from_slice(&[0x00, 0x00]); // data_ref_index = 0
        content.extend_from_slice(&[0x00, 0x01]); // extent_count = 1
        content.extend_from_slice(&[0x00, 0x00, 0x10, 0x00]); // offset = 4096
        content.extend_from_slice(&[0x00, 0x00, 0x00, 0x80]); // length = 128

        let data = make_full_box(b"iloc", 0, 0, &content);
        let locs = parse_iloc(&data, 0).unwrap();
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0].extents.len(), 2);
        assert_eq!(locs[0].extents[1].offset, 2304);
        assert_eq!(locs[1].item_id, 2);
        assert_eq!(locs[1].extents[0].length, 128);
    }

    #[test]
    fn iref_v0_dimg_and_thmb() {
        // Build iref with two reference groups: dimg and thmb
        let mut dimg_content = Vec::new();
        dimg_content.extend_from_slice(&[0x00, 0x01]); // from_item_id = 1
        dimg_content.extend_from_slice(&[0x00, 0x03]); // reference_count = 3
        dimg_content.extend_from_slice(&[0x00, 0x02]); // to_id = 2
        dimg_content.extend_from_slice(&[0x00, 0x03]); // to_id = 3
        dimg_content.extend_from_slice(&[0x00, 0x04]); // to_id = 4
        let dimg = make_box(b"dimg", &dimg_content);

        let mut thmb_content = Vec::new();
        thmb_content.extend_from_slice(&[0x00, 0x05]); // from_item_id = 5
        thmb_content.extend_from_slice(&[0x00, 0x01]); // reference_count = 1
        thmb_content.extend_from_slice(&[0x00, 0x01]); // to_id = 1
        let thmb = make_box(b"thmb", &thmb_content);

        let mut iref_content = Vec::new();
        iref_content.extend(&dimg);
        iref_content.extend(&thmb);
        let iref = make_full_box(b"iref", 0, 0, &iref_content);

        let refs = parse_iref(&iref, 0).unwrap();
        assert_eq!(refs.len(), 2);

        assert_eq!(refs[0].ref_type, ReferenceType::Dimg);
        assert_eq!(refs[0].from_item_id, 1);
        assert_eq!(refs[0].to_item_ids, vec![2, 3, 4]);

        assert_eq!(refs[1].ref_type, ReferenceType::Thmb);
        assert_eq!(refs[1].from_item_id, 5);
        assert_eq!(refs[1].to_item_ids, vec![1]);
    }

    #[test]
    fn iref_auxl_reference() {
        let mut auxl_content = Vec::new();
        auxl_content.extend_from_slice(&[0x00, 0x0A]); // from_item_id = 10
        auxl_content.extend_from_slice(&[0x00, 0x01]); // ref_count = 1
        auxl_content.extend_from_slice(&[0x00, 0x01]); // to_id = 1
        let auxl = make_box(b"auxl", &auxl_content);

        let iref = make_full_box(b"iref", 0, 0, &auxl);
        let refs = parse_iref(&iref, 0).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].ref_type, ReferenceType::Auxl);
        assert_eq!(refs[0].from_item_id, 10);
    }

    #[test]
    fn full_meta_box() {
        // Build a complete meta box with pitm, iinf (1 item), iloc (1 item), iref (1 ref)
        let pitm = make_full_box(b"pitm", 0, 0, &[0x00, 0x01]); // primary = 1

        let mut infe_content = Vec::new();
        infe_content.extend_from_slice(&[0x00, 0x01]); // item_id = 1
        infe_content.extend_from_slice(&[0x00, 0x00]); // protection = 0
        infe_content.extend_from_slice(b"hvc1");
        infe_content.push(0x00); // name
        let infe = make_full_box(b"infe", 2, 0, &infe_content);
        let mut iinf_content = Vec::new();
        iinf_content.extend_from_slice(&[0x00, 0x01]); // count = 1
        iinf_content.extend(&infe);
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_content);

        let mut iloc_content = Vec::new();
        iloc_content.push(0x44);
        iloc_content.push(0x00);
        iloc_content.extend_from_slice(&[0x00, 0x01]); // 1 item
        iloc_content.extend_from_slice(&[0x00, 0x01]); // item_id=1
        iloc_content.extend_from_slice(&[0x00, 0x00]); // data_ref=0
        iloc_content.extend_from_slice(&[0x00, 0x01]); // 1 extent
        iloc_content.extend_from_slice(&[0x00, 0x00, 0x10, 0x00]); // offset=4096
        iloc_content.extend_from_slice(&[0x00, 0x00, 0x08, 0x00]); // length=2048
        let iloc = make_full_box(b"iloc", 0, 0, &iloc_content);

        let mut thmb_c = Vec::new();
        thmb_c.extend_from_slice(&[0x00, 0x02]); // from=2
        thmb_c.extend_from_slice(&[0x00, 0x01]); // count=1
        thmb_c.extend_from_slice(&[0x00, 0x01]); // to=1
        let thmb = make_box(b"thmb", &thmb_c);
        let iref = make_full_box(b"iref", 0, 0, &thmb);

        let mut meta_content = Vec::new();
        meta_content.extend(&pitm);
        meta_content.extend(&iinf);
        meta_content.extend(&iloc);
        meta_content.extend(&iref);
        let meta = make_full_box(b"meta", 0, 0, &meta_content);

        let result = parse_meta(&meta, 0).unwrap();
        assert_eq!(result.primary_item_id, 1);
        assert_eq!(result.items.len(), 1);
        assert_eq!(&result.items[0].item_type, b"hvc1");
        assert_eq!(result.locations.len(), 1);
        assert_eq!(result.locations[0].extents[0].offset, 4096);
        assert_eq!(result.locations[0].extents[0].length, 2048);
        assert_eq!(result.references.len(), 1);
        assert_eq!(result.references[0].ref_type, ReferenceType::Thmb);
    }
}
