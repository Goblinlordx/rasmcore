//! Generic ISOBMFF box reader.
//!
//! Reads box headers (type + size) and provides iteration over sibling boxes.

use crate::error::IsobmffError;
use crate::types::{BoxHeader, FullBoxHeader};

/// Read a box header from the given data at `offset`.
///
/// Returns the parsed header. The caller can then slice into the data
/// at `header.content_offset` for `header.content_size` bytes to get
/// the box body.
pub fn read_box_header(data: &[u8], offset: usize) -> Result<BoxHeader, IsobmffError> {
    let remaining = data.len().saturating_sub(offset);
    if remaining < 8 {
        return Err(IsobmffError::Truncated {
            expected: 8,
            available: remaining,
        });
    }

    let size_bytes = &data[offset..offset + 4];
    let size32 = u32::from_be_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]]);

    let mut box_type = [0u8; 4];
    box_type.copy_from_slice(&data[offset + 4..offset + 8]);

    let (box_size, header_size): (Option<u64>, u8) = match size32 {
        // size == 0 means box extends to end of data
        0 => (None, 8),
        // size == 1 means extended 64-bit size follows
        1 => {
            if remaining < 16 {
                return Err(IsobmffError::Truncated {
                    expected: 16,
                    available: remaining,
                });
            }
            let ext = &data[offset + 8..offset + 16];
            let size64 = u64::from_be_bytes([
                ext[0], ext[1], ext[2], ext[3], ext[4], ext[5], ext[6], ext[7],
            ]);
            if size64 < 16 {
                return Err(IsobmffError::InvalidBoxSize {
                    box_type,
                    size: size64,
                });
            }
            (Some(size64), 16)
        }
        // Normal 32-bit size
        s => {
            if s < 8 {
                return Err(IsobmffError::InvalidBoxSize {
                    box_type,
                    size: s as u64,
                });
            }
            (Some(s as u64), 8)
        }
    };

    let content_offset = offset + header_size as usize;
    let content_size = box_size.map(|s| s - header_size as u64);

    Ok(BoxHeader {
        box_type,
        box_size,
        content_offset,
        content_size,
        header_size,
    })
}

/// Read a full-box header (box header + version + flags).
///
/// Full boxes start with a 1-byte version and 3-byte flags field
/// after the standard box header.
pub fn read_full_box_header(data: &[u8], offset: usize) -> Result<FullBoxHeader, IsobmffError> {
    let box_header = read_box_header(data, offset)?;

    let vf_offset = box_header.content_offset;
    let remaining = data.len().saturating_sub(vf_offset);
    if remaining < 4 {
        return Err(IsobmffError::Truncated {
            expected: 4,
            available: remaining,
        });
    }

    let version = data[vf_offset];
    let flags = u32::from_be_bytes([
        0,
        data[vf_offset + 1],
        data[vf_offset + 2],
        data[vf_offset + 3],
    ]);

    // Adjust the box header to reflect that content starts after version+flags
    let adjusted = BoxHeader {
        content_offset: vf_offset + 4,
        content_size: box_header.content_size.map(|s| s.saturating_sub(4)),
        ..box_header
    };

    Ok(FullBoxHeader {
        box_header: adjusted,
        version,
        flags,
    })
}

/// Iterator over sibling boxes within a data slice.
///
/// Walks consecutive boxes from `start` to `end` within `data`.
pub struct BoxIterator<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> BoxIterator<'a> {
    /// Create an iterator over boxes in the given data range.
    pub fn new(data: &'a [u8], start: usize, end: usize) -> Self {
        Self {
            data,
            pos: start,
            end: end.min(data.len()),
        }
    }

    /// Create an iterator over all top-level boxes in the data.
    pub fn top_level(data: &'a [u8]) -> Self {
        Self::new(data, 0, data.len())
    }
}

impl<'a> Iterator for BoxIterator<'a> {
    type Item = Result<BoxHeader, IsobmffError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }

        match read_box_header(self.data, self.pos) {
            Ok(header) => {
                // Advance past this box
                match header.box_size {
                    Some(size) => {
                        self.pos += size as usize;
                    }
                    None => {
                        // Box extends to end — this is the last box
                        self.pos = self.end;
                    }
                }
                Some(Ok(header))
            }
            Err(e) => {
                // Stop iteration on error
                self.pos = self.end;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size = (8 + content.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(fourcc);
        buf.extend_from_slice(content);
        buf
    }

    fn make_extended_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size64 = (16 + content.len()) as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_be_bytes()); // size32 = 1 => extended
        buf.extend_from_slice(fourcc);
        buf.extend_from_slice(&size64.to_be_bytes());
        buf.extend_from_slice(content);
        buf
    }

    #[test]
    fn normal_box() {
        let data = make_box(b"test", &[0xAA; 10]);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_type, *b"test");
        assert_eq!(hdr.box_size, Some(18));
        assert_eq!(hdr.header_size, 8);
        assert_eq!(hdr.content_offset, 8);
        assert_eq!(hdr.content_size, Some(10));
    }

    #[test]
    fn extended_size_box() {
        let data = make_extended_box(b"big!", &[0xBB; 20]);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_type, *b"big!");
        assert_eq!(hdr.box_size, Some(36));
        assert_eq!(hdr.header_size, 16);
        assert_eq!(hdr.content_offset, 16);
        assert_eq!(hdr.content_size, Some(20));
    }

    #[test]
    fn extends_to_eof() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_be_bytes()); // size = 0
        data.extend_from_slice(b"last");
        data.extend_from_slice(&[0xCC; 5]);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_type, *b"last");
        assert_eq!(hdr.box_size, None);
        assert_eq!(hdr.header_size, 8);
        assert_eq!(hdr.content_size, None);
    }

    #[test]
    fn truncated_header() {
        let data = [0u8; 5];
        let err = read_box_header(&data, 0).unwrap_err();
        assert!(matches!(
            err,
            IsobmffError::Truncated {
                expected: 8,
                available: 5
            }
        ));
    }

    #[test]
    fn truncated_extended() {
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(b"ext!");
        data.extend_from_slice(&[0; 4]); // only 4 of needed 8 extended bytes
        let err = read_box_header(&data, 0).unwrap_err();
        assert!(matches!(err, IsobmffError::Truncated { expected: 16, .. }));
    }

    #[test]
    fn invalid_size_too_small() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_be_bytes()); // size < 8 is invalid
        data.extend_from_slice(b"bad!");
        let err = read_box_header(&data, 0).unwrap_err();
        assert!(matches!(err, IsobmffError::InvalidBoxSize { .. }));
    }

    #[test]
    fn full_box_header() {
        let mut data = make_box(b"meta", &[0; 20]);
        // Set version=1, flags=0x000001 at content start (offset 8)
        data[8] = 1; // version
        data[9] = 0;
        data[10] = 0;
        data[11] = 1; // flags = 1

        let fbh = read_full_box_header(&data, 0).unwrap();
        assert_eq!(fbh.version, 1);
        assert_eq!(fbh.flags, 1);
        assert_eq!(fbh.box_header.content_offset, 12); // 8 (box) + 4 (ver+flags)
        assert_eq!(fbh.box_header.content_size, Some(16)); // 20 - 4
    }

    #[test]
    fn iterator_walks_sibling_boxes() {
        let mut data = Vec::new();
        data.extend(make_box(b"aaaa", &[1, 2, 3]));
        data.extend(make_box(b"bbbb", &[4, 5]));
        data.extend(make_box(b"cccc", &[6]));

        let boxes: Vec<BoxHeader> = BoxIterator::top_level(&data)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(boxes.len(), 3);
        assert_eq!(boxes[0].box_type, *b"aaaa");
        assert_eq!(boxes[1].box_type, *b"bbbb");
        assert_eq!(boxes[2].box_type, *b"cccc");
    }

    #[test]
    fn iterator_handles_eof_box() {
        let mut data = Vec::new();
        data.extend(make_box(b"aaaa", &[1]));
        // Last box extends to EOF
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"last");
        data.extend_from_slice(&[9; 10]);

        let boxes: Vec<BoxHeader> = BoxIterator::top_level(&data)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].box_type, *b"aaaa");
        assert_eq!(boxes[1].box_type, *b"last");
        assert_eq!(boxes[1].box_size, None);
    }

    #[test]
    fn nested_container_iteration() {
        // Create a container box with child boxes inside
        let child1 = make_box(b"ch_1", &[0xAA]);
        let child2 = make_box(b"ch_2", &[0xBB, 0xCC]);
        let mut children = Vec::new();
        children.extend(&child1);
        children.extend(&child2);
        let container = make_box(b"cont", &children);

        // Parse the container header
        let hdr = read_box_header(&container, 0).unwrap();
        assert_eq!(hdr.box_type, *b"cont");

        // Iterate children within the container's content range
        let content_end = hdr.content_offset + hdr.content_size.unwrap() as usize;
        let kids: Vec<BoxHeader> = BoxIterator::new(&container, hdr.content_offset, content_end)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].box_type, *b"ch_1");
        assert_eq!(kids[1].box_type, *b"ch_2");
    }
}
