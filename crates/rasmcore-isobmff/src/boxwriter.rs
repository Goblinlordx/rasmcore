//! Generic ISOBMFF box writer — counterpart to boxreader.rs.
//!
//! Writes box headers (type + size) and provides helpers for building
//! nested box structures into a `Vec<u8>` buffer.

/// Write a standard box: 4-byte size + 4-byte type + content.
///
/// Returns the complete box bytes. Size includes the 8-byte header.
pub fn write_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = (8 + content.len()) as u32;
    let mut buf = Vec::with_capacity(size as usize);
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(fourcc);
    buf.extend_from_slice(content);
    buf
}

/// Write a full box: 4-byte size + 4-byte type + 1-byte version + 3-byte flags + content.
///
/// Returns the complete box bytes. Size includes the 12-byte header.
pub fn write_full_box(fourcc: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let size = (12 + content.len()) as u32;
    let mut buf = Vec::with_capacity(size as usize);
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(fourcc);
    buf.push(version);
    let fb = flags.to_be_bytes();
    buf.extend_from_slice(&fb[1..4]); // 3-byte flags
    buf.extend_from_slice(content);
    buf
}

/// Write a box that extends to end of file (size = 0).
/// Used for mdat as the last box — avoids needing to know the total size upfront.
pub fn write_box_to_eof(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + content.len());
    buf.extend_from_slice(&0u32.to_be_bytes()); // size = 0 → extends to EOF
    buf.extend_from_slice(fourcc);
    buf.extend_from_slice(content);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxreader::read_box_header;

    #[test]
    fn write_then_read_box() {
        let data = write_box(b"test", &[0xAA, 0xBB, 0xCC]);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_type, *b"test");
        assert_eq!(hdr.box_size, Some(11));
        assert_eq!(hdr.header_size, 8);
        assert_eq!(hdr.content_offset, 8);
        assert_eq!(hdr.content_size, Some(3));
        assert_eq!(&data[8..], &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn write_then_read_full_box() {
        let data = write_full_box(b"meta", 1, 0x000001, &[0xDD; 5]);
        let hdr = crate::boxreader::read_full_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_header.box_type, *b"meta");
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.flags, 1);
        assert_eq!(hdr.box_header.content_offset, 12);
        assert_eq!(&data[12..], &[0xDD; 5]);
    }

    #[test]
    fn write_eof_box() {
        let data = write_box_to_eof(b"mdat", &[1, 2, 3]);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.box_type, *b"mdat");
        assert_eq!(hdr.box_size, None); // extends to EOF
        assert_eq!(&data[8..], &[1, 2, 3]);
    }

    #[test]
    fn empty_content() {
        let data = write_box(b"free", &[]);
        assert_eq!(data.len(), 8);
        let hdr = read_box_header(&data, 0).unwrap();
        assert_eq!(hdr.content_size, Some(0));
    }
}
