//! Minimal TIFF/IFD container parser for DNG RAW files.
//!
//! Parses the TIFF header, navigates IFD chains, and reads tag values.
//! Handles both little-endian (II) and big-endian (MM) byte orders.
//! Only implements the subset needed for DNG decode — not a general TIFF decoder.

use crate::RawError;

/// TIFF byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    pub fn u16(self, buf: &[u8]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes([buf[0], buf[1]]),
            Self::Big => u16::from_be_bytes([buf[0], buf[1]]),
        }
    }

    pub fn u32(self, buf: &[u8]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            Self::Big => u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
        }
    }

    pub fn i32(self, buf: &[u8]) -> i32 {
        match self {
            Self::Little => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            Self::Big => i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
        }
    }
}

/// TIFF IFD tag data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum TagType {
    Byte = 1,
    Ascii = 2,
    Short = 3,
    Long = 4,
    Rational = 5,
    SByte = 6,
    Undefined = 7,
    SShort = 8,
    SLong = 9,
    SRational = 10,
    Float = 11,
    Double = 12,
}

impl TagType {
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            1 => Some(Self::Byte),
            2 => Some(Self::Ascii),
            3 => Some(Self::Short),
            4 => Some(Self::Long),
            5 => Some(Self::Rational),
            6 => Some(Self::SByte),
            7 => Some(Self::Undefined),
            8 => Some(Self::SShort),
            9 => Some(Self::SLong),
            10 => Some(Self::SRational),
            11 => Some(Self::Float),
            12 => Some(Self::Double),
            _ => None,
        }
    }

    /// Size in bytes of one element of this type.
    pub fn element_size(self) -> usize {
        match self {
            Self::Byte | Self::SByte | Self::Ascii | Self::Undefined => 1,
            Self::Short | Self::SShort => 2,
            Self::Long | Self::SLong | Self::Float => 4,
            Self::Rational | Self::SRational | Self::Double => 8,
        }
    }
}

/// A raw IFD entry (tag + value location).
#[derive(Debug, Clone)]
pub struct IfdEntry {
    pub tag: u16,
    pub tag_type: TagType,
    pub count: u32,
    /// The 4-byte value/offset field (raw bytes, interpreted based on type/count).
    pub value_offset_raw: [u8; 4],
}

/// Parsed TIFF container with IFD access.
pub struct TiffContainer<'a> {
    pub data: &'a [u8],
    pub order: ByteOrder,
}

impl<'a> TiffContainer<'a> {
    /// Parse TIFF header and return container.
    pub fn parse(data: &'a [u8]) -> Result<Self, RawError> {
        if data.len() < 8 {
            return Err(RawError::InvalidFormat("TIFF header too short".into()));
        }
        let order = match &data[0..2] {
            b"II" => ByteOrder::Little,
            b"MM" => ByteOrder::Big,
            _ => return Err(RawError::InvalidFormat("invalid TIFF byte order".into())),
        };
        let magic = order.u16(&data[2..4]);
        if magic != 42 {
            return Err(RawError::InvalidFormat(format!(
                "invalid TIFF magic: {magic}"
            )));
        }
        Ok(Self { data, order })
    }

    /// Offset to the first IFD.
    pub fn first_ifd_offset(&self) -> u32 {
        self.order.u32(&self.data[4..8])
    }

    /// Read all entries in the IFD at the given offset.
    pub fn read_ifd(&self, offset: u32) -> Result<Vec<IfdEntry>, RawError> {
        let off = offset as usize;
        if off + 2 > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        let count = self.order.u16(&self.data[off..off + 2]) as usize;
        let entries_start = off + 2;
        let entries_end = entries_start + count * 12;
        if entries_end > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let base = entries_start + i * 12;
            let tag = self.order.u16(&self.data[base..base + 2]);
            let type_val = self.order.u16(&self.data[base + 2..base + 4]);
            let cnt = self.order.u32(&self.data[base + 4..base + 8]);
            let mut vo = [0u8; 4];
            vo.copy_from_slice(&self.data[base + 8..base + 12]);
            let tag_type = TagType::from_u16(type_val).ok_or_else(|| {
                RawError::InvalidFormat(format!("unknown IFD type {type_val} for tag {tag}"))
            })?;
            entries.push(IfdEntry {
                tag,
                tag_type,
                count: cnt,
                value_offset_raw: vo,
            });
        }
        Ok(entries)
    }

    /// Offset to the next IFD (0 if none).
    pub fn next_ifd_offset(&self, current_ifd_offset: u32) -> Result<u32, RawError> {
        let off = current_ifd_offset as usize;
        if off + 2 > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        let count = self.order.u16(&self.data[off..off + 2]) as usize;
        let next_off_pos = off + 2 + count * 12;
        if next_off_pos + 4 > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        Ok(self.order.u32(&self.data[next_off_pos..next_off_pos + 4]))
    }

    /// Find an entry by tag ID in a list of entries.
    pub fn find_tag(entries: &[IfdEntry], tag: u16) -> Option<&IfdEntry> {
        entries.iter().find(|e| e.tag == tag)
    }

    /// Read the value(s) of a tag as bytes. If the total size fits in 4 bytes,
    /// the value is inline; otherwise it's an offset into the file.
    pub fn tag_data<'b>(&self, entry: &'b IfdEntry) -> Result<&'b [u8], RawError>
    where
        'a: 'b,
    {
        let total = entry.tag_type.element_size() * entry.count as usize;
        if total <= 4 {
            Ok(&entry.value_offset_raw[..total])
        } else {
            let offset = self.order.u32(&entry.value_offset_raw) as usize;
            if offset + total > self.data.len() {
                return Err(RawError::DataTruncated);
            }
            Ok(&self.data[offset..offset + total])
        }
    }

    /// Read tag as a single u16.
    pub fn tag_u16(&self, entry: &IfdEntry) -> Result<u16, RawError> {
        if entry.count != 1 {
            return Err(RawError::InvalidFormat(format!(
                "expected 1 value for tag {}, got {}",
                entry.tag, entry.count
            )));
        }
        match entry.tag_type {
            TagType::Short => Ok(self.order.u16(&entry.value_offset_raw)),
            TagType::Byte => Ok(entry.value_offset_raw[0] as u16),
            _ => Err(RawError::InvalidFormat(format!(
                "tag {} not SHORT/BYTE",
                entry.tag
            ))),
        }
    }

    /// Read tag as a single u32.
    pub fn tag_u32(&self, entry: &IfdEntry) -> Result<u32, RawError> {
        if entry.count != 1 {
            return Err(RawError::InvalidFormat(format!(
                "expected 1 value for tag {}, got {}",
                entry.tag, entry.count
            )));
        }
        match entry.tag_type {
            TagType::Long => Ok(self.order.u32(&entry.value_offset_raw)),
            TagType::Short => Ok(self.order.u16(&entry.value_offset_raw) as u32),
            _ => Err(RawError::InvalidFormat(format!(
                "tag {} not LONG/SHORT",
                entry.tag
            ))),
        }
    }

    /// Read tag as a vector of u32 values.
    pub fn tag_u32_vec(&self, entry: &IfdEntry) -> Result<Vec<u32>, RawError> {
        let data = self.tag_data(entry)?;
        match entry.tag_type {
            TagType::Long => {
                let mut vals = Vec::with_capacity(entry.count as usize);
                for i in 0..entry.count as usize {
                    vals.push(self.order.u32(&data[i * 4..(i + 1) * 4]));
                }
                Ok(vals)
            }
            TagType::Short => {
                let mut vals = Vec::with_capacity(entry.count as usize);
                for i in 0..entry.count as usize {
                    vals.push(self.order.u16(&data[i * 2..(i + 1) * 2]) as u32);
                }
                Ok(vals)
            }
            _ => Err(RawError::InvalidFormat(format!(
                "tag {} not LONG/SHORT array",
                entry.tag
            ))),
        }
    }

    /// Read tag as a vector of RATIONAL values (pairs of u32: numerator/denominator) -> f64.
    pub fn tag_rational_vec(&self, entry: &IfdEntry) -> Result<Vec<f64>, RawError> {
        let count = entry.count as usize;
        let offset = self.order.u32(&entry.value_offset_raw) as usize;
        let total = count * 8;
        if offset + total > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        let data = &self.data[offset..offset + total];
        let mut vals = Vec::with_capacity(count);
        for i in 0..count {
            let num = self.order.u32(&data[i * 8..i * 8 + 4]);
            let den = self.order.u32(&data[i * 8 + 4..i * 8 + 8]);
            if den == 0 {
                vals.push(0.0);
            } else {
                vals.push(num as f64 / den as f64);
            }
        }
        Ok(vals)
    }

    /// Read tag as a vector of SRATIONAL values (pairs of i32) -> f64.
    pub fn tag_srational_vec(&self, entry: &IfdEntry) -> Result<Vec<f64>, RawError> {
        let count = entry.count as usize;
        let offset = self.order.u32(&entry.value_offset_raw) as usize;
        let total = count * 8;
        if offset + total > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        let data = &self.data[offset..offset + total];
        let mut vals = Vec::with_capacity(count);
        for i in 0..count {
            let num = self.order.i32(&data[i * 8..i * 8 + 4]);
            let den = self.order.i32(&data[i * 8 + 4..i * 8 + 8]);
            if den == 0 {
                vals.push(0.0);
            } else {
                vals.push(num as f64 / den as f64);
            }
        }
        Ok(vals)
    }

    /// Read a slice of raw data from the file at the given offset and length.
    pub fn raw_data(&self, offset: u32, length: u32) -> Result<&'a [u8], RawError> {
        let off = offset as usize;
        let len = length as usize;
        if off + len > self.data.len() {
            return Err(RawError::DataTruncated);
        }
        Ok(&self.data[off..off + len])
    }
}

// ─── Well-known TIFF tag IDs ─────────────────────────────────────────────────

pub const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub const TAG_IMAGE_WIDTH: u16 = 256;
pub const TAG_IMAGE_LENGTH: u16 = 257;
pub const TAG_BITS_PER_SAMPLE: u16 = 258;
pub const TAG_COMPRESSION: u16 = 259;
pub const _TAG_PHOTOMETRIC: u16 = 262;
pub const TAG_STRIP_OFFSETS: u16 = 273;
pub const _TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub const TAG_ROWS_PER_STRIP: u16 = 278;
pub const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub const TAG_TILE_WIDTH: u16 = 322;
pub const TAG_TILE_LENGTH: u16 = 323;
pub const TAG_TILE_OFFSETS: u16 = 324;
pub const TAG_TILE_BYTE_COUNTS: u16 = 325;
pub const TAG_SUB_IFDS: u16 = 330;

// DNG-specific tags
pub const _TAG_CFA_REPEAT_PATTERN_DIM: u16 = 33421;
pub const TAG_CFA_PATTERN: u16 = 33422;
pub const TAG_DNG_VERSION: u16 = 50706;
pub const TAG_COLOR_MATRIX_1: u16 = 50721;
pub const _TAG_COLOR_MATRIX_2: u16 = 50722;
pub const TAG_AS_SHOT_NEUTRAL: u16 = 50727;
pub const TAG_ACTIVE_AREA: u16 = 50829;
pub const TAG_DEFAULT_CROP_ORIGIN: u16 = 50719;
pub const TAG_DEFAULT_CROP_SIZE: u16 = 50720;
pub const TAG_BLACK_LEVEL: u16 = 50714;
pub const TAG_WHITE_LEVEL: u16 = 50717;

/// Compression values
pub const COMPRESSION_NONE: u16 = 1;
pub const COMPRESSION_LOSSLESS_JPEG: u16 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiff_header(order: ByteOrder, ifd_offset: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        match order {
            ByteOrder::Little => {
                buf.extend_from_slice(b"II");
                buf.extend_from_slice(&42u16.to_le_bytes());
                buf.extend_from_slice(&ifd_offset.to_le_bytes());
            }
            ByteOrder::Big => {
                buf.extend_from_slice(b"MM");
                buf.extend_from_slice(&42u16.to_be_bytes());
                buf.extend_from_slice(&ifd_offset.to_be_bytes());
            }
        }
        buf
    }

    #[test]
    fn parse_little_endian_header() {
        let data = make_tiff_header(ByteOrder::Little, 8);
        let c = TiffContainer::parse(&data).unwrap();
        assert_eq!(c.order, ByteOrder::Little);
        assert_eq!(c.first_ifd_offset(), 8);
    }

    #[test]
    fn parse_big_endian_header() {
        let data = make_tiff_header(ByteOrder::Big, 8);
        let c = TiffContainer::parse(&data).unwrap();
        assert_eq!(c.order, ByteOrder::Big);
        assert_eq!(c.first_ifd_offset(), 8);
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = make_tiff_header(ByteOrder::Little, 8);
        data[2] = 99; // corrupt magic
        assert!(TiffContainer::parse(&data).is_err());
    }

    #[test]
    fn read_ifd_single_entry() {
        // Build: header(8 bytes) + IFD at offset 8: count=1, one entry, next=0
        let mut data = make_tiff_header(ByteOrder::Little, 8);
        // IFD: count = 1
        data.extend_from_slice(&1u16.to_le_bytes());
        // Entry: tag=256(ImageWidth), type=SHORT(3), count=1, value=640
        data.extend_from_slice(&256u16.to_le_bytes()); // tag
        data.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
        data.extend_from_slice(&1u32.to_le_bytes()); // count
        data.extend_from_slice(&640u16.to_le_bytes()); // value
        data.extend_from_slice(&0u16.to_le_bytes()); // padding
        // next IFD offset
        data.extend_from_slice(&0u32.to_le_bytes());

        let c = TiffContainer::parse(&data).unwrap();
        let entries = c.read_ifd(8).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tag, TAG_IMAGE_WIDTH);
        assert_eq!(c.tag_u16(&entries[0]).unwrap(), 640);
    }
}
