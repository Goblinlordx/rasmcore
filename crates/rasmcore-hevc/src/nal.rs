//! NAL unit parser — header parsing, RBSP extraction, start code splitting.
//!
//! ITU-T H.265 Section 7.3.1 (NAL unit syntax) and B.2 (byte stream format).

use crate::error::HevcError;
use crate::types::{NalUnit, NalUnitType};

/// Parse a NAL unit header from 2 bytes.
///
/// Format: forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)
pub fn parse_nal_header(data: &[u8]) -> Result<(NalUnitType, u8, u8), HevcError> {
    if data.len() < 2 {
        return Err(HevcError::InvalidNal("NAL header requires 2 bytes".into()));
    }

    let forbidden = (data[0] >> 7) & 1;
    if forbidden != 0 {
        return Err(HevcError::InvalidNal(
            "forbidden_zero_bit is not zero".into(),
        ));
    }

    let nal_type = NalUnitType::from_u8((data[0] >> 1) & 0x3F);
    let nuh_layer_id = ((data[0] & 1) << 5) | ((data[1] >> 3) & 0x1F);
    let nuh_temporal_id_plus1 = data[1] & 0x07;

    if nuh_temporal_id_plus1 == 0 {
        return Err(HevcError::InvalidNal(
            "nuh_temporal_id_plus1 must not be 0".into(),
        ));
    }

    Ok((nal_type, nuh_layer_id, nuh_temporal_id_plus1))
}

/// Parse a complete NAL unit from raw data (header + payload).
///
/// Parses the 2-byte header and extracts RBSP from the remaining bytes.
pub fn parse_nal_unit(data: &[u8]) -> Result<NalUnit, HevcError> {
    let (nal_type, nuh_layer_id, nuh_temporal_id_plus1) = parse_nal_header(data)?;
    let rbsp = extract_rbsp(&data[2..]);

    Ok(NalUnit {
        nal_type,
        nuh_layer_id,
        nuh_temporal_id_plus1,
        rbsp,
    })
}

/// Extract RBSP (Raw Byte Sequence Payload) by removing emulation prevention bytes.
///
/// HEVC uses 0x000003 as emulation prevention — this function removes the 0x03 byte
/// when preceded by 0x0000, restoring the original RBSP data.
pub fn extract_rbsp(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        // Check for emulation prevention: 0x00 0x00 0x03
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            rbsp.push(0x00);
            rbsp.push(0x00);
            i += 3; // Skip the 0x03 prevention byte
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }

    rbsp
}

/// Iterator over NAL units in an Annex B byte stream.
///
/// Splits on start codes: 0x000001 (3-byte) or 0x00000001 (4-byte).
pub struct NalIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> NalIterator<'a> {
    /// Create a new iterator over NAL units in the given byte stream.
    pub fn new(data: &'a [u8]) -> Self {
        let mut iter = Self { data, pos: 0 };
        // Skip to first start code
        iter.skip_to_start_code();
        iter
    }

    fn skip_to_start_code(&mut self) {
        while self.pos < self.data.len() {
            if self.is_start_code() {
                self.skip_start_code();
                return;
            }
            self.pos += 1;
        }
    }

    fn is_start_code(&self) -> bool {
        let remaining = self.data.len() - self.pos;
        if remaining >= 3
            && self.data[self.pos] == 0
            && self.data[self.pos + 1] == 0
            && self.data[self.pos + 2] == 1
        {
            return true;
        }
        if remaining >= 4
            && self.data[self.pos] == 0
            && self.data[self.pos + 1] == 0
            && self.data[self.pos + 2] == 0
            && self.data[self.pos + 3] == 1
        {
            return true;
        }
        false
    }

    fn skip_start_code(&mut self) {
        if self.pos + 4 <= self.data.len()
            && self.data[self.pos] == 0
            && self.data[self.pos + 1] == 0
            && self.data[self.pos + 2] == 0
            && self.data[self.pos + 3] == 1
        {
            self.pos += 4;
        } else if self.pos + 3 <= self.data.len()
            && self.data[self.pos] == 0
            && self.data[self.pos + 1] == 0
            && self.data[self.pos + 2] == 1
        {
            self.pos += 3;
        }
    }

    fn find_next_start_code(&self) -> usize {
        let mut i = self.pos;
        while i + 2 < self.data.len() {
            if self.data[i] == 0 && self.data[i + 1] == 0 {
                if self.data[i + 2] == 1 {
                    return i;
                }
                if i + 3 < self.data.len() && self.data[i + 2] == 0 && self.data[i + 3] == 1 {
                    return i;
                }
            }
            i += 1;
        }
        self.data.len()
    }
}

impl<'a> Iterator for NalIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let start = self.pos;
        let end = self.find_next_start_code();

        if start == end {
            return None;
        }

        self.pos = end;
        self.skip_to_start_code();

        // Trim trailing zero bytes that are part of the next start code
        let mut nal_end = end;
        while nal_end > start && self.data[nal_end - 1] == 0 {
            nal_end -= 1;
        }

        if nal_end > start {
            Some(&self.data[start..nal_end])
        } else {
            self.next()
        }
    }
}

/// Parse NAL units from an hvcC configuration record's NAL arrays.
///
/// The hvcC config (from ISOBMFF) contains pre-parsed NAL units without
/// start codes or emulation prevention bytes. This function converts them
/// to our NalUnit type.
pub fn parse_hvcc_nals(
    nal_arrays: &[rasmcore_isobmff::NalArray],
) -> Result<Vec<NalUnit>, HevcError> {
    let mut units = Vec::new();
    for array in nal_arrays {
        for nal_data in &array.nal_units {
            if nal_data.len() < 2 {
                continue;
            }
            let (nal_type, nuh_layer_id, nuh_temporal_id_plus1) = parse_nal_header(nal_data)?;
            units.push(NalUnit {
                nal_type,
                nuh_layer_id,
                nuh_temporal_id_plus1,
                rbsp: nal_data[2..].to_vec(), // hvcC NALs don't have emulation prevention
            });
        }
    }
    Ok(units)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vps_header() {
        // VPS NAL: type=32, layer_id=0, temporal_id=1
        // forbidden(0) | type=32=100000 | layer_id=000000 | temporal_id+1=001
        // Byte 0: 0_100000_0 = 0x40
        // Byte 1: 00000_001 = 0x01
        let (nal_type, layer_id, tid) = parse_nal_header(&[0x40, 0x01]).unwrap();
        assert_eq!(nal_type, NalUnitType::VpsNut);
        assert_eq!(layer_id, 0);
        assert_eq!(tid, 1);
    }

    #[test]
    fn parse_sps_header() {
        // SPS NAL: type=33, layer_id=0, temporal_id=1
        // Byte 0: 0_100001_0 = 0x42
        // Byte 1: 00000_001 = 0x01
        let (nal_type, layer_id, tid) = parse_nal_header(&[0x42, 0x01]).unwrap();
        assert_eq!(nal_type, NalUnitType::SpsNut);
        assert_eq!(layer_id, 0);
        assert_eq!(tid, 1);
    }

    #[test]
    fn parse_pps_header() {
        // PPS NAL: type=34
        let (nal_type, _, _) = parse_nal_header(&[0x44, 0x01]).unwrap();
        assert_eq!(nal_type, NalUnitType::PpsNut);
    }

    #[test]
    fn parse_idr_header() {
        // IDR_W_RADL: type=19
        // Byte 0: 0_010011_0 = 0x26
        let (nal_type, _, _) = parse_nal_header(&[0x26, 0x01]).unwrap();
        assert_eq!(nal_type, NalUnitType::IdrWRadl);
        assert!(nal_type.is_idr());
        assert!(nal_type.is_irap());
        assert!(nal_type.is_vcl());
    }

    #[test]
    fn parse_cra_header() {
        // CRA_NUT: type=21
        // Byte 0: 0_010101_0 = 0x2A
        let (nal_type, _, _) = parse_nal_header(&[0x2A, 0x01]).unwrap();
        assert_eq!(nal_type, NalUnitType::CraNut);
        assert!(nal_type.is_irap());
    }

    #[test]
    fn forbidden_bit_set_is_error() {
        // Byte 0: 1_000000_0 = 0x80
        let err = parse_nal_header(&[0x80, 0x01]).unwrap_err();
        assert!(matches!(err, HevcError::InvalidNal(_)));
    }

    #[test]
    fn temporal_id_zero_is_error() {
        let err = parse_nal_header(&[0x40, 0x00]).unwrap_err();
        assert!(matches!(err, HevcError::InvalidNal(_)));
    }

    #[test]
    fn too_short_is_error() {
        let err = parse_nal_header(&[0x40]).unwrap_err();
        assert!(matches!(err, HevcError::InvalidNal(_)));
    }

    #[test]
    fn rbsp_no_prevention_bytes() {
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let rbsp = extract_rbsp(&data);
        assert_eq!(rbsp, data);
    }

    #[test]
    fn rbsp_removes_single_prevention_byte() {
        // 0x00 0x00 0x03 -> 0x00 0x00
        let data = vec![0xAA, 0x00, 0x00, 0x03, 0xBB];
        let rbsp = extract_rbsp(&data);
        assert_eq!(rbsp, vec![0xAA, 0x00, 0x00, 0xBB]);
    }

    #[test]
    fn rbsp_removes_multiple_prevention_bytes() {
        let data = vec![0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0xFF];
        let rbsp = extract_rbsp(&data);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn rbsp_preserves_0x00_0x00_without_0x03() {
        let data = vec![0x00, 0x00, 0x01]; // start code, not prevention
        let rbsp = extract_rbsp(&data);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x01]);
    }

    #[test]
    fn nal_iterator_single_nal() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&[0x00, 0x00, 0x01]); // start code
        stream.extend_from_slice(&[0x40, 0x01, 0xAA, 0xBB]); // VPS NAL

        let nals: Vec<&[u8]> = NalIterator::new(&stream).collect();
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0], &[0x40, 0x01, 0xAA, 0xBB]);
    }

    #[test]
    fn nal_iterator_multiple_nals() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // 4-byte start code
        stream.extend_from_slice(&[0x40, 0x01, 0x11]); // VPS
        stream.extend_from_slice(&[0x00, 0x00, 0x01]); // 3-byte start code
        stream.extend_from_slice(&[0x42, 0x01, 0x22, 0x33]); // SPS
        stream.extend_from_slice(&[0x00, 0x00, 0x01]); // 3-byte start code
        stream.extend_from_slice(&[0x44, 0x01, 0x44]); // PPS

        let nals: Vec<&[u8]> = NalIterator::new(&stream).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0][0], 0x40); // VPS
        assert_eq!(nals[1][0], 0x42); // SPS
        assert_eq!(nals[2][0], 0x44); // PPS
    }

    #[test]
    fn nal_iterator_empty() {
        let nals: Vec<&[u8]> = NalIterator::new(&[]).collect();
        assert_eq!(nals.len(), 0);
    }

    #[test]
    fn nal_iterator_no_start_code() {
        let nals: Vec<&[u8]> = NalIterator::new(&[0x40, 0x01, 0xAA]).collect();
        assert_eq!(nals.len(), 0);
    }

    #[test]
    fn parse_full_nal_unit() {
        let data = vec![0x42, 0x01, 0x00, 0x00, 0x03, 0xFF]; // SPS with prevention byte
        let nal = parse_nal_unit(&data).unwrap();
        assert_eq!(nal.nal_type, NalUnitType::SpsNut);
        assert_eq!(nal.nuh_layer_id, 0);
        assert_eq!(nal.nuh_temporal_id_plus1, 1);
        assert_eq!(nal.rbsp, vec![0x00, 0x00, 0xFF]); // prevention byte removed
    }

    #[test]
    fn nal_type_classification() {
        assert!(NalUnitType::IdrWRadl.is_vcl());
        assert!(NalUnitType::IdrNLp.is_vcl());
        assert!(NalUnitType::TrailR.is_vcl());
        assert!(!NalUnitType::VpsNut.is_vcl());
        assert!(!NalUnitType::SpsNut.is_vcl());

        assert!(NalUnitType::IdrWRadl.is_idr());
        assert!(NalUnitType::IdrNLp.is_idr());
        assert!(!NalUnitType::CraNut.is_idr());

        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(!NalUnitType::TrailR.is_irap());
    }
}
