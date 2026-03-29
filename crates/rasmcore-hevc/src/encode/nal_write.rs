//! NAL unit writer — header assembly, emulation prevention, Annex B packaging.
//!
//! Direct port of x265 4.1 encoder/nal.cpp NAL unit assembly.
//! This is the write counterpart to `nal.rs` (parser).
//!
//! Ref: x265 4.1 encoder/nal.cpp
//! Ref: ITU-T H.265 Section 7.3.1.1 (NAL unit syntax)
//! Ref: ITU-T H.265 Annex B (byte stream NAL unit syntax)

use crate::types::NalUnitType;

/// Assemble a complete NAL unit with header, emulation prevention, and Annex B start code.
///
/// Takes RBSP data and wraps it in a NAL unit with:
/// 1. Annex B start code (0x00000001)
/// 2. 2-byte NAL header (type, layer_id=0, temporal_id=1)
/// 3. RBSP with emulation prevention bytes inserted
///
/// Ref: x265 4.1 encoder/nal.cpp — writeNalUnitHeader + encodeNAL
pub fn assemble_nal_unit(
    nal_type: NalUnitType,
    rbsp: &[u8],
    use_long_start_code: bool,
) -> Vec<u8> {
    let nuh_layer_id: u8 = 0;
    let nuh_temporal_id_plus1: u8 = 1;

    // Estimate output size: start code + header + rbsp + some prevention bytes
    let mut out = Vec::with_capacity(4 + 2 + rbsp.len() + rbsp.len() / 256 + 16);

    // Annex B start code
    if use_long_start_code {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    } else {
        out.extend_from_slice(&[0x00, 0x00, 0x01]);
    }

    // NAL header (2 bytes)
    // Byte 0: forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id[5](1)
    // Byte 1: nuh_layer_id[4:0](5) | nuh_temporal_id_plus1(3)
    let nal_type_val = nal_type.as_u8();
    let header_byte0 = (nal_type_val << 1) | (nuh_layer_id >> 5);
    let header_byte1 = ((nuh_layer_id & 0x1F) << 3) | nuh_temporal_id_plus1;
    out.push(header_byte0);
    out.push(header_byte1);

    // Insert emulation prevention bytes in RBSP.
    // When the byte sequence 0x000000, 0x000001, 0x000002, or 0x000003 appears
    // in the RBSP, insert 0x03 after the second 0x00 to prevent start code confusion.
    //
    // Ref: x265 4.1 encoder/nal.cpp — emulation prevention insertion loop
    // Ref: ITU-T H.265 Section 7.4.1 — emulation prevention
    let mut zeros = 0u32;
    for &byte in rbsp {
        if zeros == 2 && byte <= 0x03 {
            // Insert emulation prevention byte
            out.push(0x03);
            zeros = 0;
        }
        out.push(byte);
        if byte == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }

    out
}

/// Assemble a complete Annex B bitstream from multiple NAL units.
///
/// VPS, SPS, PPS use 4-byte start codes. Slice NALs also use 4-byte start codes
/// (first slice in picture) or 3-byte (subsequent).
pub fn assemble_annex_b(nals: &[(NalUnitType, Vec<u8>)]) -> Vec<u8> {
    let total_size: usize = nals
        .iter()
        .map(|(_, rbsp)| 4 + 2 + rbsp.len() + rbsp.len() / 256 + 16)
        .sum();
    let mut out = Vec::with_capacity(total_size);

    for (nal_type, rbsp) in nals {
        // All NALs in a single-access-unit stream use 4-byte start codes.
        let nal = assemble_nal_unit(*nal_type, rbsp, true);
        out.extend_from_slice(&nal);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal;

    #[test]
    fn nal_header_roundtrip() {
        let rbsp = vec![0x01, 0x02, 0x03, 0x04];
        let nal_bytes = assemble_nal_unit(NalUnitType::SpsNut, &rbsp, true);

        // Should start with 0x00000001
        assert_eq!(&nal_bytes[..4], &[0x00, 0x00, 0x00, 0x01]);

        // Parse back the NAL header (skip start code)
        let (nal_type, layer_id, temporal_id) = nal::parse_nal_header(&nal_bytes[4..]).unwrap();
        assert_eq!(nal_type, NalUnitType::SpsNut);
        assert_eq!(layer_id, 0);
        assert_eq!(temporal_id, 1);
    }

    #[test]
    fn emulation_prevention_insertion() {
        // RBSP containing 0x000000 which needs prevention
        let rbsp = vec![0x00, 0x00, 0x00, 0x05];
        let nal_bytes = assemble_nal_unit(NalUnitType::SpsNut, &rbsp, true);

        // After start code (4) + header (2), the RBSP should have 0x03 inserted:
        // 0x00 0x00 0x03 0x00 0x05
        let payload = &nal_bytes[6..];
        assert_eq!(payload, &[0x00, 0x00, 0x03, 0x00, 0x05]);
    }

    #[test]
    fn emulation_prevention_for_start_codes() {
        // Test all patterns that need prevention: 00 00 00, 00 00 01, 00 00 02, 00 00 03
        for byte in 0x00..=0x03 {
            let rbsp = vec![0x00, 0x00, byte, 0xFF];
            let nal_bytes = assemble_nal_unit(NalUnitType::PpsNut, &rbsp, true);
            let payload = &nal_bytes[6..];
            assert_eq!(
                payload,
                &[0x00, 0x00, 0x03, byte, 0xFF],
                "prevention failed for 00 00 {:02x}",
                byte
            );
        }
    }

    #[test]
    fn no_prevention_for_04_and_above() {
        let rbsp = vec![0x00, 0x00, 0x04, 0xFF];
        let nal_bytes = assemble_nal_unit(NalUnitType::PpsNut, &rbsp, true);
        let payload = &nal_bytes[6..];
        // 0x04 doesn't need prevention
        assert_eq!(payload, &[0x00, 0x00, 0x04, 0xFF]);
    }

    #[test]
    fn full_nal_roundtrip() {
        // Write a NAL, then parse it back with the decoder's NAL parser
        let original_rbsp = vec![0x42, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0xFF];
        let nal_bytes = assemble_nal_unit(NalUnitType::VpsNut, &original_rbsp, true);

        // Parse using NalIterator
        let mut found = false;
        for nal_data in nal::NalIterator::new(&nal_bytes) {
            let parsed = nal::parse_nal_unit(nal_data).unwrap();
            assert_eq!(parsed.nal_type, NalUnitType::VpsNut);
            // RBSP should match after emulation prevention removal
            assert_eq!(parsed.rbsp, original_rbsp);
            found = true;
        }
        assert!(found, "NalIterator should find one NAL unit");
    }

    #[test]
    fn multi_nal_annex_b() {
        let nals = vec![
            (NalUnitType::VpsNut, vec![0x01, 0x02]),
            (NalUnitType::SpsNut, vec![0x03, 0x04]),
            (NalUnitType::PpsNut, vec![0x05, 0x06]),
        ];
        let stream = assemble_annex_b(&nals);

        // Parse back with NalIterator — should find 3 NALs
        let parsed: Vec<_> = nal::NalIterator::new(&stream)
            .map(|data| nal::parse_nal_unit(data).unwrap())
            .collect();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].nal_type, NalUnitType::VpsNut);
        assert_eq!(parsed[1].nal_type, NalUnitType::SpsNut);
        assert_eq!(parsed[2].nal_type, NalUnitType::PpsNut);
        assert_eq!(parsed[0].rbsp, vec![0x01, 0x02]);
        assert_eq!(parsed[1].rbsp, vec![0x03, 0x04]);
        assert_eq!(parsed[2].rbsp, vec![0x05, 0x06]);
    }

    #[test]
    fn short_start_code() {
        let rbsp = vec![0x01];
        let nal_bytes = assemble_nal_unit(NalUnitType::SpsNut, &rbsp, false);
        // 3-byte start code
        assert_eq!(&nal_bytes[..3], &[0x00, 0x00, 0x01]);
        assert_eq!(nal_bytes.len(), 3 + 2 + 1); // start + header + rbsp
    }
}
