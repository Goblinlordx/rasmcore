//! JPEG marker segment writing (ITU-T T.81 Section B).
//!
//! Writes all marker segments needed for baseline JFIF:
//! SOI, APP0, DQT, SOF0, DHT, SOS, RST, EOI.

use crate::entropy;

// Marker codes
const SOI: u16 = 0xFFD8;
const EOI: u16 = 0xFFD9;
const APP0: u16 = 0xFFE0;
const DQT: u16 = 0xFFDB;
const SOF0: u16 = 0xFFC0; // Baseline sequential, Huffman
const SOF1: u16 = 0xFFC1; // Extended sequential, Huffman
const SOF2: u16 = 0xFFC2; // Progressive, Huffman
const SOF9: u16 = 0xFFC9; // Extended sequential, Arithmetic
const SOF10: u16 = 0xFFCA; // Progressive, Arithmetic
const DHT: u16 = 0xFFC4;
const DAC: u16 = 0xFFCC; // Define Arithmetic Conditioning
const SOS: u16 = 0xFFDA;
const DRI: u16 = 0xFFDD;

fn write_marker(out: &mut Vec<u8>, marker: u16) {
    out.push((marker >> 8) as u8);
    out.push(marker as u8);
}

/// Write SOI (Start of Image) marker.
pub fn write_soi(out: &mut Vec<u8>) {
    write_marker(out, SOI);
}

/// Write EOI (End of Image) marker.
pub fn write_eoi(out: &mut Vec<u8>) {
    write_marker(out, EOI);
}

/// Write JFIF APP0 marker segment.
pub fn write_app0(out: &mut Vec<u8>) {
    write_marker(out, APP0);
    let len = 16u16; // segment length including itself
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(b"JFIF\0"); // identifier
    out.push(1); // version major
    out.push(1); // version minor
    out.push(0); // units (0 = no units, aspect ratio only)
    out.extend_from_slice(&1u16.to_be_bytes()); // x density
    out.extend_from_slice(&1u16.to_be_bytes()); // y density
    out.push(0); // thumbnail width
    out.push(0); // thumbnail height
}

/// Write DQT (Define Quantization Table) marker segment.
///
/// `table_id`: 0 = luma, 1 = chroma. `table`: 64 values in zigzag order.
pub fn write_dqt(out: &mut Vec<u8>, table_id: u8, table: &[u16; 64]) {
    write_marker(out, DQT);
    // Check if any value > 255 (needs 16-bit precision)
    let precision = if table.iter().any(|&v| v > 255) {
        1u8
    } else {
        0u8
    };
    let entry_size = if precision == 0 { 1 } else { 2 };
    let len = 2 + 1 + 64 * entry_size;
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push((precision << 4) | (table_id & 0x0F));
    for &v in table {
        if precision == 0 {
            out.push(v as u8);
        } else {
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
}

/// Write SOF0 (Start of Frame — Baseline DCT) marker segment.
///
/// `components`: slice of (component_id, h_sampling, v_sampling, quant_table_id).
pub fn write_sof0(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_sof(out, SOF0, width, height, precision, components);
}

/// Write SOF1 (Extended sequential, Huffman) marker. Same format as SOF0.
pub fn write_sof1(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_sof(out, SOF1, width, height, precision, components);
}

/// Write SOF2 (Progressive, Huffman) marker.
pub fn write_sof2(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_sof(out, SOF2, width, height, precision, components);
}

/// Write SOF9 (Extended sequential, Arithmetic) marker.
pub fn write_sof9(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_sof(out, SOF9, width, height, precision, components);
}

/// Write SOF10 (Progressive, Arithmetic) marker.
pub fn write_sof10(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_sof(out, SOF10, width, height, precision, components);
}

/// Generic SOF writer used by all SOF variants.
fn write_sof(
    out: &mut Vec<u8>,
    marker: u16,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    write_marker(out, marker);
    let len = 2 + 1 + 2 + 2 + 1 + components.len() as u16 * 3;
    out.extend_from_slice(&len.to_be_bytes());
    out.push(precision);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(components.len() as u8);
    for &(id, h, v, qt) in components {
        out.push(id);
        out.push((h << 4) | v);
        out.push(qt);
    }
}

/// Write SOS for progressive scan with spectral selection and successive approximation.
///
/// `components`: (component_id, dc_table, ac_table)
/// `ss`: spectral selection start (0 = DC, 1-63 = AC bands)
/// `se`: spectral selection end (0 for DC-only, 63 for full AC)
/// `ah`: successive approximation high bit (0 for first scan)
/// `al`: successive approximation low bit
pub fn write_sos_progressive(
    out: &mut Vec<u8>,
    components: &[(u8, u8, u8)],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) {
    write_marker(out, SOS);
    let len = 2 + 1 + components.len() as u16 * 2 + 3;
    out.extend_from_slice(&len.to_be_bytes());
    out.push(components.len() as u8);
    for &(id, dc_table, ac_table) in components {
        out.push(id);
        out.push((dc_table << 4) | ac_table);
    }
    out.push(ss);
    out.push(se);
    out.push((ah << 4) | al);
}

/// Write DAC (Define Arithmetic Conditioning) marker.
///
/// Sets conditioning values for arithmetic coding contexts.
pub fn write_dac(out: &mut Vec<u8>, table_class: u8, table_id: u8, value: u8) {
    write_marker(out, DAC);
    let len: u16 = 2 + 2; // length field + 1 table spec
    out.extend_from_slice(&len.to_be_bytes());
    out.push((table_class << 4) | table_id);
    out.push(value);
}

/// Write DHT (Define Huffman Table) marker segment.
///
/// `table_class`: 0 = DC, 1 = AC. `table_id`: 0 = luma, 1 = chroma.
/// `code_lengths`: the code lengths array used to build the Huffman table.
pub fn write_dht(out: &mut Vec<u8>, table_class: u8, table_id: u8, code_lengths: &[u8]) {
    write_marker(out, DHT);

    // Count codes of each length (1-16 bits)
    let mut counts = [0u8; 16];
    let mut symbols = Vec::new();
    for (sym, &len) in code_lengths.iter().enumerate() {
        if len > 0 && len <= 16 {
            counts[len as usize - 1] += 1;
            symbols.push((len, sym as u8));
        }
    }
    // Sort symbols by code length, then by symbol value
    symbols.sort();

    let len = 2 + 1 + 16 + symbols.len() as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.push((table_class << 4) | (table_id & 0x0F));
    out.extend_from_slice(&counts);
    for (_, sym) in &symbols {
        out.push(*sym);
    }
}

/// Write SOS (Start of Scan) marker segment.
///
/// `components`: slice of (component_id, dc_table_id, ac_table_id).
pub fn write_sos(out: &mut Vec<u8>, components: &[(u8, u8, u8)]) {
    write_marker(out, SOS);
    let len = 2 + 1 + components.len() as u16 * 2 + 3;
    out.extend_from_slice(&len.to_be_bytes());
    out.push(components.len() as u8);
    for &(id, dc, ac) in components {
        out.push(id);
        out.push((dc << 4) | ac);
    }
    out.push(0); // spectral selection start (Ss)
    out.push(63); // spectral selection end (Se)
    out.push(0); // successive approximation (Ah=0, Al=0)
}

/// Write DRI (Define Restart Interval) marker segment.
pub fn write_dri(out: &mut Vec<u8>, interval: u16) {
    write_marker(out, DRI);
    out.extend_from_slice(&4u16.to_be_bytes()); // length
    out.extend_from_slice(&interval.to_be_bytes());
}

/// Write RST marker (RST0-RST7, cycles).
pub fn write_rst(out: &mut Vec<u8>, count: u32) {
    let marker = 0xFFD0 + (count % 8) as u16;
    write_marker(out, marker);
}

/// Write the standard 4 Huffman tables (DC luma, DC chroma, AC luma, AC chroma).
pub fn write_standard_huffman_tables(out: &mut Vec<u8>) {
    write_dht(out, 0, 0, &entropy::DC_LUMA_CODE_LENGTHS); // DC luma
    write_dht(out, 0, 1, &entropy::DC_CHROMA_CODE_LENGTHS); // DC chroma
    write_dht(out, 1, 0, &entropy::AC_LUMA_CODE_LENGTHS); // AC luma
    write_dht(out, 1, 1, &entropy::AC_CHROMA_CODE_LENGTHS); // AC chroma
}

/// Apply byte stuffing to entropy-coded data.
///
/// In JPEG, 0xFF bytes in data must be followed by 0x00 (byte stuffing)
/// to distinguish from marker codes.
pub fn byte_stuff(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 64);
    for &b in data {
        out.push(b);
        if b == 0xFF {
            out.push(0x00);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soi_eoi_markers() {
        let mut out = Vec::new();
        write_soi(&mut out);
        write_eoi(&mut out);
        assert_eq!(out, [0xFF, 0xD8, 0xFF, 0xD9]);
    }

    #[test]
    fn app0_jfif_header() {
        let mut out = Vec::new();
        write_app0(&mut out);
        assert_eq!(&out[0..2], &[0xFF, 0xE0]);
        assert_eq!(&out[4..9], b"JFIF\0");
    }

    #[test]
    fn byte_stuffing() {
        let data = [0xFF, 0x00, 0xFF];
        let stuffed = byte_stuff(&data);
        assert_eq!(stuffed, [0xFF, 0x00, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn dqt_8bit() {
        let table = [10u16; 64];
        let mut out = Vec::new();
        write_dqt(&mut out, 0, &table);
        assert_eq!(&out[0..2], &[0xFF, 0xDB]);
        // precision = 0 (8-bit), so 2+1+64 = 67 length
        let len = u16::from_be_bytes(out[2..4].try_into().unwrap());
        assert_eq!(len, 67);
    }

    #[test]
    fn sof0_baseline() {
        let mut out = Vec::new();
        let components = [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]; // 4:2:0
        write_sof0(&mut out, 640, 480, 8, &components);
        assert_eq!(&out[0..2], &[0xFF, 0xC0]);
        // Check dimensions
        let h = u16::from_be_bytes(out[5..7].try_into().unwrap());
        let w = u16::from_be_bytes(out[7..9].try_into().unwrap());
        assert_eq!(h, 480);
        assert_eq!(w, 640);
    }
}
