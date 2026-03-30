//! Lossless JPEG decoder (ITU-T T.81, Annex H).
//!
//! Lossless JPEG uses predictive coding (not DCT). Each pixel value is predicted
//! from neighboring pixels, and only the difference is entropy-coded (Huffman).
//! Common in DNG RAW files for lossless compression of sensor data.
//!
//! Only the subset used by DNG is implemented:
//! - SOF3 (lossless sequential Huffman)
//! - 1 or 2 components, 12-16 bits precision
//! - Huffman decode of difference values

use crate::RawError;

/// JPEG markers
const SOI: u8 = 0xD8;
const SOF3: u8 = 0xC3; // Lossless sequential Huffman
const DHT: u8 = 0xC4;
const SOS: u8 = 0xDA;
const EOI: u8 = 0xD9;

/// Huffman table for lossless JPEG decode.
struct HuffTable {
    /// Lookup table: for codes up to 16 bits, (value, code_length).
    /// Index = code << (16 - length), zero-padded.
    lut: Vec<(u8, u8)>,
}

impl HuffTable {
    fn build(bits: [u8; 17], huffval: Vec<u8>) -> Self {
        // Build fast lookup table (16-bit indexed)
        let mut lut = vec![(0u8, 0u8); 65536];
        let mut code: u32 = 0;
        let mut val_idx = 0usize;
        for length in 1..=16u8 {
            for _ in 0..bits[length as usize] {
                if val_idx < huffval.len() {
                    let val = huffval[val_idx];
                    // Fill all LUT entries that start with this code
                    let shift = 16 - length;
                    let fill_count = 1u32 << shift;
                    let base = (code as usize) << shift;
                    for j in 0..fill_count as usize {
                        if base + j < 65536 {
                            lut[base + j] = (val, length);
                        }
                    }
                    val_idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }
        Self { lut }
    }
}

/// Bit reader for the entropy-coded segment.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bit buffer (up to 32 bits accumulated).
    buf: u32,
    /// Number of valid bits in buf.
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            buf: 0,
            bits_left: 0,
        }
    }

    /// Fill buffer to at least `need` bits.
    fn fill(&mut self, need: u8) {
        while self.bits_left < need {
            if self.pos >= self.data.len() {
                // Pad with zeros at end
                self.buf <<= 8;
                self.bits_left += 8;
                continue;
            }
            let mut byte = self.data[self.pos];
            self.pos += 1;
            // Handle JPEG byte stuffing: 0xFF 0x00 -> 0xFF
            if byte == 0xFF {
                if self.pos < self.data.len() && self.data[self.pos] == 0x00 {
                    self.pos += 1; // skip stuffed zero
                }
                // If it's a marker (not 0x00), we've hit end of scan
                // For robustness, still use the 0xFF byte
            }
            // Avoid shifting u32 by 32 or more
            if self.bits_left <= 24 {
                self.buf = (self.buf << 8) | byte as u32;
                self.bits_left += 8;
            } else {
                // Edge case: can happen if need < 8 was called multiple times
                let shift = 32 - self.bits_left;
                byte >>= 8 - shift;
                self.buf |= (byte as u32) & ((1 << shift) - 1);
                self.bits_left = 32;
            }
        }
    }

    /// Peek at the top `n` bits without consuming them.
    fn peek(&mut self, n: u8) -> u32 {
        self.fill(n);
        (self.buf >> (self.bits_left - n)) & ((1 << n) - 1)
    }

    /// Consume `n` bits.
    fn skip(&mut self, n: u8) {
        self.bits_left -= n;
    }

    /// Read `n` bits as unsigned.
    fn read(&mut self, n: u8) -> u32 {
        if n == 0 {
            return 0;
        }
        let val = self.peek(n);
        self.skip(n);
        val
    }

    /// Decode one Huffman symbol.
    fn decode_huff(&mut self, table: &HuffTable) -> Result<u8, RawError> {
        self.fill(16);
        let top16 = (self.buf >> (self.bits_left - 16)) & 0xFFFF;
        let (val, len) = table.lut[top16 as usize];
        if len == 0 {
            return Err(RawError::InvalidFormat(
                "lossless JPEG: invalid Huffman code".into(),
            ));
        }
        self.bits_left -= len;
        Ok(val)
    }
}

/// Decoded lossless JPEG image.
pub struct LjpegImage {
    pub width: u32,
    pub height: u32,
    pub components: u8,
    /// Row-major pixel data, one u16 per sample. For 2-component images,
    /// samples are interleaved: [c0, c1, c0, c1, ...].
    pub data: Vec<u16>,
}

/// Decode a lossless JPEG bitstream.
pub fn decode_ljpeg(data: &[u8]) -> Result<LjpegImage, RawError> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != SOI {
        return Err(RawError::InvalidFormat(
            "lossless JPEG: missing SOI marker".into(),
        ));
    }

    let mut pos = 2;
    let mut huff_tables: [Option<HuffTable>; 4] = [const { None }; 4];
    let mut width = 0u32;
    let mut height = 0u32;
    let mut precision = 0u8;
    let mut num_components = 0u8;
    let mut comp_huff_table: [u8; 4] = [0; 4]; // which Huffman table each component uses
    let predictor: u8;

    while pos + 1 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        pos += 2;

        match marker {
            SOF3 => {
                // Lossless JPEG frame header
                if pos + 2 > data.len() {
                    return Err(RawError::DataTruncated);
                }
                let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + seg_len > data.len() {
                    return Err(RawError::DataTruncated);
                }
                precision = data[pos + 2];
                height = u16::from_be_bytes([data[pos + 3], data[pos + 4]]) as u32;
                width = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as u32;
                num_components = data[pos + 7];
                for i in 0..num_components as usize {
                    let base = pos + 8 + i * 3;
                    // component ID = data[base], sampling = data[base+1], quant = data[base+2]
                    // We don't need these for lossless
                    let _ = data[base]; // component ID
                }
                pos += seg_len;
            }
            DHT => {
                // Huffman table definition
                if pos + 2 > data.len() {
                    return Err(RawError::DataTruncated);
                }
                let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + seg_len > data.len() {
                    return Err(RawError::DataTruncated);
                }
                let seg_end = pos + seg_len;
                let mut p = pos + 2;
                while p < seg_end {
                    let tc_th = data[p]; // upper nibble = class (0=DC), lower = table ID
                    let table_id = (tc_th & 0x0F) as usize;
                    p += 1;
                    let mut bits = [0u8; 17];
                    let mut total = 0usize;
                    for i in 1..=16 {
                        bits[i] = data[p + i - 1];
                        total += bits[i] as usize;
                    }
                    p += 16;
                    let huffval: Vec<u8> = data[p..p + total].to_vec();
                    p += total;
                    if table_id < 4 {
                        huff_tables[table_id] = Some(HuffTable::build(bits, huffval));
                    }
                }
                pos += seg_len;
            }
            SOS => {
                // Start of scan
                if pos + 2 > data.len() {
                    return Err(RawError::DataTruncated);
                }
                let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + seg_len > data.len() {
                    return Err(RawError::DataTruncated);
                }
                let ns = data[pos + 2]; // number of components in scan
                for i in 0..ns as usize {
                    let base = pos + 3 + i * 2;
                    let _cs = data[base]; // component selector
                    let td_ta = data[base + 1]; // upper = DC table, lower = AC table
                    let td = (td_ta >> 4) & 0x0F;
                    if i < 4 {
                        comp_huff_table[i] = td;
                    }
                }
                // Ss = predictor selection
                predictor = data[pos + 3 + ns as usize * 2];
                // Se and Ah/Al are not used for lossless
                pos += seg_len;

                // Now decode the entropy-coded segment
                return decode_scan(
                    &data[pos..],
                    width,
                    height,
                    num_components,
                    precision,
                    predictor,
                    &comp_huff_table,
                    &huff_tables,
                );
            }
            EOI => break,
            0xFF => {
                // Padding 0xFF bytes
                continue;
            }
            0x00 => {
                // Byte stuffing (shouldn't appear here)
                continue;
            }
            _ => {
                // Skip unknown marker segment
                if pos + 2 > data.len() {
                    break;
                }
                let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                pos += seg_len;
            }
        }
    }

    Err(RawError::InvalidFormat(
        "lossless JPEG: no scan data found".into(),
    ))
}

fn decode_scan(
    scan_data: &[u8],
    width: u32,
    height: u32,
    components: u8,
    precision: u8,
    predictor: u8,
    comp_tables: &[u8; 4],
    huff_tables: &[Option<HuffTable>; 4],
) -> Result<LjpegImage, RawError> {
    let nc = components as usize;
    let w = width as usize;
    let h = height as usize;
    let total_samples = w * h * nc;
    let mut output = vec![0u16; total_samples];
    let mut reader = BitReader::new(scan_data);

    // Initial prediction value: 1 << (precision - 1)
    let initial_pred = 1u32 << (precision - 1);
    let max_val = (1u32 << precision) - 1;

    for row in 0..h {
        for col in 0..w {
            for c in 0..nc {
                let table_id = comp_tables[c] as usize;
                let table = huff_tables[table_id].as_ref().ok_or_else(|| {
                    RawError::InvalidFormat(format!(
                        "lossless JPEG: missing Huffman table {table_id}"
                    ))
                })?;

                // Decode the category (SSSS)
                let ssss = reader.decode_huff(table)? as u8;

                // Decode the difference value
                let diff = if ssss == 0 {
                    0i32
                } else if ssss == 16 {
                    // Special case: diff = 32768
                    32768i32
                } else {
                    let bits = reader.read(ssss) as i32;
                    // Convert to signed: if MSB is 0, value is negative
                    if bits < (1 << (ssss - 1)) {
                        bits - (1 << ssss) + 1
                    } else {
                        bits
                    }
                };

                // Compute prediction
                let idx = (row * w + col) * nc + c;
                let pred = if row == 0 && col == 0 {
                    initial_pred as i32
                } else if row == 0 {
                    // First row: predict from left
                    output[idx - nc] as i32
                } else if col == 0 {
                    // First column: predict from above
                    output[idx - w * nc] as i32
                } else {
                    let ra = output[idx - nc] as i32; // left
                    let rb = output[idx - w * nc] as i32; // above
                    let rc = output[idx - w * nc - nc] as i32; // above-left
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra + rb - rc,
                        5 => ra + ((rb - rc) >> 1),
                        6 => rb + ((ra - rc) >> 1),
                        7 => (ra + rb) >> 1,
                        _ => ra,
                    }
                };

                let val = (pred + diff).clamp(0, max_val as i32) as u16;
                output[idx] = val;
            }
        }
    }

    Ok(LjpegImage {
        width,
        height,
        components,
        data: output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_basic() {
        let data = [0b10110011, 0b01010101];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(4), 0b1011);
        assert_eq!(r.read(4), 0b0011);
        assert_eq!(r.read(8), 0b01010101);
    }

    #[test]
    fn reject_invalid_soi() {
        assert!(decode_ljpeg(&[0x00, 0x00]).is_err());
    }
}
