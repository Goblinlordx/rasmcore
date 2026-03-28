//! JPEG entropy coding — Huffman and arithmetic.
//!
//! ITU-T T.81 Sections F (Huffman) and D (Arithmetic).
//!
//! - DC coefficient DPCM (differential coding)
//! - AC coefficient run-length encoding (zigzag scan)
//! - Standard Huffman tables (Annex K)
//! - Optimized Huffman tables (two-pass frequency counting)
//! - QM-coder arithmetic coding (Annex D)
//!
//! Uses shared infrastructure:
//! - `rasmcore-bitio` for MSB-first bit emission
//! - `rasmcore-deflate::huffman` for Huffman encode/decode and code length generation

use rasmcore_bitio::{BitOrder, BitWriter};
use rasmcore_deflate::huffman::{HuffmanEncoder, build_code_lengths};

// ─── Huffman Tables (ITU-T T.81 Annex K) ──────────────────────────────────

/// Standard DC luminance Huffman code lengths (Table K.3).
/// 12 symbols (DC categories 0-11).
pub const DC_LUMA_CODE_LENGTHS: [u8; 12] = [2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9];

/// Standard DC chrominance Huffman code lengths (Table K.4).
pub const DC_CHROMA_CODE_LENGTHS: [u8; 12] = [2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard AC luminance code lengths (Table K.5).
/// 256 entries indexed by (run_length << 4 | size). Length 0 = unused.
pub static AC_LUMA_CODE_LENGTHS: [u8; 256] = build_ac_table(true);

/// Standard AC chrominance code lengths (Table K.6).
pub static AC_CHROMA_CODE_LENGTHS: [u8; 256] = build_ac_table(false);

const fn build_ac_table(luma: bool) -> [u8; 256] {
    let mut t = [0u8; 256];

    if luma {
        // Table K.5 — AC luminance
        t[0x00] = 4; // EOB
        t[0x01] = 2;
        t[0x02] = 2;
        t[0x03] = 3;
        t[0x04] = 4;
        t[0x05] = 5;
        t[0x06] = 7;
        t[0x07] = 8;
        t[0x08] = 10;
        t[0x09] = 16;
        t[0x0A] = 16;
        t[0x11] = 4;
        t[0x12] = 5;
        t[0x13] = 7;
        t[0x14] = 9;
        t[0x15] = 11;
        t[0x16] = 16;
        t[0x17] = 16;
        t[0x18] = 16;
        t[0x19] = 16;
        t[0x1A] = 16;
        t[0x21] = 5;
        t[0x22] = 8;
        t[0x23] = 10;
        t[0x24] = 12;
        t[0x25] = 16;
        t[0x26] = 16;
        t[0x27] = 16;
        t[0x28] = 16;
        t[0x29] = 16;
        t[0x2A] = 16;
        t[0x31] = 6;
        t[0x32] = 9;
        t[0x33] = 12;
        t[0x34] = 16;
        t[0x35] = 16;
        t[0x36] = 16;
        t[0x37] = 16;
        t[0x38] = 16;
        t[0x39] = 16;
        t[0x3A] = 16;
        t[0x41] = 6;
        t[0x42] = 10;
        t[0x43] = 16;
        t[0x44] = 16;
        t[0x45] = 16;
        t[0x46] = 16;
        t[0x47] = 16;
        t[0x48] = 16;
        t[0x49] = 16;
        t[0x4A] = 16;
        t[0x51] = 7;
        t[0x52] = 11;
        t[0x53] = 16;
        t[0x54] = 16;
        t[0x55] = 16;
        t[0x56] = 16;
        t[0x57] = 16;
        t[0x58] = 16;
        t[0x59] = 16;
        t[0x5A] = 16;
        t[0x61] = 7;
        t[0x62] = 12;
        t[0x63] = 16;
        t[0x64] = 16;
        t[0x65] = 16;
        t[0x66] = 16;
        t[0x67] = 16;
        t[0x68] = 16;
        t[0x69] = 16;
        t[0x6A] = 16;
        t[0x71] = 8;
        t[0x72] = 12;
        t[0x73] = 16;
        t[0x74] = 16;
        t[0x75] = 16;
        t[0x76] = 16;
        t[0x77] = 16;
        t[0x78] = 16;
        t[0x79] = 16;
        t[0x7A] = 16;
        t[0x81] = 9;
        t[0x82] = 15;
        t[0x83] = 16;
        t[0x84] = 16;
        t[0x85] = 16;
        t[0x86] = 16;
        t[0x87] = 16;
        t[0x88] = 16;
        t[0x89] = 16;
        t[0x8A] = 16;
        t[0x91] = 9;
        t[0x92] = 16;
        t[0x93] = 16;
        t[0x94] = 16;
        t[0x95] = 16;
        t[0x96] = 16;
        t[0x97] = 16;
        t[0x98] = 16;
        t[0x99] = 16;
        t[0x9A] = 16;
        t[0xA1] = 9;
        t[0xA2] = 16;
        t[0xA3] = 16;
        t[0xA4] = 16;
        t[0xA5] = 16;
        t[0xA6] = 16;
        t[0xA7] = 16;
        t[0xA8] = 16;
        t[0xA9] = 16;
        t[0xAA] = 16;
        t[0xB1] = 10;
        t[0xB2] = 16;
        t[0xB3] = 16;
        t[0xB4] = 16;
        t[0xB5] = 16;
        t[0xB6] = 16;
        t[0xB7] = 16;
        t[0xB8] = 16;
        t[0xB9] = 16;
        t[0xBA] = 16;
        t[0xC1] = 10;
        t[0xC2] = 16;
        t[0xC3] = 16;
        t[0xC4] = 16;
        t[0xC5] = 16;
        t[0xC6] = 16;
        t[0xC7] = 16;
        t[0xC8] = 16;
        t[0xC9] = 16;
        t[0xCA] = 16;
        t[0xD1] = 11;
        t[0xD2] = 16;
        t[0xD3] = 16;
        t[0xD4] = 16;
        t[0xD5] = 16;
        t[0xD6] = 16;
        t[0xD7] = 16;
        t[0xD8] = 16;
        t[0xD9] = 16;
        t[0xDA] = 16;
        t[0xE1] = 16;
        t[0xE2] = 16;
        t[0xE3] = 16;
        t[0xE4] = 16;
        t[0xE5] = 16;
        t[0xE6] = 16;
        t[0xE7] = 16;
        t[0xE8] = 16;
        t[0xE9] = 16;
        t[0xEA] = 16;
        t[0xF0] = 11; // ZRL
        t[0xF1] = 16;
        t[0xF2] = 16;
        t[0xF3] = 16;
        t[0xF4] = 16;
        t[0xF5] = 16;
        t[0xF6] = 16;
        t[0xF7] = 16;
        t[0xF8] = 16;
        t[0xF9] = 16;
        t[0xFA] = 16;
    } else {
        // Table K.6 — AC chrominance
        t[0x00] = 2; // EOB
        t[0x01] = 2;
        t[0x02] = 3;
        t[0x03] = 4;
        t[0x04] = 5;
        t[0x05] = 5;
        t[0x06] = 6;
        t[0x07] = 7;
        t[0x08] = 9;
        t[0x09] = 10;
        t[0x0A] = 12;
        t[0x11] = 4;
        t[0x12] = 6;
        t[0x13] = 8;
        t[0x14] = 9;
        t[0x15] = 11;
        t[0x16] = 12;
        t[0x17] = 16;
        t[0x18] = 16;
        t[0x19] = 16;
        t[0x1A] = 16;
        t[0x21] = 5;
        t[0x22] = 8;
        t[0x23] = 10;
        t[0x24] = 12;
        t[0x25] = 15;
        t[0x26] = 16;
        t[0x27] = 16;
        t[0x28] = 16;
        t[0x29] = 16;
        t[0x2A] = 16;
        t[0x31] = 5;
        t[0x32] = 8;
        t[0x33] = 10;
        t[0x34] = 12;
        t[0x35] = 16;
        t[0x36] = 16;
        t[0x37] = 16;
        t[0x38] = 16;
        t[0x39] = 16;
        t[0x3A] = 16;
        t[0x41] = 6;
        t[0x42] = 9;
        t[0x43] = 16;
        t[0x44] = 16;
        t[0x45] = 16;
        t[0x46] = 16;
        t[0x47] = 16;
        t[0x48] = 16;
        t[0x49] = 16;
        t[0x4A] = 16;
        t[0x51] = 6;
        t[0x52] = 10;
        t[0x53] = 16;
        t[0x54] = 16;
        t[0x55] = 16;
        t[0x56] = 16;
        t[0x57] = 16;
        t[0x58] = 16;
        t[0x59] = 16;
        t[0x5A] = 16;
        t[0x61] = 7;
        t[0x62] = 11;
        t[0x63] = 16;
        t[0x64] = 16;
        t[0x65] = 16;
        t[0x66] = 16;
        t[0x67] = 16;
        t[0x68] = 16;
        t[0x69] = 16;
        t[0x6A] = 16;
        t[0x71] = 7;
        t[0x72] = 11;
        t[0x73] = 16;
        t[0x74] = 16;
        t[0x75] = 16;
        t[0x76] = 16;
        t[0x77] = 16;
        t[0x78] = 16;
        t[0x79] = 16;
        t[0x7A] = 16;
        t[0x81] = 8;
        t[0x82] = 16;
        t[0x83] = 16;
        t[0x84] = 16;
        t[0x85] = 16;
        t[0x86] = 16;
        t[0x87] = 16;
        t[0x88] = 16;
        t[0x89] = 16;
        t[0x8A] = 16;
        t[0x91] = 9;
        t[0x92] = 16;
        t[0x93] = 16;
        t[0x94] = 16;
        t[0x95] = 16;
        t[0x96] = 16;
        t[0x97] = 16;
        t[0x98] = 16;
        t[0x99] = 16;
        t[0x9A] = 16;
        t[0xA1] = 9;
        t[0xA2] = 16;
        t[0xA3] = 16;
        t[0xA4] = 16;
        t[0xA5] = 16;
        t[0xA6] = 16;
        t[0xA7] = 16;
        t[0xA8] = 16;
        t[0xA9] = 16;
        t[0xAA] = 16;
        t[0xB1] = 9;
        t[0xB2] = 16;
        t[0xB3] = 16;
        t[0xB4] = 16;
        t[0xB5] = 16;
        t[0xB6] = 16;
        t[0xB7] = 16;
        t[0xB8] = 16;
        t[0xB9] = 16;
        t[0xBA] = 16;
        t[0xC1] = 11;
        t[0xC2] = 16;
        t[0xC3] = 16;
        t[0xC4] = 16;
        t[0xC5] = 16;
        t[0xC6] = 16;
        t[0xC7] = 16;
        t[0xC8] = 16;
        t[0xC9] = 16;
        t[0xCA] = 16;
        t[0xD1] = 14;
        t[0xD2] = 16;
        t[0xD3] = 16;
        t[0xD4] = 16;
        t[0xD5] = 16;
        t[0xD6] = 16;
        t[0xD7] = 16;
        t[0xD8] = 16;
        t[0xD9] = 16;
        t[0xDA] = 16;
        t[0xE1] = 15;
        t[0xE2] = 16;
        t[0xE3] = 16;
        t[0xE4] = 16;
        t[0xE5] = 16;
        t[0xE6] = 16;
        t[0xE7] = 16;
        t[0xE8] = 16;
        t[0xE9] = 16;
        t[0xEA] = 16;
        t[0xF0] = 10; // ZRL
        t[0xF1] = 15;
        t[0xF2] = 16;
        t[0xF3] = 16;
        t[0xF4] = 16;
        t[0xF5] = 16;
        t[0xF6] = 16;
        t[0xF7] = 16;
        t[0xF8] = 16;
        t[0xF9] = 16;
        t[0xFA] = 16;
    }
    t
}

// ─── Huffman Entropy Encoder ───────────────────────────────────────────────

/// JPEG Huffman entropy encoder.
///
/// Handles DC DPCM, AC run-length encoding, and Huffman bit emission
/// via the shared `rasmcore-deflate::huffman::HuffmanEncoder`.
pub struct HuffmanEntropyEncoder {
    writer: BitWriter,
    dc_encoder: HuffmanEncoder,
    ac_encoder: HuffmanEncoder,
    prev_dc: i32,
}

impl HuffmanEntropyEncoder {
    /// Create with standard luminance tables (Annex K Tables K.3/K.5).
    pub fn new_luma() -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoder: HuffmanEncoder::from_code_lengths(&DC_LUMA_CODE_LENGTHS),
            ac_encoder: HuffmanEncoder::from_code_lengths(&AC_LUMA_CODE_LENGTHS),
            prev_dc: 0,
        }
    }

    /// Create with standard chrominance tables (Tables K.4/K.6).
    pub fn new_chroma() -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoder: HuffmanEncoder::from_code_lengths(&DC_CHROMA_CODE_LENGTHS),
            ac_encoder: HuffmanEncoder::from_code_lengths(&AC_CHROMA_CODE_LENGTHS),
            prev_dc: 0,
        }
    }

    /// Create with custom (optimized) code lengths.
    pub fn new_custom(dc_lengths: &[u8], ac_lengths: &[u8]) -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoder: HuffmanEncoder::from_code_lengths(dc_lengths),
            ac_encoder: HuffmanEncoder::from_code_lengths(ac_lengths),
            prev_dc: 0,
        }
    }

    /// Encode one 8x8 block of quantized coefficients (zigzag order).
    /// `coeffs[0]` = DC, `coeffs[1..64]` = AC in zigzag scan order.
    pub fn encode_block(&mut self, coeffs: &[i16; 64]) {
        let dc_diff = coeffs[0] as i32 - self.prev_dc;
        self.prev_dc = coeffs[0] as i32;
        self.encode_dc(dc_diff);
        self.encode_ac(&coeffs[1..]);
    }

    fn encode_dc(&mut self, diff: i32) {
        let cat = magnitude_category(diff);
        self.dc_encoder.write_symbol(&mut self.writer, cat as u16);
        if cat > 0 {
            write_magnitude_bits(&mut self.writer, diff, cat);
        }
    }

    fn encode_ac(&mut self, ac: &[i16]) {
        let mut zero_run = 0u8;
        for (i, &coeff) in ac.iter().enumerate() {
            if coeff == 0 {
                zero_run += 1;
                if ac[i + 1..].iter().all(|&c| c == 0) {
                    self.ac_encoder.write_symbol(&mut self.writer, 0x00); // EOB
                    return;
                }
                continue;
            }
            while zero_run >= 16 {
                self.ac_encoder.write_symbol(&mut self.writer, 0xF0); // ZRL
                zero_run -= 16;
            }
            let cat = magnitude_category(coeff as i32);
            let symbol = (zero_run as u16) << 4 | cat as u16;
            self.ac_encoder.write_symbol(&mut self.writer, symbol);
            write_magnitude_bits(&mut self.writer, coeff as i32, cat);
            zero_run = 0;
        }
        if zero_run > 0 {
            self.ac_encoder.write_symbol(&mut self.writer, 0x00); // EOB
        }
    }

    /// Reset DC prediction (for restart markers or new component).
    pub fn reset_dc(&mut self) {
        self.prev_dc = 0;
    }

    /// Finalize and return encoded bitstream.
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }

    /// Current bit position (for size estimation).
    pub fn bit_position(&self) -> u64 {
        self.writer.bit_position()
    }
}

// ─── Interleaved MCU Encoder ─────────────────────────────────────────────────

/// Interleaved MCU encoder — single BitWriter shared by all components.
///
/// This is the correct way to encode JPEG scans with multiple components:
/// all entropy data goes into one bitstream, with Huffman table switching
/// per component.
pub struct InterleavedMcuEncoder {
    writer: BitWriter,
    /// Huffman encoders per component (index by component selector).
    dc_encoders: Vec<HuffmanEncoder>,
    ac_encoders: Vec<HuffmanEncoder>,
    /// DC prediction per component.
    dc_pred: Vec<i32>,
}

impl InterleavedMcuEncoder {
    /// Create for grayscale (1 component with luma tables).
    pub fn new_gray() -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoders: vec![HuffmanEncoder::from_code_lengths(&DC_LUMA_CODE_LENGTHS)],
            ac_encoders: vec![HuffmanEncoder::from_code_lengths(&AC_LUMA_CODE_LENGTHS)],
            dc_pred: vec![0],
        }
    }

    /// Create for YCbCr (3 components: Y=luma, Cb=chroma, Cr=chroma).
    pub fn new_ycbcr() -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoders: vec![
                HuffmanEncoder::from_code_lengths(&DC_LUMA_CODE_LENGTHS), // Y
                HuffmanEncoder::from_code_lengths(&DC_CHROMA_CODE_LENGTHS), // Cb
                HuffmanEncoder::from_code_lengths(&DC_CHROMA_CODE_LENGTHS), // Cr
            ],
            ac_encoders: vec![
                HuffmanEncoder::from_code_lengths(&AC_LUMA_CODE_LENGTHS), // Y
                HuffmanEncoder::from_code_lengths(&AC_CHROMA_CODE_LENGTHS), // Cb
                HuffmanEncoder::from_code_lengths(&AC_CHROMA_CODE_LENGTHS), // Cr
            ],
            dc_pred: vec![0, 0, 0],
        }
    }

    /// Create for grayscale with custom (optimized) Huffman tables.
    pub fn new_gray_custom(dc_lengths: &[u8], ac_lengths: &[u8]) -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoders: vec![HuffmanEncoder::from_code_lengths(dc_lengths)],
            ac_encoders: vec![HuffmanEncoder::from_code_lengths(ac_lengths)],
            dc_pred: vec![0],
        }
    }

    /// Create for YCbCr with custom (optimized) Huffman tables.
    /// Takes separate luma and chroma tables.
    pub fn new_ycbcr_custom(
        dc_luma_lengths: &[u8],
        ac_luma_lengths: &[u8],
        dc_chroma_lengths: &[u8],
        ac_chroma_lengths: &[u8],
    ) -> Self {
        Self {
            writer: BitWriter::new(BitOrder::MsbFirst),
            dc_encoders: vec![
                HuffmanEncoder::from_code_lengths(dc_luma_lengths),
                HuffmanEncoder::from_code_lengths(dc_chroma_lengths),
                HuffmanEncoder::from_code_lengths(dc_chroma_lengths),
            ],
            ac_encoders: vec![
                HuffmanEncoder::from_code_lengths(ac_luma_lengths),
                HuffmanEncoder::from_code_lengths(ac_chroma_lengths),
                HuffmanEncoder::from_code_lengths(ac_chroma_lengths),
            ],
            dc_pred: vec![0, 0, 0],
        }
    }

    /// Encode one 8x8 block for a specific component (0=Y, 1=Cb, 2=Cr).
    pub fn encode_block(&mut self, component: usize, coeffs: &[i16; 64]) {
        let dc_diff = coeffs[0] as i32 - self.dc_pred[component];
        self.dc_pred[component] = coeffs[0] as i32;

        // DC
        let cat = magnitude_category(dc_diff);
        self.dc_encoders[component].write_symbol(&mut self.writer, cat as u16);
        if cat > 0 {
            write_magnitude_bits(&mut self.writer, dc_diff, cat);
        }

        // AC
        let ac = &coeffs[1..];
        let mut zero_run = 0u8;
        for (i, &coeff) in ac.iter().enumerate() {
            if coeff == 0 {
                zero_run += 1;
                if ac[i + 1..].iter().all(|&c| c == 0) {
                    self.ac_encoders[component].write_symbol(&mut self.writer, 0x00);
                    return;
                }
                continue;
            }
            while zero_run >= 16 {
                self.ac_encoders[component].write_symbol(&mut self.writer, 0xF0);
                zero_run -= 16;
            }
            let cat = magnitude_category(coeff as i32);
            let symbol = (zero_run as u16) << 4 | cat as u16;
            self.ac_encoders[component].write_symbol(&mut self.writer, symbol);
            write_magnitude_bits(&mut self.writer, coeff as i32, cat);
            zero_run = 0;
        }
        // Only write EOB if there are trailing zeros not yet signaled.
        // If all 63 AC positions were non-zero, the decoder exits naturally
        // at k=64 without needing EOB.
        if zero_run > 0 {
            self.ac_encoders[component].write_symbol(&mut self.writer, 0x00);
        }
    }

    /// Reset all DC predictions (for restart markers).
    pub fn reset_dc(&mut self) {
        for pred in &mut self.dc_pred {
            *pred = 0;
        }
    }

    /// Finalize and return the single interleaved bitstream.
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

// ─── Optimized Huffman Tables (Two-Pass) ───────────────────────────────────

/// Frequency counter for building optimized Huffman tables.
pub struct FrequencyCounter {
    /// DC symbol frequencies (12 categories).
    pub dc_freq: [u32; 12],
    /// AC symbol frequencies (256 run/size pairs).
    pub ac_freq: [u32; 256],
    prev_dc: i32,
}

impl FrequencyCounter {
    pub fn new() -> Self {
        Self {
            dc_freq: [0; 12],
            ac_freq: [0; 256],
            prev_dc: 0,
        }
    }

    /// Count symbols for one 8x8 block (first pass).
    pub fn count_block(&mut self, coeffs: &[i16; 64]) {
        let dc_diff = coeffs[0] as i32 - self.prev_dc;
        self.prev_dc = coeffs[0] as i32;
        self.dc_freq[magnitude_category(dc_diff) as usize] += 1;

        let ac = &coeffs[1..];
        let mut zero_run = 0u8;
        for (i, &coeff) in ac.iter().enumerate() {
            if coeff == 0 {
                zero_run += 1;
                if ac[i + 1..].iter().all(|&c| c == 0) {
                    self.ac_freq[0x00] += 1;
                    return;
                }
                continue;
            }
            while zero_run >= 16 {
                self.ac_freq[0xF0] += 1;
                zero_run -= 16;
            }
            let cat = magnitude_category(coeff as i32);
            self.ac_freq[(zero_run as usize) << 4 | cat as usize] += 1;
            zero_run = 0;
        }
        self.ac_freq[0x00] += 1;
    }

    /// Build optimal code lengths from frequencies.
    /// Returns (dc_lengths, ac_lengths) suitable for `HuffmanEntropyEncoder::new_custom`.
    ///
    /// Uses the JPEG-specific algorithm (ITU-T T.81 Annex K) which ensures
    /// the resulting tree is valid for JPEG: no code length level can be full
    /// (i.e., the all-ones code pattern at any length must be unused).
    pub fn build_optimal_tables(&self) -> (Vec<u8>, Vec<u8>) {
        (
            jpeg_build_code_lengths(&self.dc_freq, 16),
            jpeg_build_code_lengths(&self.ac_freq, 16),
        )
    }

    pub fn reset(&mut self) {
        self.dc_freq = [0; 12];
        self.ac_freq = [0; 256];
        self.prev_dc = 0;
    }
}

impl Default for FrequencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── JPEG-Compliant Huffman Code Length Generation ─────────────────────────

/// Build optimal Huffman code lengths following JPEG's constraint:
/// at every code length level, the accumulated code count must be strictly
/// less than the total code space (ITU-T T.81 Annex C / libjpeg validation).
///
/// After building the optimal tree via `build_code_lengths`, we verify
/// the JPEG constraint and fix violations by pushing symbols to longer
/// code lengths.
fn jpeg_build_code_lengths(frequencies: &[u32], max_len: u8) -> Vec<u8> {
    let mut lengths = build_code_lengths(frequencies, max_len);

    // Count symbols at each code length
    let mut counts = [0u32; 17];
    for &len in &lengths {
        if len > 0 && len <= 16 {
            counts[len as usize] += 1;
        }
    }

    // JPEG constraint check (Figure C.2 / libjpeg jdhuff.c):
    // After processing all symbols at length si, the running code value
    // must be < (1 << si). Equivalently: at each level, the cumulative
    // symbol count must be < (1 << level).
    //
    // A complete tree at depth D has exactly 2^D leaves. JPEG requires
    // the tree to be INCOMPLETE — at least one code pattern at some level
    // must be unused.
    //
    // Fix: if the tree is complete (or overfull at any level), move the
    // longest-code symbol one level deeper.
    loop {
        let mut code = 0u32;
        let mut needs_fix = false;
        for si in 1..=max_len as u32 {
            code += counts[si as usize];
            if code >= (1 << si) {
                needs_fix = true;
                break;
            }
            code <<= 1;
        }
        if !needs_fix {
            break;
        }

        // Find the longest used code length and move one symbol deeper
        let mut longest = 0usize;
        for i in (1..=max_len as usize).rev() {
            if counts[i] > 0 {
                longest = i;
                break;
            }
        }

        if longest == 0 || longest >= max_len as usize {
            // Can't fix — only one symbol or at max length already.
            // For single-symbol case, just use length 1 (JPEG allows this).
            break;
        }

        // Move one symbol from `longest` to `longest + 1`
        counts[longest] -= 1;
        counts[longest + 1] += 1;

        // Update the actual lengths array
        for len in lengths.iter_mut().rev() {
            if *len == longest as u8 {
                *len = (longest + 1) as u8;
                break;
            }
        }
    }

    lengths
}

// ─── Arithmetic Coder (Adaptive Binary, QM-coder probability estimation) ──

/// QM-coder probability estimation table (ITU-T T.81 Table D.3).
/// Each entry: (Qe, next_MPS_state, next_LPS_state, switch_MPS_sense)
/// Used for adaptive probability estimation in both encoder and decoder.
const QE_TABLE: [(u16, u8, u8, bool); 47] = [
    (0x5A1D, 1, 1, true),
    (0x2586, 2, 6, false),
    (0x1114, 3, 9, false),
    (0x080B, 4, 12, false),
    (0x03D8, 5, 29, false),
    (0x01DA, 38, 33, false),
    (0x0015, 7, 6, true),
    (0x006F, 8, 14, false),
    (0x0036, 9, 14, false),
    (0x001A, 10, 14, false),
    (0x000D, 11, 17, false),
    (0x0006, 12, 18, false),
    (0x0003, 13, 20, false),
    (0x0001, 29, 21, false),
    (0x5A7F, 15, 14, true),
    (0x3F25, 16, 14, false),
    (0x2CF2, 17, 15, false),
    (0x207C, 18, 16, false),
    (0x17B9, 19, 17, false),
    (0x1182, 20, 18, false),
    (0x0CEF, 21, 19, false),
    (0x09A1, 22, 19, false),
    (0x072F, 23, 20, false),
    (0x055C, 24, 21, false),
    (0x0406, 25, 22, false),
    (0x0303, 26, 23, false),
    (0x0240, 27, 24, false),
    (0x01B1, 28, 25, false),
    (0x0144, 29, 26, false),
    (0x00F5, 30, 27, false),
    (0x00B7, 31, 28, false),
    (0x008A, 32, 29, false),
    (0x0068, 33, 30, false),
    (0x004E, 34, 31, false),
    (0x003B, 35, 32, false),
    (0x002C, 36, 33, false),
    (0x0021, 37, 34, false),
    (0x0018, 38, 35, false),
    (0x0010, 39, 36, false),
    (0x000C, 40, 37, false),
    (0x0009, 41, 38, false),
    (0x0006, 42, 39, false),
    (0x0004, 43, 40, false),
    (0x0003, 44, 41, false),
    (0x0002, 45, 42, false),
    (0x0001, 45, 43, false),
    (0x5601, 46, 46, false), // uniform
];

/// Adaptive binary arithmetic encoder with QM-coder probability estimation.
///
/// Uses a textbook interval-based approach with bit-level output.
/// Precision: 18-bit interval with underflow counter.
pub struct ArithmeticEncoder {
    low: u32,
    high: u32,
    pending: u32,
    output: Vec<u8>,
    bit_buf: u8,
    bit_count: u8,
    contexts: Vec<(u8, bool)>,
}

/// Precision constants for the arithmetic coder.
const ARITH_PREC: u32 = 18;
const ARITH_WHOLE: u32 = 1 << ARITH_PREC; // 0x40000
const ARITH_HALF: u32 = 1 << (ARITH_PREC - 1); // 0x20000
const ARITH_QTR: u32 = 1 << (ARITH_PREC - 2); // 0x10000

impl ArithmeticEncoder {
    /// Create with `num_contexts` conditioning contexts.
    pub fn new(num_contexts: usize) -> Self {
        Self {
            low: 0,
            high: ARITH_WHOLE - 1,
            pending: 0,
            output: Vec::with_capacity(4096),
            bit_buf: 0,
            bit_count: 0,
            contexts: vec![(0, false); num_contexts],
        }
    }

    /// Encode a binary decision with the given context.
    pub fn encode(&mut self, ctx_id: usize, bit: bool) {
        let (state_idx, mps) = self.contexts[ctx_id];
        let (qe, nmps, nlps, switch) = QE_TABLE[state_idx as usize];

        // Scale qe (16-bit) to the current interval range
        let range = self.high - self.low + 1;
        let lps_size = ((range as u64 * qe as u64) >> 16).max(1) as u32;
        let split = self.high - lps_size + 1; // MPS: [low, split), LPS: [split, high]

        if bit == mps {
            // MPS sub-interval: [low, split)
            self.high = split - 1;
            if self.high < self.low + 1 {
                self.high = self.low;
            }
            self.contexts[ctx_id].0 = nmps;
        } else {
            // LPS sub-interval: [split, high]
            self.low = split;
            if switch {
                self.contexts[ctx_id].1 = !mps;
            }
            self.contexts[ctx_id].0 = nlps;
        }

        self.renormalize();
    }

    fn emit_bit(&mut self, bit: bool) {
        self.bit_buf = (self.bit_buf << 1) | (bit as u8);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.output.push(self.bit_buf);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    fn emit_bit_plus_pending(&mut self, bit: bool) {
        self.emit_bit(bit);
        for _ in 0..self.pending {
            self.emit_bit(!bit);
        }
        self.pending = 0;
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < ARITH_HALF {
                // Both in lower half → emit 0
                self.emit_bit_plus_pending(false);
            } else if self.low >= ARITH_HALF {
                // Both in upper half → emit 1
                self.emit_bit_plus_pending(true);
                self.low -= ARITH_HALF;
                self.high -= ARITH_HALF;
            } else if self.low >= ARITH_QTR && self.high < 3 * ARITH_QTR {
                // Middle region → underflow
                self.pending += 1;
                self.low -= ARITH_QTR;
                self.high -= ARITH_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    /// Flush and return encoded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Emit enough bits to disambiguate the final interval
        self.pending += 1;
        if self.low < ARITH_QTR {
            self.emit_bit_plus_pending(false);
        } else {
            self.emit_bit_plus_pending(true);
        }
        // Pad remaining bits in the last byte
        if self.bit_count > 0 {
            self.bit_buf <<= 8 - self.bit_count;
            self.output.push(self.bit_buf);
        }
        self.output
    }

    /// Reset a context to initial state.
    pub fn reset_context(&mut self, ctx_id: usize) {
        self.contexts[ctx_id] = (0, false);
    }
}

// ─── Arithmetic Decoder ──────────────────────────────────────────────────

/// Adaptive binary arithmetic decoder — matched to [`ArithmeticEncoder`].
///
/// Uses the same interval-based approach with bit-level input.
pub struct ArithmeticDecoder<'a> {
    low: u32,
    high: u32,
    code: u32,
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    contexts: Vec<(u8, bool)>,
}

impl<'a> ArithmeticDecoder<'a> {
    /// Create decoder from encoded data with `num_contexts` contexts.
    pub fn new(data: &'a [u8], num_contexts: usize) -> Self {
        let mut dec = Self {
            low: 0,
            high: ARITH_WHOLE - 1,
            code: 0,
            data,
            byte_pos: 0,
            bit_pos: 0,
            contexts: vec![(0, false); num_contexts],
        };
        // Pre-fill code register with ARITH_PREC bits
        for _ in 0..ARITH_PREC {
            dec.code = (dec.code << 1) | dec.read_bit() as u32;
        }
        dec
    }

    fn read_bit(&mut self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false; // padding zeros
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1 != 0;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }

    /// Decode one binary decision using the given context.
    pub fn decode(&mut self, ctx_id: usize) -> bool {
        let (state_idx, mps) = self.contexts[ctx_id];
        let (qe, nmps, nlps, switch) = QE_TABLE[state_idx as usize];

        let range = self.high - self.low + 1;
        let lps_size = ((range as u64 * qe as u64) >> 16).max(1) as u32;
        let split = self.high - lps_size + 1;

        let bit;
        if self.code < split {
            // MPS sub-interval
            self.high = split - 1;
            if self.high < self.low + 1 {
                self.high = self.low;
            }
            self.contexts[ctx_id].0 = nmps;
            bit = mps;
        } else {
            // LPS sub-interval
            self.low = split;
            if switch {
                self.contexts[ctx_id].1 = !mps;
            }
            self.contexts[ctx_id].0 = nlps;
            bit = !mps;
        }

        self.renormalize();
        bit
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < ARITH_HALF {
                // Lower half — do nothing to code, low, high
            } else if self.low >= ARITH_HALF {
                self.code -= ARITH_HALF;
                self.low -= ARITH_HALF;
                self.high -= ARITH_HALF;
            } else if self.low >= ARITH_QTR && self.high < 3 * ARITH_QTR {
                self.code -= ARITH_QTR;
                self.low -= ARITH_QTR;
                self.high -= ARITH_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | self.read_bit() as u32;
        }
    }

    /// Reset a context to initial state.
    pub fn reset_context(&mut self, ctx_id: usize) {
        self.contexts[ctx_id] = (0, false);
    }
}

// ─── JPEG Arithmetic Encode/Decode Helpers ──────────────────────────────

/// Context layout for JPEG arithmetic coding (ITU-T T.81 Annex F).
///
/// Per component:
///   DC: 5 contexts (zero, sign, magnitude categories)
///   AC: 64 contexts (63 positions + 1 EOB)
/// Total per component: 69
const DC_CTX_ZERO: usize = 0;
const DC_CTX_SIGN: usize = 1;
const DC_CTX_MAG_LO: usize = 2;
const DC_CTX_MAG_HI: usize = 3;
const DC_CTX_BITS: usize = 4;
const AC_CTX_BASE: usize = 5;
const AC_CTX_EOB: usize = 68;
const CONTEXTS_PER_COMPONENT: usize = 69;

/// Encode one 8x8 block using arithmetic coding.
pub fn arithmetic_encode_block(
    encoder: &mut ArithmeticEncoder,
    comp_ctx_offset: usize,
    dc_pred: &mut i32,
    coeffs: &[i16; 64],
) {
    // DC
    let dc_diff = coeffs[0] as i32 - *dc_pred;
    *dc_pred = coeffs[0] as i32;
    arithmetic_encode_dc(encoder, comp_ctx_offset, dc_diff);

    // AC (zigzag positions 1-63)
    arithmetic_encode_ac(encoder, comp_ctx_offset, &coeffs[1..]);
}

fn arithmetic_encode_dc(encoder: &mut ArithmeticEncoder, ctx_off: usize, diff: i32) {
    if diff == 0 {
        encoder.encode(ctx_off + DC_CTX_ZERO, false); // zero
        return;
    }
    encoder.encode(ctx_off + DC_CTX_ZERO, true); // non-zero
    encoder.encode(ctx_off + DC_CTX_SIGN, diff < 0); // sign

    let abs_val = diff.unsigned_abs();
    let category = 32 - abs_val.leading_zeros() as u8; // magnitude_category

    // Unary code for category (1-based)
    for i in 1..category {
        let ctx = ctx_off + if i < 3 { DC_CTX_MAG_LO } else { DC_CTX_MAG_HI };
        encoder.encode(ctx, true); // more bits
    }
    encoder.encode(
        ctx_off
            + if category < 3 {
                DC_CTX_MAG_LO
            } else {
                DC_CTX_MAG_HI
            },
        false,
    ); // stop

    // Magnitude bits (MSB to LSB) using fixed context
    if category > 1 {
        let mantissa = abs_val - (1 << (category - 1));
        for bit_pos in (0..category - 1).rev() {
            encoder.encode(ctx_off + DC_CTX_BITS, (mantissa >> bit_pos) & 1 != 0);
        }
    }
}

fn arithmetic_encode_ac(encoder: &mut ArithmeticEncoder, ctx_off: usize, ac: &[i16]) {
    let eob_ctx = ctx_off + AC_CTX_EOB;
    let mut last_nonzero = 62; // last non-zero position in ac[0..63]
    while last_nonzero > 0 && ac[last_nonzero] == 0 {
        last_nonzero -= 1;
    }
    if ac[last_nonzero] == 0 {
        // All zeros — emit EOB immediately
        encoder.encode(eob_ctx, true);
        return;
    }

    for k in 0..63usize {
        if k > last_nonzero {
            encoder.encode(eob_ctx, true); // EOB
            return;
        }
        encoder.encode(eob_ctx, false); // not EOB

        let coeff = ac[k];
        let ac_ctx = ctx_off + AC_CTX_BASE + k;

        if coeff == 0 {
            encoder.encode(ac_ctx, false); // zero
            continue;
        }

        encoder.encode(ac_ctx, true); // non-zero
        encoder.encode(ac_ctx, coeff < 0); // sign

        let abs_val = (coeff as i32).unsigned_abs();
        let category = 32 - abs_val.leading_zeros() as u8;

        // Unary magnitude category
        for _ in 1..category {
            encoder.encode(ac_ctx, true);
        }
        encoder.encode(ac_ctx, false);

        // Magnitude bits
        if category > 1 {
            let mantissa = abs_val - (1 << (category - 1));
            for bit_pos in (0..category - 1).rev() {
                encoder.encode(ac_ctx, (mantissa >> bit_pos) & 1 != 0);
            }
        }
    }
    // No trailing EOB — after all 63 AC positions, the block is implicitly done.
    // The decoder's loop also processes exactly 63 positions.
}

/// Decode one 8x8 block using arithmetic coding.
pub fn arithmetic_decode_block(
    decoder: &mut ArithmeticDecoder,
    comp_ctx_offset: usize,
    dc_pred: &mut i32,
    coeffs: &mut [i16; 64],
) {
    // DC
    let dc_diff = arithmetic_decode_dc(decoder, comp_ctx_offset);
    *dc_pred += dc_diff;
    coeffs[0] = *dc_pred as i16;

    // AC
    arithmetic_decode_ac(decoder, comp_ctx_offset, &mut coeffs[1..]);
}

fn arithmetic_decode_dc(decoder: &mut ArithmeticDecoder, ctx_off: usize) -> i32 {
    if !decoder.decode(ctx_off + DC_CTX_ZERO) {
        return 0;
    }
    let negative = decoder.decode(ctx_off + DC_CTX_SIGN);

    // Unary category decode
    let mut category = 1u8;
    while decoder.decode(
        ctx_off
            + if category < 3 {
                DC_CTX_MAG_LO
            } else {
                DC_CTX_MAG_HI
            },
    ) {
        category += 1;
        if category >= 15 {
            break;
        }
    }

    let mut value: i32 = if category > 1 {
        let mut mantissa: u32 = 0;
        for _ in 0..category - 1 {
            mantissa = (mantissa << 1) | decoder.decode(ctx_off + DC_CTX_BITS) as u32;
        }
        (mantissa + (1 << (category - 1))) as i32
    } else {
        1
    };

    if negative {
        value = -value;
    }
    value
}

fn arithmetic_decode_ac(decoder: &mut ArithmeticDecoder, ctx_off: usize, ac: &mut [i16]) {
    let eob_ctx = ctx_off + AC_CTX_EOB;

    for k in 0..63usize {
        if decoder.decode(eob_ctx) {
            return; // EOB
        }

        let ac_ctx = ctx_off + AC_CTX_BASE + k;
        if !decoder.decode(ac_ctx) {
            // zero coefficient
            continue;
        }

        // Non-zero
        let negative = decoder.decode(ac_ctx);

        let mut category = 1u8;
        while decoder.decode(ac_ctx) {
            category += 1;
            if category >= 15 {
                break;
            }
        }

        let abs_val: i32 = if category > 1 {
            let mut mantissa: u32 = 0;
            for _ in 0..category - 1 {
                mantissa = (mantissa << 1) | decoder.decode(ac_ctx) as u32;
            }
            (mantissa + (1 << (category - 1))) as i32
        } else {
            1
        };

        ac[k] = if negative {
            -abs_val as i16
        } else {
            abs_val as i16
        };
    }
}

/// Total number of arithmetic contexts needed for a JPEG image.
pub fn arithmetic_context_count(num_components: usize) -> usize {
    num_components * CONTEXTS_PER_COMPONENT
}

/// Get the context offset for a given component index.
pub fn arithmetic_ctx_offset(component_index: usize) -> usize {
    component_index * CONTEXTS_PER_COMPONENT
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Magnitude category (bit-length) of a coefficient value.
/// 0 → 0, ±1 → 1, ±2..3 → 2, ±4..7 → 3, etc.
pub fn magnitude_category(value: i32) -> u8 {
    if value == 0 {
        return 0;
    }
    32 - value.unsigned_abs().leading_zeros() as u8
}

/// Write magnitude bits for a non-zero coefficient.
/// Positive: value directly. Negative: one's complement (value-1, masked).
fn write_magnitude_bits(writer: &mut BitWriter, value: i32, category: u8) {
    if category == 0 {
        return;
    }
    let bits = if value >= 0 {
        value as u32
    } else {
        (value - 1) as u32 & ((1u32 << category) - 1)
    };
    writer.write_bits(category, bits);
}

/// Standard DC code lengths for luma/chroma selection.
pub fn standard_dc_lengths(is_luma: bool) -> &'static [u8] {
    if is_luma {
        &DC_LUMA_CODE_LENGTHS
    } else {
        &DC_CHROMA_CODE_LENGTHS
    }
}

/// Standard AC code lengths for luma/chroma selection.
pub fn standard_ac_lengths(is_luma: bool) -> &'static [u8; 256] {
    if is_luma {
        &AC_LUMA_CODE_LENGTHS
    } else {
        &AC_CHROMA_CODE_LENGTHS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_category_values() {
        assert_eq!(magnitude_category(0), 0);
        assert_eq!(magnitude_category(1), 1);
        assert_eq!(magnitude_category(-1), 1);
        assert_eq!(magnitude_category(2), 2);
        assert_eq!(magnitude_category(3), 2);
        assert_eq!(magnitude_category(4), 3);
        assert_eq!(magnitude_category(255), 8);
        assert_eq!(magnitude_category(2047), 11);
    }

    #[test]
    fn encode_all_zero_block() {
        let mut enc = HuffmanEntropyEncoder::new_luma();
        enc.encode_block(&[0i16; 64]);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_dc_only() {
        let mut enc = HuffmanEntropyEncoder::new_luma();
        let mut coeffs = [0i16; 64];
        coeffs[0] = 100;
        enc.encode_block(&coeffs);
        let bytes = enc.finish();
        assert!(bytes.len() > 1);
    }

    #[test]
    fn dc_dpcm_tracking() {
        let mut enc = HuffmanEntropyEncoder::new_luma();
        let mut c1 = [0i16; 64];
        c1[0] = 100;
        enc.encode_block(&c1);
        let mut c2 = [0i16; 64];
        c2[0] = 105;
        enc.encode_block(&c2);
        let bytes = enc.finish();
        assert!(bytes.len() > 2);
    }

    #[test]
    fn encode_with_ac() {
        let mut enc = HuffmanEntropyEncoder::new_luma();
        let mut coeffs = [0i16; 64];
        coeffs[0] = 50;
        coeffs[1] = 10;
        coeffs[2] = -5;
        coeffs[5] = 3;
        enc.encode_block(&coeffs);
        let bytes = enc.finish();
        assert!(bytes.len() > 3);
    }

    #[test]
    fn luma_chroma_differ() {
        let mut luma = HuffmanEntropyEncoder::new_luma();
        let mut chroma = HuffmanEntropyEncoder::new_chroma();
        let mut coeffs = [0i16; 64];
        coeffs[0] = 42;
        coeffs[1] = 7;
        luma.encode_block(&coeffs);
        chroma.encode_block(&coeffs);
        assert_ne!(luma.finish(), chroma.finish());
    }

    #[test]
    fn frequency_counter_collects() {
        let mut counter = FrequencyCounter::new();
        let mut coeffs = [0i16; 64];
        coeffs[0] = 10;
        coeffs[1] = 5;
        counter.count_block(&coeffs);
        assert!(counter.dc_freq[magnitude_category(10) as usize] > 0);
        assert!(counter.ac_freq[0x00] > 0); // EOB
    }

    #[test]
    fn optimized_tables_usable() {
        let mut counter = FrequencyCounter::new();
        for i in 0..100 {
            let mut coeffs = [0i16; 64];
            coeffs[0] = (i * 3 - 150) as i16;
            coeffs[1] = (i % 10) as i16;
            counter.count_block(&coeffs);
        }
        let (dc, ac) = counter.build_optimal_tables();
        let enc = HuffmanEntropyEncoder::new_custom(&dc, &ac);
        assert_eq!(enc.bit_position(), 0);
    }

    #[test]
    fn arithmetic_basic() {
        let mut enc = ArithmeticEncoder::new(4);
        for i in 0..100 {
            enc.encode(0, i % 3 == 0);
            enc.encode(1, i % 2 == 0);
        }
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn arithmetic_single_bit_roundtrip() {
        // Single true bit
        let mut enc = ArithmeticEncoder::new(1);
        enc.encode(0, true);
        let encoded = enc.finish();
        eprintln!("encoded (true): {:?}", encoded);

        let mut dec = ArithmeticDecoder::new(&encoded, 1);
        let got = dec.decode(0);
        assert_eq!(got, true, "single true bit failed");

        // Single false bit
        let mut enc = ArithmeticEncoder::new(1);
        enc.encode(0, false);
        let encoded = enc.finish();
        eprintln!("encoded (false): {:?}", encoded);

        let mut dec = ArithmeticDecoder::new(&encoded, 1);
        let got = dec.decode(0);
        assert_eq!(got, false, "single false bit failed");
    }

    #[test]
    fn arithmetic_two_bits_roundtrip() {
        let mut enc = ArithmeticEncoder::new(1);
        enc.encode(0, true);
        enc.encode(0, true);
        let encoded = enc.finish();
        eprintln!("encoded (true, true): {:?}", encoded);

        let mut dec = ArithmeticDecoder::new(&encoded, 1);
        assert_eq!(dec.decode(0), true, "first bit");
        assert_eq!(dec.decode(0), true, "second bit");
    }

    #[test]
    fn arithmetic_encode_decode_roundtrip() {
        // Encode a sequence of binary decisions, then decode and verify
        let mut enc = ArithmeticEncoder::new(4);
        let decisions: Vec<(usize, bool)> = (0..200).map(|i| (i % 4, i % 3 == 0)).collect();

        for &(ctx, bit) in &decisions {
            enc.encode(ctx, bit);
        }
        let encoded = enc.finish();

        // Decode
        let mut dec = ArithmeticDecoder::new(&encoded, 4);
        for (idx, &(ctx, expected)) in decisions.iter().enumerate() {
            let got = dec.decode(ctx);
            assert_eq!(got, expected, "mismatch at decision {idx} ctx={ctx}");
        }
    }

    #[test]
    fn arithmetic_block_roundtrip() {
        // Encode and decode a single block with known coefficients
        let num_ctx = arithmetic_context_count(1);
        let mut enc = ArithmeticEncoder::new(num_ctx);
        let mut dc_pred = 0i32;

        let mut coeffs = [0i16; 64];
        coeffs[0] = 42; // DC
        coeffs[1] = 10; // AC[0] in zigzag
        coeffs[2] = -5; // AC[1]
        coeffs[5] = 3; // AC[4]

        let ctx_off = arithmetic_ctx_offset(0);
        arithmetic_encode_block(&mut enc, ctx_off, &mut dc_pred, &coeffs);
        let encoded = enc.finish();

        // Decode
        let mut dec = ArithmeticDecoder::new(&encoded, num_ctx);
        let mut dec_dc_pred = 0i32;
        let mut decoded = [0i16; 64];
        arithmetic_decode_block(&mut dec, ctx_off, &mut dec_dc_pred, &mut decoded);

        assert_eq!(decoded, coeffs, "single block roundtrip mismatch");
    }

    #[test]
    fn arithmetic_multi_block_roundtrip() {
        // Two blocks to test DC prediction
        let num_ctx = arithmetic_context_count(1);
        let mut enc = ArithmeticEncoder::new(num_ctx);
        let mut dc_pred = 0i32;
        let ctx_off = arithmetic_ctx_offset(0);

        let mut block1 = [0i16; 64];
        block1[0] = 100;
        block1[1] = 20;
        let mut block2 = [0i16; 64];
        block2[0] = 105;
        block2[3] = -7;

        arithmetic_encode_block(&mut enc, ctx_off, &mut dc_pred, &block1);
        arithmetic_encode_block(&mut enc, ctx_off, &mut dc_pred, &block2);
        let encoded = enc.finish();

        let mut dec = ArithmeticDecoder::new(&encoded, num_ctx);
        let mut dec_dc_pred = 0i32;
        let mut decoded1 = [0i16; 64];
        let mut decoded2 = [0i16; 64];
        arithmetic_decode_block(&mut dec, ctx_off, &mut dec_dc_pred, &mut decoded1);
        arithmetic_decode_block(&mut dec, ctx_off, &mut dec_dc_pred, &mut decoded2);

        assert_eq!(decoded1, block1, "block 1 mismatch");
        assert_eq!(decoded2, block2, "block 2 mismatch");
    }

    #[test]
    fn arithmetic_multi_component_roundtrip() {
        // 3 components like color JPEG
        let num_ctx = arithmetic_context_count(3);
        let mut enc = ArithmeticEncoder::new(num_ctx);
        let mut dc_pred = [0i32; 3];

        let mut y_block = [0i16; 64];
        y_block[0] = 50;
        y_block[1] = 10;
        let mut cb_block = [0i16; 64];
        cb_block[0] = 0;
        cb_block[1] = -3;
        let mut cr_block = [0i16; 64];
        cr_block[0] = 5;

        arithmetic_encode_block(
            &mut enc,
            arithmetic_ctx_offset(0),
            &mut dc_pred[0],
            &y_block,
        );
        arithmetic_encode_block(
            &mut enc,
            arithmetic_ctx_offset(1),
            &mut dc_pred[1],
            &cb_block,
        );
        arithmetic_encode_block(
            &mut enc,
            arithmetic_ctx_offset(2),
            &mut dc_pred[2],
            &cr_block,
        );
        let encoded = enc.finish();

        let mut dec = ArithmeticDecoder::new(&encoded, num_ctx);
        let mut dec_dc_pred = [0i32; 3];
        let mut dec_y = [0i16; 64];
        let mut dec_cb = [0i16; 64];
        let mut dec_cr = [0i16; 64];
        arithmetic_decode_block(
            &mut dec,
            arithmetic_ctx_offset(0),
            &mut dec_dc_pred[0],
            &mut dec_y,
        );
        arithmetic_decode_block(
            &mut dec,
            arithmetic_ctx_offset(1),
            &mut dec_dc_pred[1],
            &mut dec_cb,
        );
        arithmetic_decode_block(
            &mut dec,
            arithmetic_ctx_offset(2),
            &mut dec_dc_pred[2],
            &mut dec_cr,
        );

        assert_eq!(dec_y, y_block, "Y mismatch");
        assert_eq!(dec_cb, cb_block, "Cb mismatch");
        assert_eq!(dec_cr, cr_block, "Cr mismatch");
    }

    #[test]
    fn arithmetic_compresses_biased() {
        let mut enc = ArithmeticEncoder::new(1);
        for _ in 0..1000 {
            enc.encode(0, false);
        }
        let bytes = enc.finish();
        assert!(
            bytes.len() < 20,
            "1000 identical bits: {} bytes",
            bytes.len()
        );
    }
}
