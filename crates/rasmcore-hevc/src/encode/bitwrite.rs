//! HEVC RBSP bitstream writer — write bits, Exp-Golomb codes, and flags.
//!
//! Direct port of x265 4.1 common/bitstream.cpp Bitstream class and
//! encoder/entropy.cpp SyntaxElementWriter methods.
//!
//! This is the write counterpart to `bitread::HevcBitReader`.
//!
//! Ref: x265 4.1 common/bitstream.h — Bitstream class
//! Ref: ITU-T H.265 Section 7.2 — RBSP syntax structures

/// HEVC bitstream writer for RBSP (Raw Byte Sequence Payload) data.
///
/// Accumulates bits in a buffer, MSB-first. Supports fixed-length writes,
/// Exp-Golomb coded values, and byte-aligned output.
///
/// Ref: x265 4.1 common/bitstream.cpp
pub struct BitstreamWriter {
    /// Output byte buffer.
    buf: Vec<u8>,
    /// Current byte being accumulated (MSB-first).
    held_bits: u32,
    /// Number of valid bits in `held_bits` (0..8).
    num_held_bits: u8,
}

impl BitstreamWriter {
    /// Create a new bitstream writer with the given initial capacity.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(1024),
            held_bits: 0,
            num_held_bits: 0,
        }
    }

    /// Create a new bitstream writer with a specific capacity hint.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            held_bits: 0,
            num_held_bits: 0,
        }
    }

    /// Write `num_bits` bits from `value` (MSB-first).
    ///
    /// Ref: x265 4.1 common/bitstream.cpp Bitstream::write()
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);
        if num_bits == 0 {
            return;
        }

        // Mask off unused high bits
        let val = if num_bits < 32 {
            value & ((1u32 << num_bits) - 1)
        } else {
            value
        };

        let total_bits = self.num_held_bits + num_bits;

        if total_bits < 8 {
            // All bits fit in the current held byte
            self.held_bits |= val << (8 - total_bits);
            self.num_held_bits = total_bits;
        } else {
            // Flush complete bytes
            // First, fill the current held byte
            let bits_to_fill = 8 - self.num_held_bits;
            let first_byte = self.held_bits | (val >> (num_bits - bits_to_fill));
            self.buf.push(first_byte as u8);

            // Write remaining complete bytes
            let mut remaining_bits = num_bits - bits_to_fill;
            while remaining_bits >= 8 {
                remaining_bits -= 8;
                self.buf.push((val >> remaining_bits) as u8);
            }

            // Store leftover bits
            self.held_bits = if remaining_bits > 0 {
                (val & ((1u32 << remaining_bits) - 1)) << (8 - remaining_bits)
            } else {
                0
            };
            self.num_held_bits = remaining_bits;
        }
    }

    /// Write a single bit (0 or 1).
    #[inline]
    pub fn write_flag(&mut self, flag: bool) {
        self.write_bits(flag as u32, 1);
    }

    /// Write an unsigned Exp-Golomb coded value ue(v).
    ///
    /// Format: (leading_zeros) 1 (suffix_bits)
    /// where value = 2^leading_zeros - 1 + suffix
    ///
    /// Ref: ITU-T H.265 Section 9.2 — Exp-Golomb coding
    pub fn write_ue(&mut self, value: u32) {
        let val_plus1 = value + 1;
        // Number of bits needed = floor(log2(val+1)) + 1
        let num_bits = 32 - val_plus1.leading_zeros(); // 1-based
        let leading_zeros = num_bits - 1;

        // Write leading_zeros zeros, then the (leading_zeros + 1)-bit value
        // Total bits = 2 * leading_zeros + 1
        self.write_bits(0, leading_zeros as u8); // leading zeros
        self.write_bits(val_plus1, num_bits as u8); // 1 followed by suffix
    }

    /// Write a signed Exp-Golomb coded value se(v).
    ///
    /// Maps signed to unsigned: 0→0, 1→1, -1→2, 2→3, -2→4, ...
    ///
    /// Ref: ITU-T H.265 Section 9.2.1
    pub fn write_se(&mut self, value: i32) {
        let mapped = if value <= 0 {
            (-2 * value) as u32
        } else {
            (2 * value - 1) as u32
        };
        self.write_ue(mapped);
    }

    /// Write a byte-aligned byte directly (must be byte-aligned).
    pub fn write_byte(&mut self, byte: u8) {
        debug_assert_eq!(
            self.num_held_bits, 0,
            "write_byte called when not byte-aligned"
        );
        self.buf.push(byte);
    }

    /// Write raw bytes directly (must be byte-aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(
            self.num_held_bits, 0,
            "write_bytes called when not byte-aligned"
        );
        self.buf.extend_from_slice(bytes);
    }

    /// Write RBSP trailing bits — a 1 bit followed by 0 bits to byte-align.
    ///
    /// Ref: ITU-T H.265 Section 7.3.2.11
    pub fn write_rbsp_trailing_bits(&mut self) {
        self.write_flag(true); // rbsp_stop_one_bit
        while self.num_held_bits != 0 {
            self.write_flag(false); // rbsp_alignment_zero_bit
        }
    }

    /// Byte-align by writing zero bits if not already aligned.
    pub fn byte_align(&mut self) {
        if self.num_held_bits > 0 {
            self.buf.push(self.held_bits as u8);
            self.held_bits = 0;
            self.num_held_bits = 0;
        }
    }

    /// Get the number of bits written so far.
    pub fn bits_written(&self) -> usize {
        self.buf.len() * 8 + self.num_held_bits as usize
    }

    /// Get the number of complete bytes written.
    pub fn bytes_written(&self) -> usize {
        self.buf.len()
    }

    /// Consume the writer and return the RBSP byte buffer.
    ///
    /// Flushes any remaining held bits before returning.
    pub fn finish(mut self) -> Vec<u8> {
        if self.num_held_bits > 0 {
            self.buf.push(self.held_bits as u8);
            self.held_bits = 0;
            self.num_held_bits = 0;
        }
        self.buf
    }

    /// Get a reference to the current buffer (without flushing held bits).
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

impl Default for BitstreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_single_bits() {
        let mut w = BitstreamWriter::new();
        w.write_flag(true);
        w.write_flag(false);
        w.write_flag(true);
        w.write_flag(true);
        w.write_flag(false);
        w.write_flag(false);
        w.write_flag(true);
        w.write_flag(false);
        let data = w.finish();
        assert_eq!(data, vec![0b1011_0010]);
    }

    #[test]
    fn write_multi_bit_values() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0b110, 3); // 110
        w.write_bits(0b01010, 5); // 01010
        let data = w.finish();
        assert_eq!(data, vec![0b110_01010]);
    }

    #[test]
    fn write_crosses_byte_boundary() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0xFF, 8); // full byte
        w.write_bits(0x0A, 4); // half byte
        w.write_bits(0x05, 4); // half byte
        let data = w.finish();
        assert_eq!(data, vec![0xFF, 0xA5]);
    }

    #[test]
    fn write_ue_zero() {
        let mut w = BitstreamWriter::new();
        w.write_ue(0); // encoded as "1" (single bit)
        w.write_rbsp_trailing_bits();
        let data = w.finish();
        // "1" + trailing "1000000" = 0b1_1000000 = 0xC0
        assert_eq!(data, vec![0xC0]);
    }

    #[test]
    fn write_ue_values() {
        // ue(0) = 1            (1 bit)
        // ue(1) = 010          (3 bits)
        // ue(2) = 011          (3 bits)
        // ue(3) = 00100        (5 bits)
        // ue(4) = 00101        (5 bits)
        let mut w = BitstreamWriter::new();
        w.write_ue(0);
        w.write_ue(1);
        w.write_ue(2);
        w.write_ue(3);
        w.write_ue(4);
        // Total: 1 + 3 + 3 + 5 + 5 = 17 bits
        // 1_010_011_00 | 100_00101_0 (padded)
        // 1 + 3 + 3 + 5 + 5 = 17 bits
        assert_eq!(w.bits_written(), 17);
        let data = w.finish();
        // Verify by decoding
        let mut r = crate::bitread::HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.read_ue().unwrap(), 1);
        assert_eq!(r.read_ue().unwrap(), 2);
        assert_eq!(r.read_ue().unwrap(), 3);
        assert_eq!(r.read_ue().unwrap(), 4);
    }

    #[test]
    fn write_se_values() {
        let mut w = BitstreamWriter::new();
        w.write_se(0); // -> ue(0) = 1
        w.write_se(1); // -> ue(1) = 010
        w.write_se(-1); // -> ue(2) = 011
        w.write_se(2); // -> ue(3) = 00100
        w.write_se(-2); // -> ue(4) = 00101
        let data = w.finish();
        let mut r = crate::bitread::HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 0);
        assert_eq!(r.read_se().unwrap(), 1);
        assert_eq!(r.read_se().unwrap(), -1);
        assert_eq!(r.read_se().unwrap(), 2);
        assert_eq!(r.read_se().unwrap(), -2);
    }

    #[test]
    fn write_rbsp_trailing_bits_aligned() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0xAB, 8);
        w.write_rbsp_trailing_bits();
        let data = w.finish();
        assert_eq!(data, vec![0xAB, 0x80]); // 1000_0000
    }

    #[test]
    fn write_rbsp_trailing_bits_unaligned() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0b101, 3);
        w.write_rbsp_trailing_bits(); // 1 + 4 zeros = 10000
        let data = w.finish();
        // 101_10000 = 0xB0
        assert_eq!(data, vec![0xB0]);
    }

    #[test]
    fn write_32_bits() {
        let mut w = BitstreamWriter::new();
        w.write_bits(0xDEADBEEF, 32);
        let data = w.finish();
        assert_eq!(data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn roundtrip_large_ue() {
        let mut w = BitstreamWriter::new();
        for val in [0, 1, 7, 15, 100, 255, 1000, 65535] {
            w.write_ue(val);
        }
        let data = w.finish();
        let mut r = crate::bitread::HevcBitReader::new(&data);
        for val in [0u32, 1, 7, 15, 100, 255, 1000, 65535] {
            assert_eq!(r.read_ue().unwrap(), val, "ue roundtrip failed for {val}");
        }
    }
}
