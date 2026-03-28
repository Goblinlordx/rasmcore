//! Bit-level writer with configurable bit ordering.

use crate::BitOrder;

/// Bit-level writer that accumulates bits and flushes to a byte buffer.
///
/// Uses a u64 accumulator for batch flushing — up to 8 bytes written per
/// flush, minimizing per-bit overhead.
///
/// # Example
///
/// ```
/// use rasmcore_bitio::{BitWriter, BitOrder};
///
/// let mut w = BitWriter::new(BitOrder::MsbFirst);
/// w.write_bits(8, 0xFF);  // write 8 bits
/// w.write_bit(true);      // write single bit
/// let bytes = w.finish();
/// ```
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bit accumulator — bits are staged here before flushing to buf.
    acc: u64,
    /// Number of valid bits in the accumulator (0..64).
    nbits: u8,
    /// Bit ordering.
    order: BitOrder,
}

impl BitWriter {
    /// Create a new writer with the given bit ordering.
    pub fn new(order: BitOrder) -> Self {
        Self::with_capacity(order, 256)
    }

    /// Create a new writer with pre-allocated buffer capacity.
    pub fn with_capacity(order: BitOrder, capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            acc: 0,
            nbits: 0,
            order,
        }
    }

    /// Write `n` bits from the lower `n` bits of `value` (1 <= n <= 32).
    ///
    /// For MSB-first: the most significant of the `n` bits is written first.
    /// For LSB-first: the least significant of the `n` bits is written first.
    #[inline]
    pub fn write_bits(&mut self, n: u8, value: u32) {
        debug_assert!((1..=32).contains(&n), "n must be 1..=32");
        let mask = if n == 32 { !0u32 } else { (1u32 << n) - 1 };
        let masked = value & mask;

        match self.order {
            BitOrder::MsbFirst => {
                // Pack bits left-aligned in accumulator, MSB first.
                // Accumulator layout: [MSB ... bits ... LSB ... free space]
                self.acc |= (masked as u64) << (64 - self.nbits as u32 - n as u32);
            }
            BitOrder::LsbFirst => {
                // Pack bits right-aligned, LSB first.
                // Accumulator layout: [free space ... bits ... LSB]
                self.acc |= (masked as u64) << self.nbits;
            }
        }
        self.nbits += n;

        // Flush whole bytes when accumulator has >= 8 bytes
        if self.nbits >= 56 {
            self.flush_bytes();
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(1, bit as u32);
    }

    /// Write a full byte (byte-aligned or not).
    #[inline]
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(8, byte as u32);
    }

    /// Write raw bytes directly, flushing any partial bits first.
    /// This ensures byte alignment before writing.
    pub fn write_bytes_aligned(&mut self, bytes: &[u8]) {
        self.align_to_byte();
        self.flush_all();
        self.buf.extend_from_slice(bytes);
    }

    /// Pad to the next byte boundary with zero bits.
    pub fn align_to_byte(&mut self) {
        if !self.nbits.is_multiple_of(8) {
            let pad = 8 - (self.nbits % 8);
            self.write_bits(pad, 0);
        }
    }

    /// Total number of bits written so far (including unflushed).
    pub fn bit_position(&self) -> u64 {
        self.buf.len() as u64 * 8 + self.nbits as u64
    }

    /// Total bytes in the output buffer (not counting unflushed accumulator bits).
    pub fn byte_len(&self) -> usize {
        self.buf.len()
    }

    /// Finalize and return the byte buffer. Flushes remaining bits.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.flush_all();
        self.buf
    }

    /// Flush complete bytes from accumulator to buffer.
    fn flush_bytes(&mut self) {
        match self.order {
            BitOrder::MsbFirst => {
                while self.nbits >= 8 {
                    let byte = (self.acc >> 56) as u8;
                    self.buf.push(byte);
                    self.acc <<= 8;
                    self.nbits -= 8;
                }
            }
            BitOrder::LsbFirst => {
                while self.nbits >= 8 {
                    let byte = (self.acc & 0xFF) as u8;
                    self.buf.push(byte);
                    self.acc >>= 8;
                    self.nbits -= 8;
                }
            }
        }
    }

    /// Flush all remaining bits (including partial last byte).
    fn flush_all(&mut self) {
        match self.order {
            BitOrder::MsbFirst => {
                while self.nbits > 0 {
                    let byte = (self.acc >> 56) as u8;
                    self.buf.push(byte);
                    self.acc <<= 8;
                    self.nbits = self.nbits.saturating_sub(8);
                }
            }
            BitOrder::LsbFirst => {
                while self.nbits > 0 {
                    let byte = (self.acc & 0xFF) as u8;
                    self.buf.push(byte);
                    self.acc >>= 8;
                    self.nbits = self.nbits.saturating_sub(8);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_write_byte() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        w.write_bits(8, 0xAB);
        let out = w.finish();
        assert_eq!(out, vec![0xAB]);
    }

    #[test]
    fn lsb_write_byte() {
        let mut w = BitWriter::new(BitOrder::LsbFirst);
        w.write_bits(8, 0xAB);
        let out = w.finish();
        assert_eq!(out, vec![0xAB]);
    }

    #[test]
    fn msb_write_single_bits() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        // Write 10110000 = 0xB0
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(false);
        w.write_bit(false);
        w.write_bit(false);
        let out = w.finish();
        assert_eq!(out, vec![0xB0]);
    }

    #[test]
    fn lsb_write_single_bits() {
        let mut w = BitWriter::new(BitOrder::LsbFirst);
        // LSB first: bits 1,0,1,1,0,0,0,0 → byte 0b00001101 = 0x0D
        w.write_bit(true); // bit 0
        w.write_bit(false); // bit 1
        w.write_bit(true); // bit 2
        w.write_bit(true); // bit 3
        w.write_bit(false); // bit 4
        w.write_bit(false); // bit 5
        w.write_bit(false); // bit 6
        w.write_bit(false); // bit 7
        let out = w.finish();
        assert_eq!(out, vec![0x0D]);
    }

    #[test]
    fn msb_mixed_widths() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        w.write_bits(4, 0b1010); // 1010....
        w.write_bits(4, 0b0011); // ....0011
        let out = w.finish();
        assert_eq!(out, vec![0b10100011]);
    }

    #[test]
    fn write_multiple_bytes() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        w.write_bits(8, 0x12);
        w.write_bits(8, 0x34);
        w.write_bits(8, 0x56);
        let out = w.finish();
        assert_eq!(out, vec![0x12, 0x34, 0x56]);
    }

    #[test]
    fn partial_byte_padding() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        w.write_bits(3, 0b101); // 101.....
        let out = w.finish();
        assert_eq!(out, vec![0b10100000]);
    }

    #[test]
    fn bytes_aligned_write() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        w.write_bits(4, 0b1111);
        w.write_bytes_aligned(&[0xAA, 0xBB]);
        let out = w.finish();
        // 4 bits → padded to byte (0xF0), then aligned bytes
        assert_eq!(out, vec![0xF0, 0xAA, 0xBB]);
    }

    #[test]
    fn bit_position_tracking() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        assert_eq!(w.bit_position(), 0);
        w.write_bits(5, 0);
        assert_eq!(w.bit_position(), 5);
        w.write_bits(8, 0);
        assert_eq!(w.bit_position(), 13);
    }

    #[test]
    fn large_write_flushes_correctly() {
        let mut w = BitWriter::new(BitOrder::MsbFirst);
        for i in 0..100u32 {
            w.write_bits(8, i & 0xFF);
        }
        let out = w.finish();
        assert_eq!(out.len(), 100);
        for (i, &byte) in out.iter().enumerate() {
            assert_eq!(byte, (i & 0xFF) as u8);
        }
    }

    #[test]
    fn lsb_variable_width() {
        let mut w = BitWriter::new(BitOrder::LsbFirst);
        // Write 9-bit value 0x1AB (binary: 1_1010_1011)
        // LSB first: bits 0-7 = 0xAB, bit 8 = 1
        w.write_bits(9, 0x1AB);
        let out = w.finish();
        // Byte 0: lower 8 bits = 0xAB
        // Byte 1: bit 8 = 1, padded = 0x01
        assert_eq!(out, vec![0xAB, 0x01]);
    }

    #[test]
    fn empty_writer() {
        let w = BitWriter::new(BitOrder::MsbFirst);
        let out = w.finish();
        assert!(out.is_empty());
    }
}
