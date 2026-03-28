//! Bit-level reader with configurable bit ordering.

use crate::BitOrder;

/// Bit-level reader that consumes bits from a byte slice.
///
/// Uses a u64 accumulator for efficient multi-bit reads with
/// minimal per-read overhead.
///
/// # Example
///
/// ```
/// use rasmcore_bitio::{BitReader, BitOrder};
///
/// let data = vec![0xFF, 0x00];
/// let mut r = BitReader::new(&data, BitOrder::MsbFirst);
/// assert_eq!(r.read_bits(8).unwrap(), 0xFF);
/// assert_eq!(r.read_bit().unwrap(), false);
/// ```
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Current byte position in data.
    pos: usize,
    /// Bit accumulator.
    acc: u64,
    /// Number of valid bits in accumulator.
    nbits: u8,
    /// Bit ordering.
    order: BitOrder,
}

impl<'a> BitReader<'a> {
    /// Create a new reader over the given byte slice.
    pub fn new(data: &'a [u8], order: BitOrder) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
            order,
        }
    }

    /// Read `n` bits (1 <= n <= 32) and return as u32.
    ///
    /// Returns `None` if not enough bits remain.
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!((1..=32).contains(&n), "n must be 1..=32");

        // Refill accumulator if needed
        while self.nbits < n {
            if self.pos >= self.data.len() {
                if self.nbits >= n {
                    break;
                }
                return None;
            }
            self.refill_byte();
        }

        let value = match self.order {
            BitOrder::MsbFirst => {
                // Extract from the top of the accumulator
                let v = (self.acc >> (64 - n as u32)) as u32;
                let mask = if n == 32 { !0u32 } else { (1u32 << n) - 1 };
                self.acc <<= n;
                self.nbits -= n;
                v & mask
            }
            BitOrder::LsbFirst => {
                // Extract from the bottom of the accumulator
                let mask = if n == 32 { !0u32 } else { (1u32 << n) - 1 };
                let v = (self.acc as u32) & mask;
                self.acc >>= n;
                self.nbits -= n;
                v
            }
        };

        Some(value)
    }

    /// Read a single bit. Returns `None` if no bits remain.
    #[inline]
    pub fn read_bit(&mut self) -> Option<bool> {
        self.read_bits(1).map(|v| v != 0)
    }

    /// Peek at the next `n` bits without consuming them.
    ///
    /// Returns `None` if not enough bits remain.
    pub fn peek_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!((1..=32).contains(&n), "n must be 1..=32");

        // Refill if needed
        while self.nbits < n {
            if self.pos >= self.data.len() {
                if self.nbits >= n {
                    break;
                }
                return None;
            }
            self.refill_byte();
        }

        let value = match self.order {
            BitOrder::MsbFirst => {
                let v = (self.acc >> (64 - n as u32)) as u32;
                let mask = if n == 32 { !0u32 } else { (1u32 << n) - 1 };
                v & mask
            }
            BitOrder::LsbFirst => {
                let mask = if n == 32 { !0u32 } else { (1u32 << n) - 1 };
                (self.acc as u32) & mask
            }
        };

        Some(value)
    }

    /// Skip `n` bits without returning them.
    pub fn skip_bits(&mut self, n: u8) -> bool {
        self.read_bits(n).is_some()
    }

    /// Align to the next byte boundary by discarding remaining bits in current byte.
    pub fn align_to_byte(&mut self) {
        if !self.nbits.is_multiple_of(8) {
            let discard = self.nbits % 8;
            match self.order {
                BitOrder::MsbFirst => self.acc <<= discard,
                BitOrder::LsbFirst => self.acc >>= discard,
            }
            self.nbits -= discard;
        }
    }

    /// Total number of bits consumed so far.
    pub fn bits_read(&self) -> u64 {
        self.pos as u64 * 8 - self.nbits as u64
    }

    /// Check if all data has been consumed.
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len() && self.nbits == 0
    }

    /// Number of bits remaining (approximate — may include padding).
    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() - self.pos) as u64 * 8 + self.nbits as u64
    }

    /// Refill one byte from the data slice into the accumulator.
    #[inline]
    fn refill_byte(&mut self) {
        if self.pos < self.data.len() {
            let byte = self.data[self.pos] as u64;
            self.pos += 1;
            match self.order {
                BitOrder::MsbFirst => {
                    self.acc |= byte << (56 - self.nbits as u32);
                }
                BitOrder::LsbFirst => {
                    self.acc |= byte << self.nbits;
                }
            }
            self.nbits += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_read_byte() {
        let data = [0xAB];
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
        assert!(r.read_bit().is_none());
    }

    #[test]
    fn lsb_read_byte() {
        let data = [0xAB];
        let mut r = BitReader::new(&data, BitOrder::LsbFirst);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
    }

    #[test]
    fn msb_read_single_bits() {
        let data = [0b10110000]; // MSB first: 1,0,1,1,0,0,0,0
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.read_bit().unwrap(), true);
        assert_eq!(r.read_bit().unwrap(), false);
        assert_eq!(r.read_bit().unwrap(), true);
        assert_eq!(r.read_bit().unwrap(), true);
        assert_eq!(r.read_bit().unwrap(), false);
        assert_eq!(r.read_bit().unwrap(), false);
        assert_eq!(r.read_bit().unwrap(), false);
        assert_eq!(r.read_bit().unwrap(), false);
    }

    #[test]
    fn lsb_read_single_bits() {
        let data = [0x0D]; // 0b00001101, LSB first: 1,0,1,1,0,0,0,0
        let mut r = BitReader::new(&data, BitOrder::LsbFirst);
        assert_eq!(r.read_bit().unwrap(), true); // bit 0
        assert_eq!(r.read_bit().unwrap(), false); // bit 1
        assert_eq!(r.read_bit().unwrap(), true); // bit 2
        assert_eq!(r.read_bit().unwrap(), true); // bit 3
        assert_eq!(r.read_bit().unwrap(), false); // bit 4
        assert_eq!(r.read_bit().unwrap(), false); // bit 5
        assert_eq!(r.read_bit().unwrap(), false); // bit 6
        assert_eq!(r.read_bit().unwrap(), false); // bit 7
    }

    #[test]
    fn msb_mixed_widths() {
        let data = [0b10100011]; // 4 bits: 1010, then 4 bits: 0011
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0011);
    }

    #[test]
    fn peek_does_not_consume() {
        let data = [0xFF];
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.peek_bits(4).unwrap(), 0xF);
        assert_eq!(r.peek_bits(4).unwrap(), 0xF); // same value
        assert_eq!(r.read_bits(4).unwrap(), 0xF); // now consumed
        assert_eq!(r.read_bits(4).unwrap(), 0xF); // second nibble
    }

    #[test]
    fn lsb_variable_width() {
        // 0xAB = 10101011, 0x01 = 00000001
        // LSB first, 9 bits: bits 0-7 from 0xAB = 0xAB, bit 8 from 0x01 = 1
        // value = 0x1AB
        let data = [0xAB, 0x01];
        let mut r = BitReader::new(&data, BitOrder::LsbFirst);
        assert_eq!(r.read_bits(9).unwrap(), 0x1AB);
    }

    #[test]
    fn read_multiple_bytes() {
        let data = [0x12, 0x34, 0x56];
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.read_bits(8).unwrap(), 0x12);
        assert_eq!(r.read_bits(8).unwrap(), 0x34);
        assert_eq!(r.read_bits(8).unwrap(), 0x56);
        assert!(r.read_bit().is_none());
    }

    #[test]
    fn bits_remaining() {
        let data = [0xFF, 0x00];
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        assert_eq!(r.bits_remaining(), 16);
        r.read_bits(5).unwrap();
        assert_eq!(r.bits_remaining(), 11);
    }

    #[test]
    fn align_to_byte() {
        let data = [0xFF, 0xAA];
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        r.read_bits(3).unwrap(); // consume 3 bits
        r.align_to_byte(); // skip remaining 5 bits
        assert_eq!(r.read_bits(8).unwrap(), 0xAA); // next full byte
    }

    #[test]
    fn empty_reader() {
        let data: &[u8] = &[];
        let mut r = BitReader::new(data, BitOrder::MsbFirst);
        assert!(r.is_empty());
        assert!(r.read_bit().is_none());
    }

    #[test]
    fn large_read() {
        let data: Vec<u8> = (0..100).collect();
        let mut r = BitReader::new(&data, BitOrder::MsbFirst);
        for i in 0..100 {
            assert_eq!(r.read_bits(8).unwrap(), i as u32);
        }
        assert!(r.read_bit().is_none());
    }
}
