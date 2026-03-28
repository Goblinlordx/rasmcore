//! Boolean arithmetic encoder/decoder (RFC 6386 Section 7).
//!
//! General-purpose boolean arithmetic coder usable for VP8 and other
//! applications requiring adaptive binary entropy coding.
//!
//! The encoder follows the libvpx reference (vp8/encoder/boolhuff.c)
//! translated to safe Rust with wrapping arithmetic. The decoder uses
//! u64 value register (matching the `image-webp` crate's decoder) for
//! robust precision.

/// VP8 boolean arithmetic encoder.
///
/// Encodes a sequence of boolean values, each with an independent
/// probability, into a compact byte stream compatible with RFC 6386.
///
/// # Example
///
/// ```
/// use rasmcore_webp::boolcoder::BoolWriter;
///
/// let mut w = BoolWriter::new();
/// w.put_bit(128, true);   // 50% probability, value = true
/// w.put_bit(200, false);  // ~78% probability, value = false
/// w.put_literal(8, 42);   // 8-bit literal value
/// let bytes = w.finish();
/// ```
pub struct BoolWriter {
    buf: Vec<u8>,
    range: u32,
    lowvalue: u32,
    count: i32,
}

impl BoolWriter {
    /// Create a new encoder with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create a new encoder with pre-allocated buffer capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            range: 255,
            lowvalue: 0,
            count: -24,
        }
    }

    /// Encode a single boolean with the given probability.
    ///
    /// `prob` is in 1..=255, representing P(false) scaled to 0..256.
    pub fn put_bit(&mut self, prob: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);

        if bit {
            self.lowvalue = self.lowvalue.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }

        let shift = self.range.leading_zeros().saturating_sub(24);
        self.range <<= shift;
        self.count += shift as i32;

        if self.count >= 0 {
            let offset = (shift as i32 - self.count) as u32;

            if offset > 0 && (self.lowvalue.wrapping_shl(offset - 1) & 0x8000_0000) != 0 {
                // Carry propagation
                let mut i = self.buf.len();
                while i > 0 {
                    i -= 1;
                    if self.buf[i] == 0xFF {
                        self.buf[i] = 0;
                    } else {
                        self.buf[i] += 1;
                        break;
                    }
                }
            }

            self.buf.push(self.lowvalue.wrapping_shr(24 - offset) as u8);
            self.lowvalue = self.lowvalue.wrapping_shl(offset);
            // Remaining shift after byte extraction
            let remaining = self.count as u32;
            self.lowvalue &= 0x00FF_FFFF;
            self.count -= 8;

            self.lowvalue = self.lowvalue.wrapping_shl(remaining);
        } else {
            self.lowvalue = self.lowvalue.wrapping_shl(shift);
        }
    }

    /// Encode a boolean with uniform (50%) probability.
    pub fn put_bit_uniform(&mut self, bit: bool) {
        self.put_bit(128, bit);
    }

    /// Encode a multi-bit unsigned literal, MSB first.
    pub fn put_literal(&mut self, n_bits: u8, value: u32) {
        for i in (0..n_bits).rev() {
            self.put_bit(128, (value >> i) & 1 != 0);
        }
    }

    /// Encode a signed value: non-zero flag, magnitude, sign bit.
    pub fn put_signed(&mut self, n_bits: u8, value: i32) {
        if value == 0 {
            self.put_bit(128, false);
        } else {
            self.put_bit(128, true);
            let mag = value.unsigned_abs();
            for i in (0..n_bits).rev() {
                self.put_bit(128, (mag >> i) & 1 != 0);
            }
            self.put_bit(128, value < 0);
        }
    }

    /// Return an estimate of the current encoded size in bytes.
    pub fn size_estimate(&self) -> usize {
        self.buf.len() + ((24 + self.count) >> 3).max(0) as usize
    }

    /// Flush remaining state and return the encoded byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.put_bit(128, false);
        }
        // Padding for decoder lookahead
        self.buf.extend_from_slice(&[0, 0, 0, 0]);
        self.buf
    }
}

impl Default for BoolWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// VP8 boolean arithmetic decoder.
///
/// Uses u64 value register with dynamic bit_count tracking, matching
/// the approach used by the `image-webp` crate for robust precision.
pub struct BoolReader<'a> {
    data: &'a [u8],
    pos: usize,
    value: u64,
    range: u32,
    bit_count: i32,
}

impl<'a> BoolReader<'a> {
    /// Create a new decoder from encoded bytes.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            value: 0,
            range: 255,
            bit_count: -8,
        }
    }

    fn fill(&mut self) {
        // Load up to 4 bytes at once into the u64 value register
        let mut v = 0u32;
        let bytes_available = (self.data.len() - self.pos).min(4);
        for i in 0..bytes_available {
            v = (v << 8) | self.data[self.pos + i] as u32;
        }
        // Pad remaining positions with zeros
        v <<= (4 - bytes_available) as u32 * 8;
        self.pos += bytes_available.max(1); // advance at least 1 to avoid infinite loop at EOF

        self.value <<= 32;
        self.value |= v as u64;
        self.bit_count += 32;
    }

    /// Decode a single boolean with the given probability.
    pub fn get_bit(&mut self, prob: u8) -> bool {
        if self.bit_count < 0 {
            self.fill();
        }

        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let bigsplit = (split as u64) << self.bit_count;

        let bit = if self.value >= bigsplit {
            self.range -= split;
            self.value -= bigsplit;
            true
        } else {
            self.range = split;
            false
        };

        let shift = self.range.leading_zeros().saturating_sub(24);
        self.range <<= shift;
        self.bit_count -= shift as i32;

        bit
    }

    /// Decode a boolean with uniform (50%) probability.
    pub fn get_bit_uniform(&mut self) -> bool {
        self.get_bit(128)
    }

    /// Decode a multi-bit unsigned literal, MSB first.
    pub fn get_literal(&mut self, n_bits: u8) -> u32 {
        let mut val = 0u32;
        for _ in 0..n_bits {
            val = (val << 1) | self.get_bit(128) as u32;
        }
        val
    }

    /// Decode a signed value: non-zero flag, magnitude, sign bit.
    pub fn get_signed(&mut self, n_bits: u8) -> i32 {
        if !self.get_bit(128) {
            return 0;
        }
        let mag = self.get_literal(n_bits) as i32;
        if self.get_bit(128) { -mag } else { mag }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_single_bit_true() {
        let mut w = BoolWriter::new();
        w.put_bit(128, true);
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        assert!(r.get_bit(128));
    }

    #[test]
    fn roundtrip_single_bit_false() {
        let mut w = BoolWriter::new();
        w.put_bit(128, false);
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        assert!(!r.get_bit(128));
    }

    #[test]
    fn roundtrip_mixed_bits() {
        let bits = [true, false, true, true, false, false, true, false];
        let probs = [128u8, 200, 50, 255, 1, 128, 100, 250];
        let mut w = BoolWriter::new();
        for (&bit, &prob) in bits.iter().zip(probs.iter()) {
            w.put_bit(prob, bit);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for (&bit, &prob) in bits.iter().zip(probs.iter()) {
            assert_eq!(r.get_bit(prob), bit, "mismatch at prob={prob}");
        }
    }

    #[test]
    fn roundtrip_literal() {
        let mut w = BoolWriter::new();
        w.put_literal(8, 42);
        w.put_literal(16, 1234);
        w.put_literal(4, 15);
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        assert_eq!(r.get_literal(8), 42);
        assert_eq!(r.get_literal(16), 1234);
        assert_eq!(r.get_literal(4), 15);
    }

    #[test]
    fn roundtrip_signed() {
        let mut w = BoolWriter::new();
        w.put_signed(8, 0);
        w.put_signed(8, 100);
        w.put_signed(8, -50);
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        assert_eq!(r.get_signed(8), 0);
        assert_eq!(r.get_signed(8), 100);
        assert_eq!(r.get_signed(8), -50);
    }

    #[test]
    fn determinism() {
        let encode = || {
            let mut w = BoolWriter::new();
            for i in 0..256 {
                w.put_bit((i as u8).wrapping_add(1), i % 3 == 0);
            }
            w.put_literal(16, 0xBEEF);
            w.put_signed(8, -42);
            w.finish()
        };
        assert_eq!(
            encode(),
            encode(),
            "determinism: two identical encodes differ"
        );
    }

    #[test]
    fn all_zeros() {
        let mut w = BoolWriter::new();
        for _ in 0..1000 {
            w.put_bit(128, false);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for i in 0..1000 {
            assert!(!r.get_bit(128), "expected false at bit {i}");
        }
    }

    #[test]
    fn all_ones() {
        let mut w = BoolWriter::new();
        for _ in 0..1000 {
            w.put_bit(128, true);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for i in 0..1000 {
            assert!(r.get_bit(128), "expected true at bit {i}");
        }
    }

    #[test]
    fn alternating() {
        let mut w = BoolWriter::new();
        for i in 0..1000 {
            w.put_bit(128, i % 2 == 0);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for i in 0..1000 {
            assert_eq!(r.get_bit(128), i % 2 == 0, "mismatch at bit {i}");
        }
    }

    #[test]
    fn probability_extremes() {
        let mut w = BoolWriter::new();
        for _ in 0..100 {
            w.put_bit(1, true);
        }
        for _ in 0..100 {
            w.put_bit(1, false);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for i in 0..100 {
            assert!(r.get_bit(1), "expected true at {i}");
        }
        for i in 0..100 {
            assert!(!r.get_bit(1), "expected false at {i}");
        }

        let mut w = BoolWriter::new();
        for _ in 0..100 {
            w.put_bit(254, false);
        }
        for _ in 0..100 {
            w.put_bit(254, true);
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for _ in 0..100 {
            assert!(!r.get_bit(254));
        }
        for _ in 0..100 {
            assert!(r.get_bit(254));
        }
    }

    #[test]
    fn varying_probabilities() {
        let mut w = BoolWriter::new();
        let mut expected = Vec::new();
        for i in 0..500 {
            let prob = ((i * 7 + 13) % 254 + 1) as u8;
            let bit = i % 5 < 3;
            w.put_bit(prob, bit);
            expected.push((prob, bit));
        }
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        for (i, &(prob, bit)) in expected.iter().enumerate() {
            assert_eq!(r.get_bit(prob), bit, "mismatch at index {i}, prob={prob}");
        }
    }

    #[test]
    fn size_estimate_grows() {
        let mut w = BoolWriter::new();
        let s0 = w.size_estimate();
        for _ in 0..1000 {
            w.put_bit(128, true);
        }
        let s1 = w.size_estimate();
        assert!(s1 > s0, "size should grow after encoding bits");
    }

    #[test]
    fn with_capacity_works() {
        let mut w = BoolWriter::with_capacity(1024);
        w.put_literal(8, 255);
        let bytes = w.finish();
        let mut r = BoolReader::new(&bytes);
        assert_eq!(r.get_literal(8), 255);
    }

    #[test]
    fn performance_1m_booleans() {
        let start = std::time::Instant::now();
        let mut w = BoolWriter::with_capacity(256 * 1024);
        for i in 0..1_000_000u32 {
            let prob = ((i % 254) + 1) as u8;
            w.put_bit(prob, i & 1 == 0);
        }
        let bytes = w.finish();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 1000,
            "encoding 1M booleans took {elapsed:?}, expected < 1s"
        );
        assert!(!bytes.is_empty());

        // Verify full roundtrip of first 10000
        let mut r = BoolReader::new(&bytes);
        for i in 0..10_000u32 {
            let prob = ((i % 254) + 1) as u8;
            let expected = i & 1 == 0;
            assert_eq!(r.get_bit(prob), expected, "roundtrip fail at {i}");
        }
    }
}
