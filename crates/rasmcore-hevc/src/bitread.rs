//! HEVC bitstream reading helpers — Exp-Golomb and fixed-length readers.
//!
//! Wraps rasmcore-bitio BitReader with HEVC-specific decoding functions.

use rasmcore_bitio::{BitOrder, BitReader};

use crate::error::HevcError;

/// HEVC bitstream reader wrapping BitReader with Exp-Golomb support.
pub struct HevcBitReader<'a> {
    inner: BitReader<'a>,
}

impl<'a> HevcBitReader<'a> {
    /// Create a new HEVC bit reader over RBSP data.
    pub fn new(rbsp: &'a [u8]) -> Self {
        Self {
            inner: BitReader::new(rbsp, BitOrder::MsbFirst),
        }
    }

    /// Read `n` bits as unsigned integer u(n).
    pub fn read_u(&mut self, n: u8) -> Result<u32, HevcError> {
        self.inner.read_bits(n).ok_or(HevcError::Truncated {
            expected: n as usize,
            available: 0,
        })
    }

    /// Read a single bit as boolean flag u(1).
    pub fn read_flag(&mut self) -> Result<bool, HevcError> {
        self.inner.read_bit().ok_or(HevcError::Truncated {
            expected: 1,
            available: 0,
        })
    }

    /// Read unsigned Exp-Golomb coded value ue(v).
    ///
    /// Format: leading zeros, then 1, then value bits.
    /// Value = 2^leadingZeros - 1 + read_bits(leadingZeros)
    pub fn read_ue(&mut self) -> Result<u32, HevcError> {
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_flag()?;
            if bit {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(HevcError::InvalidParameterSet(
                    "Exp-Golomb leading zeros > 31".into(),
                ));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let suffix = self.read_u(leading_zeros as u8)?;
        Ok((1u32 << leading_zeros) - 1 + suffix)
    }

    /// Read signed Exp-Golomb coded value se(v).
    ///
    /// Maps ue(v) values to signed: 0→0, 1→1, 2→-1, 3→2, 4→-2, ...
    pub fn read_se(&mut self) -> Result<i32, HevcError> {
        let val = self.read_ue()?;
        if val == 0 {
            Ok(0)
        } else if val & 1 != 0 {
            // Odd: positive
            Ok(val.div_ceil(2) as i32)
        } else {
            // Even: negative
            Ok(-((val / 2) as i32))
        }
    }

    /// Skip `n` bits.
    pub fn skip(&mut self, n: u32) -> Result<(), HevcError> {
        for _ in 0..n {
            self.read_flag()?;
        }
        Ok(())
    }

    /// Get the underlying BitReader (for advanced use).
    pub fn inner(&self) -> &BitReader<'a> {
        &self.inner
    }

    /// Get mutable access to the underlying BitReader.
    pub fn inner_mut(&mut self) -> &mut BitReader<'a> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue_zero() {
        // ue(0) = 1 (single bit '1')
        let data = [0x80]; // 1000_0000
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
    }

    #[test]
    fn ue_one() {
        // ue(1) = 010 (1 leading zero, then 1, then 0)
        let data = [0x40]; // 0100_0000
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 1);
    }

    #[test]
    fn ue_two() {
        // ue(2) = 011 (1 leading zero, then 1, then 1)
        let data = [0x60]; // 0110_0000
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 2);
    }

    #[test]
    fn ue_three() {
        // ue(3) = 00100
        let data = [0x20]; // 0010_0000
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 3);
    }

    #[test]
    fn ue_four() {
        // ue(4) = 00101
        let data = [0x28]; // 0010_1000
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 4);
    }

    #[test]
    fn ue_large() {
        // ue(14) = 000 1 111 (3 leading zeros, 1, then 111 = 7, value = 2^3-1+7 = 14)
        // Bits: 0001111x = 0001_1110 = 0x1E
        let data = [0x1E];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 14);
    }

    #[test]
    fn se_zero() {
        let data = [0x80];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 0);
    }

    #[test]
    fn se_positive_one() {
        // se(1) = ue(1) = 010 -> +1
        let data = [0x40];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 1);
    }

    #[test]
    fn se_negative_one() {
        // se(-1) = ue(2) = 011 -> -1
        let data = [0x60];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), -1);
    }

    #[test]
    fn se_positive_two() {
        // se(2) = ue(3) = 00100 -> +2
        let data = [0x20];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 2);
    }

    #[test]
    fn se_negative_two() {
        // se(-2) = ue(4) = 00101 -> -2
        let data = [0x28];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_se().unwrap(), -2);
    }

    #[test]
    fn read_u_and_flag() {
        let data = [0b1010_1100, 0b0000_0000];
        let mut r = HevcBitReader::new(&data);
        assert!(r.read_flag().unwrap()); // 1
        assert!(!r.read_flag().unwrap()); // 0
        assert_eq!(r.read_u(4).unwrap(), 0b1011); // 1011
        assert_eq!(r.read_u(2).unwrap(), 0b00); // 00
    }

    #[test]
    fn multiple_ue_sequential() {
        // ue(0)=1, ue(1)=010, ue(0)=1 -> 1 010 1 = 0b10101000
        let data = [0b1010_1000];
        let mut r = HevcBitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.read_ue().unwrap(), 1);
        assert_eq!(r.read_ue().unwrap(), 0);
    }
}
