//! CABAC arithmetic decoder engine (ITU-T H.265 Section 9.3.3).

use super::context::ContextModel;
use super::tables::RANGE_TAB_LPS;
use crate::error::HevcError;

/// CABAC binary arithmetic decoder.
///
/// Operates on an RBSP byte stream (emulation prevention bytes already removed).
/// Maintains a 9-bit range and offset for arithmetic interval subdivision.
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bits_left: u8,
    current_byte: u8,
    /// Arithmetic coding interval range (9 bits, 256–510).
    range: u32,
    /// Arithmetic coding offset (9 bits).
    offset: u32,
}

impl<'a> CabacDecoder<'a> {
    /// Create and initialize a CABAC decoder from RBSP data.
    ///
    /// Reads the first 9 bits to set the initial offset (Section 9.3.2.2).
    pub fn new(data: &'a [u8]) -> Result<Self, HevcError> {
        let mut dec = Self {
            data,
            byte_pos: 0,
            bits_left: 0,
            current_byte: 0,
            range: 510,
            offset: 0,
        };

        // Read 9 bits for initial ivlOffset
        for _ in 0..9 {
            let bit = dec.read_bit()?;
            dec.offset = (dec.offset << 1) | bit;
        }

        Ok(dec)
    }

    /// Read one bit MSB-first from the byte stream.
    /// Read one bit MSB-first from the byte stream.
    ///
    /// Per HEVC spec, the CABAC byte stream is implicitly padded with zeros
    /// after the last byte. Reading past the end returns 0 bits rather than
    /// erroring — this is how real HEVC decoders handle stream exhaustion.
    #[inline]
    fn read_bit(&mut self) -> Result<u32, HevcError> {
        if self.bits_left == 0 {
            if self.byte_pos >= self.data.len() {
                // Past end of stream — return 0 (implicit zero padding)
                return Ok(0);
            }
            self.current_byte = self.data[self.byte_pos];
            self.byte_pos += 1;
            self.bits_left = 8;
        }
        self.bits_left -= 1;
        Ok(((self.current_byte >> self.bits_left) & 1) as u32)
    }

    /// Renormalize the arithmetic decoder state (Section 9.3.3.2.2).
    ///
    /// Called after range shrinks below 256. Doubles range and reads bits
    /// until range >= 256.
    #[inline]
    fn renormalize(&mut self) -> Result<(), HevcError> {
        while self.range < 256 {
            self.range <<= 1;
            self.offset <<= 1;
            self.offset |= self.read_bit()?;
        }
        Ok(())
    }

    /// Decode one context-coded bin (Section 9.3.3.2.1).
    ///
    /// Subdivides the current interval based on the context model's probability,
    /// determines if the decoded symbol is MPS or LPS, updates the context, and
    /// renormalizes.
    #[inline]
    pub fn decode_bin(&mut self, ctx: &mut ContextModel) -> Result<u32, HevcError> {
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps_range = RANGE_TAB_LPS[ctx.state as usize][q_range_idx] as u32;
        self.range -= lps_range;

        let bin_val;
        if self.offset >= self.range {
            // LPS path
            bin_val = 1 - ctx.mps as u32;
            self.offset -= self.range;
            self.range = lps_range;
            ctx.update_lps();
        } else {
            // MPS path
            bin_val = ctx.mps as u32;
            ctx.update_mps();
        }

        self.renormalize()?;
        Ok(bin_val)
    }

    /// Decode one bypass (equiprobable) bin (Section 9.3.3.2.3).
    ///
    /// No context model — each bin has equal probability of 0 and 1.
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u32, HevcError> {
        self.offset <<= 1;
        self.offset |= self.read_bit()?;

        if self.offset >= self.range {
            self.offset -= self.range;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Decode a terminating bin (Section 9.3.3.2.4).
    ///
    /// Used at the end of a slice segment to signal termination.
    /// Returns 1 if the slice/segment ends here.
    pub fn decode_terminate(&mut self) -> Result<u32, HevcError> {
        self.range -= 2;
        if self.offset >= self.range {
            Ok(1)
        } else {
            self.renormalize()?;
            Ok(0)
        }
    }

    /// Get the current byte position and bits remaining (for debugging).
    pub fn position(&self) -> (usize, u8) {
        (self.byte_pos, self.bits_left)
    }

    // -----------------------------------------------------------------------
    // Binarization decoders (Section 9.3.2.6)
    // -----------------------------------------------------------------------

    /// Decode a truncated Rice (TR) binarized value.
    ///
    /// TR binarization uses a unary prefix (context-coded or bypass) followed
    /// by a fixed-length suffix when the prefix reaches `c_rice_param` bins.
    ///
    /// # Arguments
    /// * `c_max` — maximum possible value
    /// * `c_rice_param` — Rice parameter (suffix bit count)
    /// * `ctxs` — context models for prefix bins (if `None`, use bypass for all)
    pub fn decode_tr(
        &mut self,
        c_max: u32,
        c_rice_param: u32,
        mut ctxs: Option<&mut [ContextModel]>,
    ) -> Result<u32, HevcError> {
        let prefix_max = c_max >> c_rice_param;
        let mut prefix = 0u32;

        // Decode unary prefix
        loop {
            if prefix >= prefix_max {
                break;
            }
            let bin = if let Some(ref mut ctx_slice) = ctxs.as_deref_mut() {
                let idx = prefix.min(ctx_slice.len() as u32 - 1) as usize;
                self.decode_bin(&mut ctx_slice[idx])?
            } else {
                self.decode_bypass()?
            };
            if bin == 0 {
                break;
            }
            prefix += 1;
        }

        // Decode fixed-length suffix
        if c_rice_param > 0 && prefix < prefix_max {
            let suffix = self.decode_fl_bypass(c_rice_param)?;
            Ok((prefix << c_rice_param) | suffix)
        } else if prefix >= prefix_max {
            Ok(c_max)
        } else {
            Ok(prefix << c_rice_param)
        }
    }

    /// Decode a k-th order Exp-Golomb (EGk) binarized value using bypass mode.
    ///
    /// Reads a unary-coded length prefix, then a suffix of increasing width.
    ///
    /// # Arguments
    /// * `k` — Exp-Golomb order (number of initial suffix bits)
    pub fn decode_egk(&mut self, k: u32) -> Result<u32, HevcError> {
        let mut prefix_len = 0u32;

        // Unary prefix: count leading 1-bits (bypass mode)
        loop {
            let bin = self.decode_bypass()?;
            if bin == 0 {
                break;
            }
            prefix_len += 1;
        }

        // Suffix: (prefix_len + k) bits, bypass mode
        let suffix_len = prefix_len + k;
        let mut value = 0u32;
        for _ in 0..suffix_len {
            value = (value << 1) | self.decode_bypass()?;
        }

        // Reconstruct: ((1 << prefix_len) - 1) << k + suffix
        Ok(((1u32 << prefix_len) - 1).wrapping_shl(k) + value)
    }

    /// Decode a fixed-length (FL) binarized value using bypass mode.
    ///
    /// Reads exactly `n_bits` bins in bypass mode, MSB first.
    pub fn decode_fl_bypass(&mut self, n_bits: u32) -> Result<u32, HevcError> {
        let mut value = 0u32;
        for _ in 0..n_bits {
            value = (value << 1) | self.decode_bypass()?;
        }
        Ok(value)
    }

    /// Decode a fixed-length (FL) binarized value using context-coded bins.
    ///
    /// Reads exactly `n_bits` bins using the provided context models.
    pub fn decode_fl_ctx(
        &mut self,
        n_bits: u32,
        ctxs: &mut [ContextModel],
    ) -> Result<u32, HevcError> {
        let mut value = 0u32;
        for i in 0..n_bits {
            let idx = (i as usize).min(ctxs.len() - 1);
            value = (value << 1) | self.decode_bin(&mut ctxs[idx])?;
        }
        Ok(value)
    }

    /// Decode a truncated unary (TU) binarized value using bypass mode.
    ///
    /// Reads up to `c_max` bins. The value equals the number of leading 1-bins
    /// before the first 0-bin (or `c_max` if all bins are 1).
    pub fn decode_tu_bypass(&mut self, c_max: u32) -> Result<u32, HevcError> {
        let mut value = 0u32;
        while value < c_max {
            let bin = self.decode_bypass()?;
            if bin == 0 {
                break;
            }
            value += 1;
        }
        Ok(value)
    }

    /// Decode a truncated unary (TU) binarized value using context-coded bins.
    ///
    /// Each bin position can use a different context from `ctxs`.
    pub fn decode_tu_ctx(
        &mut self,
        c_max: u32,
        ctxs: &mut [ContextModel],
    ) -> Result<u32, HevcError> {
        let mut value = 0u32;
        while value < c_max {
            let idx = (value as usize).min(ctxs.len() - 1);
            let bin = self.decode_bin(&mut ctxs[idx])?;
            if bin == 0 {
                break;
            }
            value += 1;
        }
        Ok(value)
    }

    /// Number of bytes consumed from the input so far.
    pub fn bytes_consumed(&self) -> usize {
        self.byte_pos
    }

    /// Current arithmetic range (for debugging/testing).
    pub fn range(&self) -> u32 {
        self.range
    }

    /// Current arithmetic offset (for debugging/testing).
    pub fn offset(&self) -> u32 {
        self.offset
    }
}
