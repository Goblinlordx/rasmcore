//! CABAC arithmetic decoder engine (ITU-T H.265 Section 9.3.3).
//!
//! Uses a byte-aligned implementation with 9-bit range and 16-bit value,
//! matching the reference decoder approach for bit-exact results.

use super::context::ContextModel;
use super::tables::{RANGE_TAB_LPS, RENORM_TABLE};
use crate::error::HevcError;

/// CABAC binary arithmetic decoder.
///
/// Operates on an RBSP byte stream (emulation prevention bytes already removed).
/// Uses a 16-bit value register with byte-granularity input for efficiency.
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Arithmetic coding interval range (9 bits, 256–510).
    range: u32,
    /// Arithmetic coding value (16-bit precision).
    value: u32,
    /// Bits needed before next byte read. Starts at 8, goes negative.
    bits_needed: i32,
    /// Bin trace log — populated when trace mode is enabled.
    #[cfg(feature = "trace")]
    pub trace: Vec<BinTrace>,
    #[cfg(feature = "trace")]
    trace_enabled: bool,
    #[cfg(feature = "trace")]
    bin_count: u32,
}

/// A single CABAC bin decode event for trace comparison.
#[cfg(feature = "trace")]
#[derive(Debug, Clone)]
pub struct BinTrace {
    pub bin_num: u32,
    pub bin_type: BinType,
    pub state_before: u8,
    pub range_before: u32,
    pub offset_before: u32,
    pub result: u32,
    pub range_after: u32,
    pub offset_after: u32,
}

#[cfg(feature = "trace")]
#[derive(Debug, Clone, PartialEq)]
pub enum BinType {
    Context,
    Bypass,
    Terminate,
}

impl<'a> CabacDecoder<'a> {
    /// Create and initialize a CABAC decoder from RBSP data.
    ///
    /// Reads the first 2 bytes to set the initial value (Section 9.3.2.2).
    pub fn new(data: &'a [u8]) -> Result<Self, HevcError> {
        let mut value = 0u32;
        let mut byte_pos = 0usize;
        let mut bits_needed = 8i32;

        // Read first 2 bytes as initial value
        if !data.is_empty() {
            value = (data[0] as u32) << 8;
            byte_pos = 1;
            bits_needed -= 8;
        }
        if data.len() > 1 {
            value |= data[1] as u32;
            byte_pos = 2;
            bits_needed -= 8;
        }

        Ok(Self {
            data,
            byte_pos,
            range: 510,
            value,
            bits_needed,
            #[cfg(feature = "trace")]
            trace: Vec::new(),
            #[cfg(feature = "trace")]
            trace_enabled: false,
            #[cfg(feature = "trace")]
            bin_count: 0,
        })
    }

    /// Read the next byte from the stream (or 0 past end).
    #[inline]
    fn read_byte(&mut self) -> u8 {
        if self.byte_pos < self.data.len() {
            let b = self.data[self.byte_pos];
            self.byte_pos += 1;
            b
        } else {
            0
        }
    }

    /// Decode one context-coded bin (Section 9.3.3.2.1).
    ///
    /// Uses the scaled-range approach: compare value against range << 7.
    #[inline]
    pub fn decode_bin(&mut self, ctx: &mut ContextModel) -> Result<u32, HevcError> {
        #[cfg(feature = "trace")]
        let (trace_state, trace_range, trace_value) = (ctx.state, self.range, self.value);

        let lps_range = RANGE_TAB_LPS[ctx.state as usize][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps_range;

        let scaled_range = self.range << 7;

        let bin_val;
        if self.value < scaled_range {
            // MPS path
            bin_val = ctx.mps as u32;
            ctx.update_mps();

            // Renormalize if needed (range might have dropped below 256)
            if scaled_range < (256 << 7) {
                self.range = scaled_range >> 6;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed == 0 {
                    self.bits_needed = -8;
                    self.value |= self.read_byte() as u32;
                }
            }
        } else {
            // LPS path
            bin_val = 1 - ctx.mps as u32;
            self.value -= scaled_range;

            let num_bits = RENORM_TABLE[(lps_range >> 3) as usize];
            self.value <<= num_bits;
            self.range = lps_range << num_bits;

            // update_lps handles both the mps flip at state 0 and state transition
            ctx.update_lps();

            self.bits_needed += num_bits as i32;
            if self.bits_needed >= 0 {
                self.value |= (self.read_byte() as u32) << self.bits_needed as u32;
                self.bits_needed -= 8;
            }
        }

        #[cfg(feature = "trace")]
        if self.trace_enabled {
            self.bin_count += 1;
            self.trace.push(BinTrace {
                bin_num: self.bin_count,
                bin_type: BinType::Context,
                state_before: trace_state,
                range_before: trace_range,
                offset_before: trace_value,
                result: bin_val,
                range_after: self.range,
                offset_after: self.value,
            });
        }

        Ok(bin_val)
    }

    /// Decode one bypass (equiprobable) bin (Section 9.3.3.2.3).
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u32, HevcError> {
        #[cfg(feature = "trace")]
        let (trace_range, trace_value) = (self.range, self.value);

        self.value <<= 1;
        self.bits_needed += 1;
        if self.bits_needed >= 0 {
            self.bits_needed = -8;
            self.value |= self.read_byte() as u32;
        }

        let scaled_range = self.range << 7;
        let bin_val = if self.value >= scaled_range {
            self.value -= scaled_range;
            1
        } else {
            0
        };

        #[cfg(feature = "trace")]
        if self.trace_enabled {
            self.bin_count += 1;
            self.trace.push(BinTrace {
                bin_num: self.bin_count,
                bin_type: BinType::Bypass,
                state_before: 0,
                range_before: trace_range,
                offset_before: trace_value,
                result: bin_val,
                range_after: self.range,
                offset_after: self.value,
            });
        }

        Ok(bin_val)
    }

    /// Decode a terminating bin (Section 9.3.3.2.4).
    ///
    /// Returns 1 if the slice/segment ends here.
    pub fn decode_terminate(&mut self) -> Result<u32, HevcError> {
        self.range -= 2;
        let scaled_range = self.range << 7;
        if self.value >= scaled_range {
            Ok(1)
        } else {
            // Renormalize using scaled_range check (matching reference decoder)
            if scaled_range < (256 << 7) {
                self.range = scaled_range >> 6;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed == 0 {
                    self.bits_needed = -8;
                    self.value |= self.read_byte() as u32;
                }
            }

            Ok(0)
        }
    }

    /// Enable CABAC bin tracing for debugging/validation.
    #[cfg(feature = "trace")]
    pub fn enable_trace(&mut self) {
        self.trace_enabled = true;
        self.trace.clear();
        self.bin_count = 0;
    }

    /// Get the current byte position and bits remaining (for debugging).
    pub fn position(&self) -> (usize, u8) {
        (self.byte_pos, 0)
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
        self.value
    }

    // -----------------------------------------------------------------------
    // Binarization decoders (Section 9.3.2.6)
    // -----------------------------------------------------------------------

    /// Decode a truncated Rice (TR) binarized value.
    pub fn decode_tr(
        &mut self,
        c_max: u32,
        c_rice_param: u32,
        mut ctxs: Option<&mut [ContextModel]>,
    ) -> Result<u32, HevcError> {
        let prefix_max = c_max >> c_rice_param;
        let mut prefix = 0u32;

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
    pub fn decode_egk(&mut self, k: u32) -> Result<u32, HevcError> {
        let mut prefix_len = 0u32;
        loop {
            let bin = self.decode_bypass()?;
            if bin == 0 {
                break;
            }
            prefix_len += 1;
        }

        let suffix_len = prefix_len + k;
        let mut value = 0u32;
        for _ in 0..suffix_len {
            value = (value << 1) | self.decode_bypass()?;
        }

        Ok(((1u32 << prefix_len) - 1).wrapping_shl(k) + value)
    }

    /// Decode a fixed-length (FL) binarized value using bypass mode.
    pub fn decode_fl_bypass(&mut self, n_bits: u32) -> Result<u32, HevcError> {
        let mut value = 0u32;
        for _ in 0..n_bits {
            value = (value << 1) | self.decode_bypass()?;
        }
        Ok(value)
    }

    /// Decode a fixed-length (FL) binarized value using context-coded bins.
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
}
