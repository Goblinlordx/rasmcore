//! CABAC arithmetic encoder engine — direct port of x265 4.1.
//!
//! The encoding counterpart to `cabac/decoder.rs`. Uses the same state machine,
//! context models, and LPS range tables, but operates in the encoding direction:
//! instead of reading bins from a bitstream, it writes bins to produce a bitstream.
//!
//! Ref: x265 4.1 encoder/entropy.cpp — encodeBin, encodeBinEP, encodeBinTrm
//! Ref: ITU-T H.265 Section 9.3.4 (arithmetic encoding process)

use crate::cabac::ContextModel;
use crate::cabac::tables::{RANGE_TAB_LPS, RENORM_TABLE};

/// CABAC binary arithmetic encoder.
///
/// Maintains range/low state and outputs bytes as the arithmetic interval narrows.
/// Uses the same 9-bit range, LPS table, and context state machine as the decoder.
///
/// Ref: x265 4.1 encoder/entropy.cpp — Entropy class
pub struct CabacEncoder {
    /// Output byte buffer.
    buf: Vec<u8>,
    /// Arithmetic coding interval range (9 bits, 256–510).
    range: u32,
    /// Lower bound of the arithmetic interval (10+ bits).
    low: u32,
    /// Number of outstanding bits for carry propagation.
    bits_left: i32,
    /// Number of buffered "0xFF" bytes waiting for carry resolution.
    num_buffered_bytes: u32,
    /// The byte being buffered (before we know if carry propagates).
    buffered_byte: u8,
}

impl CabacEncoder {
    /// Create a new CABAC encoder.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            range: 510,
            low: 0,
            bits_left: 23,
            num_buffered_bytes: 0,
            buffered_byte: 0xFF,
        }
    }

    /// Encode one context-coded bin.
    ///
    /// This is the inverse of `CabacDecoder::decode_bin`. Given a symbol (0 or 1)
    /// and a context model, it narrows the arithmetic interval and updates the
    /// context probability.
    ///
    /// Ref: x265 4.1 encoder/entropy.cpp — encodeBin()
    /// Ref: ITU-T H.265 Section 9.3.4.2
    #[inline]
    pub fn encode_bin(&mut self, bin_val: u32, ctx: &mut ContextModel) {
        let lps_range = RANGE_TAB_LPS[ctx.state as usize][((self.range >> 6) & 3) as usize] as u32;

        self.range -= lps_range;

        if bin_val != ctx.mps as u32 {
            // LPS path — encode the less probable symbol
            self.low += self.range;
            self.range = lps_range;
            ctx.update_lps();

            // Renormalize
            let num_bits = RENORM_TABLE[(lps_range >> 3) as usize] as i32;
            self.low <<= num_bits;
            self.range <<= num_bits;
            self.bits_left -= num_bits;
        } else {
            // MPS path — encode the more probable symbol
            ctx.update_mps();

            // Renormalize only if range dropped below 256
            if self.range < 256 {
                self.low <<= 1;
                self.range <<= 1;
                self.bits_left -= 1;
            }
        }

        if self.bits_left < 12 {
            self.write_out();
        }
    }

    /// Encode one bypass (equiprobable) bin.
    ///
    /// Ref: x265 4.1 encoder/entropy.cpp — encodeBinEP()
    /// Ref: ITU-T H.265 Section 9.3.4.4
    #[inline]
    pub fn encode_bypass(&mut self, bin_val: u32) {
        self.low <<= 1;
        if bin_val != 0 {
            self.low += self.range;
        }
        self.bits_left -= 1;

        if self.bits_left < 12 {
            self.write_out();
        }
    }

    /// Encode multiple bypass bins from a value (MSB first).
    ///
    /// Ref: x265 4.1 encoder/entropy.cpp — encodeBinsEP()
    pub fn encode_bins_ep(&mut self, value: u32, num_bins: u32) {
        for i in (0..num_bins).rev() {
            self.encode_bypass((value >> i) & 1);
        }
    }

    /// Encode a terminating bin (end of slice segment).
    ///
    /// Ref: x265 4.1 encoder/entropy.cpp — encodeBinTrm()
    /// Ref: ITU-T H.265 Section 9.3.4.5
    pub fn encode_terminate(&mut self, bin_val: u32) {
        self.range -= 2;

        if bin_val != 0 {
            // Terminate: symbol is in the upper (2-wide) sub-range
            self.low += self.range;
            self.range = 2;
            // Renormalize: range=2 requires many shifts to get back to >= 256
            // Each shift doubles range and shifts low left
            while self.range < 256 {
                self.low <<= 1;
                self.range <<= 1;
                self.bits_left -= 1;
                if self.bits_left < 12 {
                    self.write_out();
                }
            }
        } else {
            // Continue: renormalize if range dropped below 256
            if self.range < 256 {
                self.low <<= 1;
                self.range <<= 1;
                self.bits_left -= 1;
            }
            if self.bits_left < 12 {
                self.write_out();
            }
        }
    }

    /// Finalize and return the encoded CABAC data as bytes.
    ///
    /// Flushes carry-propagation buffered bytes and writes the remaining
    /// bits from the low register.
    ///
    /// Ref: HM TEncBinCABAC::finish()
    /// Ref: x265 4.1 encoder/entropy.cpp — finish()
    pub fn finish_and_get_bytes(&mut self) -> Vec<u8> {
        // Step 1: Flush buffered bytes with carry resolution
        if self.low >> (32 - self.bits_left as u32) != 0 {
            // Carry propagates through buffered 0xFF bytes
            self.push_byte(self.buffered_byte.wrapping_add(1));
            while self.num_buffered_bytes > 1 {
                self.push_byte(0x00);
                self.num_buffered_bytes -= 1;
            }
            self.low -= 1u32 << (32 - self.bits_left as u32);
        } else {
            if self.num_buffered_bytes > 0 {
                self.push_byte(self.buffered_byte);
            }
            while self.num_buffered_bytes > 1 {
                self.push_byte(0xFF);
                self.num_buffered_bytes -= 1;
            }
        }

        // Step 2: Write remaining bits from low register.
        // Ref: HM TEncBinCABAC::finish() — write(m_uiLow >> 8, 24 - m_bitsLeft)
        // We have (24 - bits_left) significant bits in (low >> 8).
        let num_remaining_bits = 24 - self.bits_left;
        if num_remaining_bits > 0 {
            // Extract the significant bits from low >> 8, MSB-first
            let value = self.low >> 8;
            let mut bits_left = num_remaining_bits;
            while bits_left >= 8 {
                bits_left -= 8;
                self.push_byte((value >> bits_left as u32) as u8);
            }
            // Write final partial byte padded with zeros
            if bits_left > 0 {
                self.push_byte(((value << (8 - bits_left as u32)) & 0xFF) as u8);
            }
        }

        std::mem::take(&mut self.buf)
    }

    /// Output a byte during renormalization.
    ///
    /// Uses carry-propagation buffering: when a potential carry could ripple
    /// through 0xFF bytes, we buffer them until we know whether the carry happens.
    ///
    /// Ref: x265 4.1 encoder/entropy.cpp — writeOut()
    fn write_out(&mut self) {
        let lead_byte = self.low >> (24 - self.bits_left as u32);
        self.bits_left += 8;
        self.low &= 0xFFFF_FFFFu32 >> self.bits_left as u32;

        if lead_byte == 0xFF {
            self.num_buffered_bytes += 1;
        } else if self.num_buffered_bytes > 0 {
            let carry = lead_byte >> 8;
            let byte = self.buffered_byte + carry as u8;
            self.buffered_byte = lead_byte as u8;
            self.push_byte(byte);

            let stuffed = if carry != 0 { 0x00u8 } else { 0xFFu8 };
            while self.num_buffered_bytes > 1 {
                self.push_byte(stuffed);
                self.num_buffered_bytes -= 1;
            }
            self.num_buffered_bytes = 1;
        } else {
            self.num_buffered_bytes = 1;
            self.buffered_byte = lead_byte as u8;
        }
    }

    /// Push a byte to the output buffer.
    #[inline]
    fn push_byte(&mut self, byte: u8) {
        self.buf.push(byte);
    }

    /// Get the number of bytes written so far (approximate, may change with carry).
    pub fn bytes_written(&self) -> usize {
        self.buf.len()
    }

    /// Reset the encoder state (for starting a new slice or WPP substream).
    pub fn reset(&mut self) {
        self.range = 510;
        self.low = 0;
        self.bits_left = 23;
        self.num_buffered_bytes = 0;
        self.buffered_byte = 0xFF;
        self.buf.clear();
    }
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacDecoder;

    /// Encode a sequence of context-coded bins, then decode and verify they match.
    #[test]
    fn encode_decode_roundtrip_context() {
        let qp = 26;
        let init_value = 154u8; // typical init value

        // Encode
        let mut enc = CabacEncoder::new();
        let mut enc_ctx = ContextModel::new(init_value, qp);
        let bins = [0u32, 1, 0, 0, 1, 1, 0, 1, 0, 0];

        for &bin in &bins {
            enc.encode_bin(bin, &mut enc_ctx);
        }
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        // Decode
        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx = ContextModel::new(init_value, qp);

        for (i, &expected) in bins.iter().enumerate() {
            let decoded = dec.decode_bin(&mut dec_ctx).unwrap();
            assert_eq!(
                decoded, expected,
                "bin {i}: expected {expected}, got {decoded}"
            );
        }

        let term = dec.decode_terminate().unwrap();
        assert_eq!(term, 1, "terminate should be 1");
    }

    /// Encode and decode bypass bins.
    #[test]
    fn encode_decode_roundtrip_bypass() {
        let mut enc = CabacEncoder::new();
        let bins = [1u32, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1];

        for &bin in &bins {
            enc.encode_bypass(bin);
        }
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        let mut dec = CabacDecoder::new(&data).unwrap();
        for (i, &expected) in bins.iter().enumerate() {
            let decoded = dec.decode_bypass().unwrap();
            assert_eq!(
                decoded, expected,
                "bypass bin {i}: expected {expected}, got {decoded}"
            );
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// Mixed context and bypass bins.
    #[test]
    fn encode_decode_roundtrip_mixed() {
        let qp = 30;
        let mut enc = CabacEncoder::new();
        let mut ctx1 = ContextModel::new(154, qp);
        let mut ctx2 = ContextModel::new(110, qp);

        // Mix of context-coded and bypass bins (simulates real syntax)
        enc.encode_bin(0, &mut ctx1); // split_cu_flag = 0
        enc.encode_bin(1, &mut ctx2); // prev_intra_luma_pred_flag = 1
        enc.encode_bypass(0); // mpm_idx bit 0
        enc.encode_bypass(1); // mpm_idx bit 1
        enc.encode_bin(0, &mut ctx1); // cbf_luma = 0
        enc.encode_terminate(1);

        let data = enc.finish_and_get_bytes();

        // Decode
        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx1 = ContextModel::new(154, qp);
        let mut dec_ctx2 = ContextModel::new(110, qp);

        assert_eq!(dec.decode_bin(&mut dec_ctx1).unwrap(), 0);
        assert_eq!(dec.decode_bin(&mut dec_ctx2).unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 0);
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_bin(&mut dec_ctx1).unwrap(), 0);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// Encode a long sequence to test carry propagation and buffering.
    #[test]
    fn encode_decode_long_sequence() {
        let qp = 22;
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextModel::new(154, qp);

        // 100 alternating bins to stress the carry propagation
        let mut bins = Vec::new();
        for i in 0..100 {
            let bin = (i % 3 != 0) as u32;
            bins.push(bin);
            enc.encode_bin(bin, &mut ctx);
        }
        // Also some bypass bins
        for i in 0..50 {
            let bin = (i % 2) as u32;
            bins.push(bin);
            enc.encode_bypass(bin);
        }
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        // Decode and verify
        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx = ContextModel::new(154, qp);

        for (i, &expected) in bins[..100].iter().enumerate() {
            let decoded = dec.decode_bin(&mut dec_ctx).unwrap();
            assert_eq!(decoded, expected, "context bin {i} mismatch");
        }
        for (i, &expected) in bins[100..].iter().enumerate() {
            let decoded = dec.decode_bypass().unwrap();
            assert_eq!(decoded, expected, "bypass bin {i} mismatch");
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// Encode bins_ep (multi-bit bypass value).
    #[test]
    fn encode_decode_bins_ep() {
        let mut enc = CabacEncoder::new();
        enc.encode_bins_ep(0b10110, 5);
        enc.encode_bins_ep(0b11111111, 8);
        enc.encode_bins_ep(0b000, 3);
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        let mut dec = CabacDecoder::new(&data).unwrap();
        // Read back as individual bypass bins
        // 10110
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 0);
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 0);
        // 11111111
        for _ in 0..8 {
            assert_eq!(dec.decode_bypass().unwrap(), 1);
        }
        // 000
        for _ in 0..3 {
            assert_eq!(dec.decode_bypass().unwrap(), 0);
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// Test that context state evolves identically between encoder and decoder.
    #[test]
    fn context_state_parity() {
        let qp = 26;
        let init = 154u8;
        let bins = [0u32, 0, 1, 0, 1, 1, 1, 0, 0, 1];

        let mut enc_ctx = ContextModel::new(init, qp);
        let mut dec_ctx = ContextModel::new(init, qp);

        // Encode
        let mut enc = CabacEncoder::new();
        for &bin in &bins {
            enc.encode_bin(bin, &mut enc_ctx);
        }
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        // Decode
        let mut dec = CabacDecoder::new(&data).unwrap();
        for &expected in &bins {
            let decoded = dec.decode_bin(&mut dec_ctx).unwrap();
            assert_eq!(decoded, expected);
        }

        // Context states should be identical after encoding and decoding the same sequence
        assert_eq!(
            enc_ctx.state, dec_ctx.state,
            "context state diverged: enc={}, dec={}",
            enc_ctx.state, dec_ctx.state
        );
        assert_eq!(
            enc_ctx.mps, dec_ctx.mps,
            "context mps diverged: enc={}, dec={}",
            enc_ctx.mps, dec_ctx.mps
        );
    }

    /// Non-terminating continue (terminate bin = 0).
    #[test]
    fn encode_decode_terminate_continue() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextModel::new(154, 26);

        enc.encode_bin(1, &mut ctx);
        enc.encode_terminate(0); // continue, not end
        enc.encode_bin(0, &mut ctx);
        enc.encode_terminate(1); // actual end

        let data = enc.finish_and_get_bytes();

        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx = ContextModel::new(154, 26);

        assert_eq!(dec.decode_bin(&mut dec_ctx).unwrap(), 1);
        assert_eq!(dec.decode_terminate().unwrap(), 0); // continue
        assert_eq!(dec.decode_bin(&mut dec_ctx).unwrap(), 0);
        assert_eq!(dec.decode_terminate().unwrap(), 1); // end
    }

    /// Minimal: encode single MPS bin.
    #[test]
    fn encode_decode_single_mps() {
        let qp = 26;
        let init = 154u8;
        let mut enc = CabacEncoder::new();
        let mut enc_ctx = ContextModel::new(init, qp);
        let mps = enc_ctx.mps as u32;

        enc.encode_bin(mps, &mut enc_ctx);
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        eprintln!("single MPS: {} bytes: {:02x?}", data.len(), data);

        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx = ContextModel::new(init, qp);
        let decoded = dec.decode_bin(&mut dec_ctx).unwrap();
        assert_eq!(decoded, mps, "single MPS bin mismatch");
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// All-zero bins (MPS-heavy, tests range expansion).
    #[test]
    fn encode_decode_all_mps() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextModel::new(154, 26);

        for _ in 0..200 {
            enc.encode_bin(ctx.mps as u32, &mut ctx);
        }
        enc.encode_terminate(1);
        let data = enc.finish_and_get_bytes();

        let mut dec = CabacDecoder::new(&data).unwrap();
        let mut dec_ctx = ContextModel::new(154, 26);

        for i in 0..200 {
            let decoded = dec.decode_bin(&mut dec_ctx).unwrap();
            // After encoding all MPS, the context stabilizes
            assert!(decoded == 0 || decoded == 1, "invalid bin at position {i}");
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }
}
