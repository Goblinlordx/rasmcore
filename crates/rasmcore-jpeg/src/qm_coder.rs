//! Standard QM-coder arithmetic coding per ITU-T T.81 Annex D.
//!
//! Implements both encoder and decoder with the standard byte format
//! so that our SOF9/SOF10 JPEGs are interoperable with libjpeg-turbo,
//! mozjpeg, and ImageMagick.
//!
//! Algorithm from the public ITU-T T.81 standard (JPEG specification).

// ─── ARITAB — Packed Probability Estimation Table ────────────────────────────
//
// 114-entry table from ITU-T T.81 Table D.2/D.3.
// Each u32 encodes:
//   bits  0-15: Qe (probability estimate)
//   bits 16-22: Next_Index_LPS
//   bit     23: Switch_MPS
//   bits 24-30: Next_Index_MPS
//
// The table has 113 "real" entries (indices 0-112) plus index 113
// which is the fixed probability context (Qe=0x5a1d).

const ARITAB: [u32; 114] = [
    0x0181_5a1d,
    0x020e_2586,
    0x0310_1114,
    0x0412_080b,
    0x0514_03d8,
    0x0617_01da,
    0x0719_00e5,
    0x081c_006f,
    0x091e_0036,
    0x0a21_001a,
    0x0b23_000d,
    0x0c09_0006,
    0x0d0a_0003,
    0x0d0c_0001,
    0x0f8f_5a7f,
    0x1024_3f25,
    0x1126_2cf2,
    0x1227_207c,
    0x1328_17b9,
    0x142a_1182,
    0x152b_0cef,
    0x162d_09a1,
    0x172e_072f,
    0x1830_055c,
    0x1931_0406,
    0x1a33_0303,
    0x1b34_0240,
    0x1c36_01b1,
    0x1d38_0144,
    0x1e39_00f5,
    0x1f3b_00b7,
    0x203c_008a,
    0x213e_0068,
    0x223f_004e,
    0x2320_003b,
    0x0921_002c,
    0x25a5_5ae1,
    0x2640_484c,
    0x2741_3a0d,
    0x2843_2ef1,
    0x2944_261f,
    0x2a45_1f33,
    0x2b46_19a8,
    0x2c48_1518,
    0x2d49_1177,
    0x2e4a_0e74,
    0x2f4b_0bfb,
    0x304d_09f8,
    0x314e_0861,
    0x324f_0706,
    0x3330_05cd,
    0x3432_04de,
    0x3532_040f,
    0x3633_0363,
    0x3734_02d4,
    0x3835_025c,
    0x3936_01f8,
    0x3a37_01a4,
    0x3b38_0160,
    0x3c39_0125,
    0x3d3a_00f6,
    0x3e3b_00cb,
    0x3f3d_00ab,
    0x203d_008f,
    0x41c1_5b12,
    0x4250_4d04,
    0x4351_412c,
    0x4452_37d8,
    0x4553_2fe8,
    0x4654_293c,
    0x4756_2379,
    0x4857_1edf,
    0x4957_1aa9,
    0x4a48_174e,
    0x4b48_1424,
    0x4c4a_119c,
    0x4d4a_0f6b,
    0x4e4b_0d51,
    0x4f4d_0bb6,
    0x304d_0a40,
    0x51d0_5832,
    0x5258_4d1c,
    0x5359_438e,
    0x545a_3bdd,
    0x555b_34ee,
    0x565c_2eae,
    0x575d_299a,
    0x4756_2516,
    0x59d8_5570,
    0x5a5f_4ca9,
    0x5b60_44d9,
    0x5c61_3e22,
    0x5d63_3824,
    0x5e63_32b4,
    0x565d_2e17,
    0x60df_56a8,
    0x6165_4f46,
    0x6266_47e5,
    0x6367_41cf,
    0x6468_3c3d,
    0x5d63_375e,
    0x6669_5231,
    0x676a_4c0f,
    0x686b_4639,
    0x6367_415e,
    0x6ae9_5627,
    0x6b6c_50e7,
    0x676d_4b85,
    0x6d6e_5597,
    0x6b6f_504f,
    0x6fee_5a10,
    0x6d70_5522,
    0x6ff0_59eb,
    0x7171_5a1d,
];

// ─── QM-Coder Decoder ──────────────────────────────────────────────────────

/// Standard QM-coder decoder state per ITU-T T.81 Annex D.
///
/// Reads a byte-stuffed entropy stream and decodes binary decisions.
/// Compatible with libjpeg-turbo's jdarith.c output.
pub struct QmDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    c: i32,
    a: i32,
    ct: i32,
    unread_marker: bool,
}

impl<'a> QmDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            c: 0,
            a: 0,
            ct: -16, // forces reading 2 initial bytes
            unread_marker: false,
        }
    }

    /// Reset state (for restart markers).
    pub fn reset(&mut self) {
        self.c = 0;
        self.a = 0;
        self.ct = -16;
    }

    #[inline]
    fn get_byte(&mut self) -> i32 {
        if self.pos >= self.data.len() {
            return 0;
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte != 0xFF {
            return byte as i32;
        }

        // Handle 0xFF — could be stuffed byte or marker
        loop {
            if self.pos >= self.data.len() {
                return 0xFF;
            }
            let next = self.data[self.pos];
            self.pos += 1;

            if next == 0xFF {
                continue; // skip padding 0xFF bytes
            }
            if next == 0x00 {
                return 0xFF; // byte stuffing: 0xFF00 → 0xFF
            }
            // Found a marker — stop reading
            self.unread_marker = true;
            return 0;
        }
    }

    /// Decode one binary decision using context state `st`.
    ///
    /// `st` is a u8 where bit 7 = current MPS sense, bits 0-6 = state index.
    /// Returns 0 or 1.
    #[inline]
    pub fn decode(&mut self, st: &mut u8) -> u8 {
        // Renormalization & data input (ITU-T T.81 D.2.6)
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let data = if self.unread_marker {
                    0
                } else {
                    self.get_byte()
                };
                self.c = (self.c << 8) | data;
                self.ct += 8;
                if self.ct < 0 {
                    // Still in initial fill phase
                    self.ct += 1;
                    if self.ct == 0 {
                        // Got 2 initial bytes — set A to half (becomes 0x10000 after <<= 1)
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }

        // Fetch from ARITAB
        let sv = *st;
        let entry = ARITAB[(sv & 0x7F) as usize];
        let qe = (entry & 0xFFFF) as i32;
        let nl = ((entry >> 16) & 0x7F) as u8; // next state after LPS
        let nm = ((entry >> 24) & 0x7F) as u8; // next state after MPS
        let switch_mps = (entry >> 23) & 1 != 0;

        // Decode (ITU-T T.81 D.2.4 / D.2.5)
        let temp = self.a - qe;
        self.a = temp;
        let temp_shifted = temp << self.ct;

        if self.c >= temp_shifted {
            // LPS sub-interval
            self.c -= temp_shifted;
            if self.a < qe {
                // Conditional exchange — actually MPS
                self.a = qe;
                *st = (sv & 0x80) ^ nm;
            } else {
                // Regular LPS
                self.a = qe;
                *st = (sv & 0x80) ^ nl;
                if switch_mps {
                    *st ^= 0x80;
                }
                return (sv ^ 0x80) >> 7;
            }
        } else if self.a < 0x8000 {
            // MPS sub-interval, needs renormalization
            if self.a < qe {
                // Conditional exchange — actually LPS
                *st = (sv & 0x80) ^ nl;
                if switch_mps {
                    *st ^= 0x80;
                }
                return (sv ^ 0x80) >> 7;
            } else {
                *st = (sv & 0x80) ^ nm;
            }
        }

        sv >> 7 // MPS value
    }
}

// ─── QM-Coder Encoder ──────────────────────────────────────────────────────

/// Standard QM-coder encoder state per ITU-T T.81 Annex D.
///
/// Produces a byte-stuffed entropy stream compatible with libjpeg-turbo.
pub struct QmEncoder {
    a: u32,       // interval width
    c: u32,       // code register
    ct: u8,       // bits-to-output counter
    buffer: i32,  // deferred output byte (-1 = sentinel: no push on first byte_out)
    sc: u32,      // count of deferred 0xFF bytes
    buf: Vec<u8>, // output buffer
}

impl Default for QmEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl QmEncoder {
    pub fn new() -> Self {
        Self {
            a: 0x10000,
            c: 0,
            ct: 11,
            buffer: -1, // sentinel: first byte_out just sets buffer, doesn't push
            sc: 0,
            buf: Vec::with_capacity(4096),
        }
    }

    /// Reset state (for restart markers).
    pub fn reset(&mut self) {
        self.a = 0x10000;
        self.c = 0;
        self.ct = 11;
        self.buffer = -1;
        self.sc = 0;
    }

    /// Encode one binary decision using context state `st`.
    ///
    /// `st` is a u8 where bit 7 = current MPS sense, bits 0-6 = state index.
    /// `bit`: 0 or 1.
    #[inline]
    pub fn encode(&mut self, st: &mut u8, bit: u8) {
        let sv = *st;
        let entry = ARITAB[(sv & 0x7F) as usize];
        let qe = entry & 0xFFFF;
        let nl = ((entry >> 16) & 0x7F) as u8;
        let nm = ((entry >> 24) & 0x7F) as u8;
        let switch_mps = (entry >> 23) & 1 != 0;

        let mps = sv >> 7;

        self.a -= qe;

        if bit == mps {
            // MPS path
            if self.a < 0x8000 {
                if self.a < qe {
                    // Conditional exchange
                    self.c += self.a;
                    self.a = qe;
                }
                *st = (sv & 0x80) ^ nm;
                self.renormalize_e();
            }
        } else {
            // LPS path
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            *st = (sv & 0x80) ^ nl;
            if switch_mps {
                *st ^= 0x80;
            }
            self.renormalize_e();
        }
    }

    fn renormalize_e(&mut self) {
        while self.a < 0x8000 {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
                self.ct = 8;
            }
        }
    }

    /// Emit a byte to the output stream with JPEG byte stuffing.
    /// After 0xFF, emit 0x00 to distinguish from markers.
    fn emit_byte(&mut self, byte: u8) {
        self.buf.push(byte);
        if byte == 0xFF {
            self.buf.push(0x00); // byte stuffing
        }
    }

    fn byte_out(&mut self) {
        let temp = self.c >> 19;
        if temp > 0xFF {
            // Carry propagation
            if self.buffer >= 0 {
                self.emit_byte((self.buffer + 1) as u8);
            }
            for _ in 0..self.sc {
                self.emit_byte(0x00);
            }
            self.sc = 0;
            self.buffer = (temp & 0xFF) as i32;
        } else if temp == 0xFF {
            // Defer (might get a carry later)
            self.sc += 1;
        } else {
            // Normal output
            if self.buffer >= 0 {
                self.emit_byte(self.buffer as u8);
            }
            for _ in 0..self.sc {
                self.emit_byte(0xFF);
            }
            self.sc = 0;
            self.buffer = temp as i32;
        }
        self.c &= 0x7FFFF;
    }

    /// Flush and return encoded byte stream (byte-stuffed).
    pub fn finish(&mut self) -> Vec<u8> {
        // Select a value in [c, c+a) with maximal trailing zeros
        let mut temp = (self.c + self.a - 1) & 0xFFFF0000;
        if temp < self.c {
            temp += 0x8000;
        }
        self.c = temp;

        // Shift remaining bits and emit final bytes
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= 8;
        self.byte_out();
        if self.buffer >= 0 {
            self.emit_byte(self.buffer as u8);
        }

        std::mem::take(&mut self.buf)
    }
}

// ─── JPEG Arithmetic Decoder ───────────────────────────────────────────────

/// Number of DC statistics bins per table (ITU-T T.81 Table F.4).
const DC_STAT_BINS: usize = 64;
/// Number of AC statistics bins per table.
const AC_STAT_BINS: usize = 256;
/// Maximum arithmetic coding tables.
const NUM_ARITH_TBLS: usize = 4;

/// JPEG arithmetic decoder wrapping QmDecoder with standard DC/AC context modeling.
pub struct JpegArithDecoder<'a> {
    qm: QmDecoder<'a>,
    dc_stats: [[u8; DC_STAT_BINS]; NUM_ARITH_TBLS],
    ac_stats: [[u8; AC_STAT_BINS]; NUM_ARITH_TBLS],
    fixed_bin: [u8; 4],
    last_dc_val: [i32; 4],
    dc_context: [u8; 4],
    dc_cond: [(u8, u8); NUM_ARITH_TBLS], // (L, U) DAC conditioning
    ac_kx: [u8; NUM_ARITH_TBLS],         // AC conditioning Kx
}

impl<'a> JpegArithDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            qm: QmDecoder::new(data),
            dc_stats: [[0; DC_STAT_BINS]; NUM_ARITH_TBLS],
            ac_stats: [[0; AC_STAT_BINS]; NUM_ARITH_TBLS],
            fixed_bin: [113, 0, 0, 0], // index 113 = fixed probability context
            last_dc_val: [0; 4],
            dc_context: [0; 4],
            dc_cond: [(0, 1); NUM_ARITH_TBLS], // default L=0, U=1
            ac_kx: [5; NUM_ARITH_TBLS],        // default Kx=5
        }
    }

    /// Set DC conditioning values from DAC marker.
    pub fn set_dc_conditioning(&mut self, tbl: usize, l: u8, u: u8) {
        if tbl < NUM_ARITH_TBLS {
            self.dc_cond[tbl] = (l, u);
        }
    }

    /// Set AC conditioning value from DAC marker.
    pub fn set_ac_conditioning(&mut self, tbl: usize, kx: u8) {
        if tbl < NUM_ARITH_TBLS {
            self.ac_kx[tbl] = kx;
        }
    }

    /// Decode DC coefficient (ITU-T T.81 Figure F.19-F.24).
    pub fn decode_dc(&mut self, ci: usize, tbl: usize) -> i16 {
        let ctx = self.dc_context[ci] as usize;

        // Figure F.19: Decode_DC_DIFF — is diff zero?
        if self.qm.decode(&mut self.dc_stats[tbl][ctx]) == 0 {
            self.dc_context[ci] = 0;
            return self.last_dc_val[ci] as i16;
        }

        // Figure F.22: sign
        let sign = self.qm.decode(&mut self.dc_stats[tbl][ctx + 1]);
        let mut st = ctx + 2 + sign as usize;

        // Figure F.23: magnitude category
        let mut m: i32 = self.qm.decode(&mut self.dc_stats[tbl][st]) as i32;
        if m != 0 {
            st = 20; // Table F.4: X1 = 20
            while self.qm.decode(&mut self.dc_stats[tbl][st]) != 0 {
                m <<= 1;
                if m == 0x8000 {
                    break;
                }
                st += 1;
            }
        }

        // Section F.1.4.4.1.2: DC context conditioning
        let (l, u) = self.dc_cond[tbl];
        let half_l = (1i32 << l) >> 1;
        let half_u = (1i32 << u) >> 1;
        self.dc_context[ci] = if m < half_l {
            0
        } else if m > half_u {
            12 + sign * 4
        } else {
            4 + sign * 4
        };

        // Figure F.24: magnitude bit pattern
        let mut v = m;
        st += 14;
        let mut m2 = m;
        while m2 > 1 {
            m2 >>= 1;
            if self.qm.decode(&mut self.dc_stats[tbl][st]) != 0 {
                v |= m2;
            }
        }

        v += 1;
        if sign != 0 {
            v = -v;
        }

        let prev = self.last_dc_val[ci];
        let new_dc = prev + v;
        self.last_dc_val[ci] = new_dc;
        new_dc as i16
    }

    /// Decode AC coefficients (ITU-T T.81 Figure F.20).
    ///
    /// Writes into `block[1..=se]` in zigzag order.
    pub fn decode_ac(&mut self, block: &mut [i16; 64], tbl: usize, se: u8) {
        let kx = self.ac_kx[tbl];
        let mut k: usize = 1;

        while k <= se as usize {
            let mut st = 3 * (k - 1);

            // EOB flag
            if self.qm.decode(&mut self.ac_stats[tbl][st]) != 0 {
                break;
            }

            // Run of zeros
            while self.qm.decode(&mut self.ac_stats[tbl][st + 1]) == 0 {
                st += 3;
                k += 1;
                if k > se as usize {
                    return;
                }
            }

            // Sign
            let sign = self.qm.decode(&mut self.fixed_bin[0]);

            // Magnitude category
            st += 2;
            let mut m: i32 = self.qm.decode(&mut self.ac_stats[tbl][st]) as i32;
            if m != 0 && self.qm.decode(&mut self.ac_stats[tbl][st]) != 0 {
                m <<= 1;
                st = if (k as u8) <= kx { 189 } else { 217 };
                while self.qm.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    m <<= 1;
                    if m == 0x8000 {
                        break;
                    }
                    st += 1;
                }
            }

            // Magnitude bit pattern
            let mut v = m;
            st += 14;
            let mut m2 = m;
            while m2 > 1 {
                m2 >>= 1;
                if self.qm.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    v |= m2;
                }
            }

            v += 1;
            if sign != 0 {
                v = -v;
            }

            block[k] = v as i16;
            k += 1;
        }
    }

    /// Decode a full block (DC + AC) for sequential mode.
    pub fn decode_block(&mut self, block: &mut [i16; 64], ci: usize, dc_tbl: usize, ac_tbl: usize) {
        block[0] = self.decode_dc(ci, dc_tbl);
        self.decode_ac(block, ac_tbl, 63);
    }

    /// Reset for restart marker.
    pub fn process_restart(&mut self) {
        self.dc_stats = [[0; DC_STAT_BINS]; NUM_ARITH_TBLS];
        self.ac_stats = [[0; AC_STAT_BINS]; NUM_ARITH_TBLS];
        self.last_dc_val = [0; 4];
        self.dc_context = [0; 4];
        self.qm.reset();
    }
}

// ─── JPEG Arithmetic Encoder ───────────────────────────────────────────────

/// JPEG arithmetic encoder wrapping QmEncoder with standard DC/AC context modeling.
pub struct JpegArithEncoder {
    qm: QmEncoder,
    dc_stats: [[u8; DC_STAT_BINS]; NUM_ARITH_TBLS],
    ac_stats: [[u8; AC_STAT_BINS]; NUM_ARITH_TBLS],
    fixed_bin: [u8; 4],
    last_dc_val: [i32; 4],
    dc_context: [u8; 4],
    dc_cond: [(u8, u8); NUM_ARITH_TBLS],
    ac_kx: [u8; NUM_ARITH_TBLS],
}

impl Default for JpegArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl JpegArithEncoder {
    pub fn new() -> Self {
        Self {
            qm: QmEncoder::new(),
            dc_stats: [[0; DC_STAT_BINS]; NUM_ARITH_TBLS],
            ac_stats: [[0; AC_STAT_BINS]; NUM_ARITH_TBLS],
            fixed_bin: [113, 0, 0, 0],
            last_dc_val: [0; 4],
            dc_context: [0; 4],
            dc_cond: [(0, 1); NUM_ARITH_TBLS],
            ac_kx: [5; NUM_ARITH_TBLS],
        }
    }

    /// Encode DC coefficient (mirror of decode_dc).
    pub fn encode_dc(&mut self, ci: usize, tbl: usize, val: i16) {
        let diff = val as i32 - self.last_dc_val[ci];
        self.last_dc_val[ci] = val as i32;

        let ctx = self.dc_context[ci] as usize;

        if diff == 0 {
            self.qm.encode(&mut self.dc_stats[tbl][ctx], 0); // zero
            self.dc_context[ci] = 0;
            return;
        }

        self.qm.encode(&mut self.dc_stats[tbl][ctx], 1); // nonzero

        // Sign
        let (sign, abs_diff) = if diff < 0 {
            (1u8, (-diff) as u32)
        } else {
            (0u8, diff as u32)
        };
        self.qm.encode(&mut self.dc_stats[tbl][ctx + 1], sign);

        let mut st = ctx + 2 + sign as usize;

        // Magnitude category: find m such that 2^m <= abs_diff-1 < 2^(m+1)
        let v = abs_diff - 1;
        let mut m: i32 = 0;
        let mut temp = v as i32;
        if temp > 0 {
            // First bit of magnitude category
            self.qm.encode(&mut self.dc_stats[tbl][st], 1);
            st = 20;
            temp >>= 1;
            while temp > 0 {
                self.qm.encode(&mut self.dc_stats[tbl][st], 1);
                m = (m << 1) | 1; // track magnitude for context conditioning
                st += 1;
                temp >>= 1;
            }
            self.qm.encode(&mut self.dc_stats[tbl][st], 0); // stop
            let _ = (m << 1) | 1;
        } else {
            self.qm.encode(&mut self.dc_stats[tbl][st], 0);
        }

        // DC context conditioning
        let m_for_ctx = if v > 0 {
            // m_for_ctx = magnitude of (abs_diff - 1), same as decoder's m
            let mut mc = 0i32;
            let mut t = v as i32;
            while t > 0 {
                mc = (mc << 1) | 1;
                t >>= 1;
            }
            mc
        } else {
            0
        };

        let (l, u) = self.dc_cond[tbl];
        let half_l = (1i32 << l) >> 1;
        let half_u = (1i32 << u) >> 1;
        self.dc_context[ci] = if m_for_ctx < half_l {
            0
        } else if m_for_ctx > half_u {
            12 + sign * 4
        } else {
            4 + sign * 4
        };

        // Magnitude bit pattern
        st += 14;
        let bits_in_v = if v > 0 {
            32 - v.leading_zeros()
        } else {
            0
        };
        let mut m2 = if bits_in_v > 1 {
            1i32 << (bits_in_v as i32 - 2)
        } else {
            0
        };
        while m2 > 0 {
            let bit = if (v as i32 & m2) != 0 { 1 } else { 0 };
            self.qm.encode(&mut self.dc_stats[tbl][st], bit);
            m2 >>= 1;
        }
    }

    /// Encode AC coefficients (mirror of decode_ac).
    ///
    /// Structure matches decoder exactly: outer loop emits EOB, inner loop
    /// emits zero-run bits until a nonzero coefficient.
    pub fn encode_ac(&mut self, block: &[i16; 64], tbl: usize, se: u8) {
        let kx = self.ac_kx[tbl];

        // Find last nonzero coefficient
        let mut eob = 0usize;
        for k in (1..=se as usize).rev() {
            if block[k] != 0 {
                eob = k;
                break;
            }
        }

        let mut k: usize = 1;
        while k <= se as usize {
            let mut st = 3 * (k - 1);

            // EOB flag (only at top of outer loop)
            if k > eob {
                self.qm.encode(&mut self.ac_stats[tbl][st], 1); // EOB
                return;
            }
            self.qm.encode(&mut self.ac_stats[tbl][st], 0); // not EOB

            // Zero-run: advance k for consecutive zeros
            while block[k] == 0 {
                self.qm.encode(&mut self.ac_stats[tbl][st + 1], 0); // zero
                st += 3;
                k += 1;
                if k > se as usize {
                    return;
                }
            }
            // Nonzero coefficient found
            self.qm.encode(&mut self.ac_stats[tbl][st + 1], 1); // nonzero

            // Sign
            let coeff = block[k];
            let (sign, abs_val) = if coeff < 0 {
                (1u8, (-coeff) as u32)
            } else {
                (0u8, coeff as u32)
            };
            self.qm.encode(&mut self.fixed_bin[0], sign);

            // Magnitude category (ITU-T T.81 Figure F.23)
            st += 2;
            let v = abs_val - 1;
            if v == 0 {
                self.qm.encode(&mut self.ac_stats[tbl][st], 0); // m=0
            } else {
                self.qm.encode(&mut self.ac_stats[tbl][st], 1); // m>=1
                let bits = 32 - v.leading_zeros();
                if bits <= 1 {
                    self.qm.encode(&mut self.ac_stats[tbl][st], 0); // stop at m=1
                } else {
                    self.qm.encode(&mut self.ac_stats[tbl][st], 1); // extend
                    st = if (k as u8) <= kx { 189 } else { 217 };
                    for _ in 0..bits - 2 {
                        self.qm.encode(&mut self.ac_stats[tbl][st], 1);
                        st += 1;
                    }
                    self.qm.encode(&mut self.ac_stats[tbl][st], 0); // stop
                }

                // Magnitude bit pattern (ITU-T T.81 Figure F.24)
                if bits > 1 {
                    st += 14;
                    let mut m2 = 1u32 << (bits - 2);
                    while m2 > 0 {
                        let bit = if (v & m2) != 0 { 1 } else { 0 };
                        self.qm.encode(&mut self.ac_stats[tbl][st], bit);
                        m2 >>= 1;
                    }
                }
            }

            k += 1;
        }
    }

    /// Encode a full block (DC + AC) for sequential mode.
    pub fn encode_block(&mut self, block: &[i16; 64], ci: usize, dc_tbl: usize, ac_tbl: usize) {
        self.encode_dc(ci, dc_tbl, block[0]);
        self.encode_ac(block, ac_tbl, 63);
    }

    /// Flush and return the encoded byte stream.
    pub fn finish(&mut self) -> Vec<u8> {
        self.qm.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qm_decoder_creation() {
        let data = [0u8; 100];
        let _dec = QmDecoder::new(&data);
    }

    #[test]
    fn qm_encoder_creation() {
        let _enc = QmEncoder::new();
    }

    #[test]
    fn qm_trace_10_decisions() {
        // Trace 10 decisions: the first byte_out should have temp > 0xFF (carry)
        let mut enc = QmEncoder::new();
        let mut sts = [0u8; 8];
        let bits = [1u8, 0, 0, 1, 0, 0, 1, 0, 0, 1]; // LPS every 3rd

        for (i, &bit) in bits.iter().enumerate() {
            let ctx = i % 8;
            enc.encode(&mut sts[ctx], bit);
            eprintln!(
                "after d{}: c=0x{:X}, a=0x{:X}, ct={}, buf_len={}",
                i,
                enc.c,
                enc.a,
                enc.ct,
                enc.buf.len()
            );
        }
        let encoded = enc.finish();
        eprintln!("encoded: {:02X?}", &encoded[..encoded.len().min(10)]);
    }

    #[test]
    fn qm_debug_first_bytes() {
        let mut enc = QmEncoder::new();
        let mut sts = [0u8; 8];
        for i in 0..2000usize {
            let ctx = i % 8;
            let bit = ((i * 7 + 3) % 3 == 0) as u8;
            enc.encode(&mut sts[ctx], bit);
        }
        let encoded = enc.finish();
        eprintln!("Encoded {} bytes", encoded.len());
        eprintln!("First 20 bytes: {:02X?}", &encoded[..20.min(encoded.len())]);
        // Check the first byte is NOT zero (carry should have propagated)
        assert_ne!(
            encoded[0], 0,
            "first byte should not be 0 after 2000 decisions with LPS"
        );
    }

    #[test]
    fn qm_roundtrip_many_decisions() {
        // The QM-coder's carry-propagating byte output needs enough decisions
        // for the code value to grow large enough to trigger carries. A few
        // hundred decisions (realistic for one JPEG block) is sufficient.
        let mut enc = QmEncoder::new();
        let mut sts_enc = [0u8; 8];
        let decisions: Vec<(usize, u8)> = (0..2000)
            .map(|i| (i % 8, ((i * 7 + 3) % 3 == 0) as u8))
            .collect();

        for &(ctx, bit) in &decisions {
            enc.encode(&mut sts_enc[ctx], bit);
        }
        let encoded = enc.finish();

        let mut dec = QmDecoder::new(&encoded);
        let mut sts_dec = [0u8; 8];
        for (i, &(ctx, expected)) in decisions.iter().enumerate() {
            let got = dec.decode(&mut sts_dec[ctx]);
            assert_eq!(got, expected, "mismatch at decision {i} ctx={ctx}");
        }
    }

    #[test]
    fn jpeg_arith_block_roundtrip() {
        let mut enc = JpegArithEncoder::new();
        let mut block = [0i16; 64];
        block[0] = 42; // DC
        block[1] = 10; // AC in zigzag
        block[2] = -5;
        block[5] = 3;

        enc.encode_block(&block, 0, 0, 0);
        let encoded = enc.finish();

        let mut dec = JpegArithDecoder::new(&encoded);
        let mut decoded = [0i16; 64];
        dec.decode_block(&mut decoded, 0, 0, 0);

        assert_eq!(decoded, block, "block roundtrip mismatch");
    }

    #[test]
    fn jpeg_arith_multi_block_roundtrip() {
        let mut enc = JpegArithEncoder::new();

        let mut b1 = [0i16; 64];
        b1[0] = 100;
        b1[1] = 20;
        b1[3] = -7;
        let mut b2 = [0i16; 64];
        b2[0] = 105;
        b2[2] = 15;
        let mut b3 = [0i16; 64];
        b3[0] = -50;
        b3[1] = 1;

        // Encode Y, Cb, Cr blocks
        enc.encode_block(&b1, 0, 0, 0);
        enc.encode_block(&b2, 1, 1, 1);
        enc.encode_block(&b3, 2, 1, 1);
        let encoded = enc.finish();

        let mut dec = JpegArithDecoder::new(&encoded);
        let mut d1 = [0i16; 64];
        let mut d2 = [0i16; 64];
        let mut d3 = [0i16; 64];
        dec.decode_block(&mut d1, 0, 0, 0);
        dec.decode_block(&mut d2, 1, 1, 1);
        dec.decode_block(&mut d3, 2, 1, 1);

        assert_eq!(d1, b1, "Y block mismatch");
        assert_eq!(d2, b2, "Cb block mismatch");
        assert_eq!(d3, b3, "Cr block mismatch");
    }
}
