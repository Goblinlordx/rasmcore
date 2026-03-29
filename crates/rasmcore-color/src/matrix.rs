//! Color conversion coefficient matrices.
//!
//! Each standard (BT.601, BT.709, BT.2020) defines different luma/chroma
//! weights. The codec spec dictates which to use — JPEG and VP8 use BT.601,
//! H.264/AV1 use BT.709, HDR content uses BT.2020.

/// Integer coefficient matrix for RGB↔YCbCr conversion.
///
/// Forward (RGB → YCbCr):
/// ```text
///   Y  = ( yr*R +  yg*G +  yb*B + 128) >> 8 + y_offset
///   Cb = (cbr*R + cbg*G + cbb*B + 128) >> 8 + c_offset
///   Cr = (crr*R + crg*G + crb*B + 128) >> 8 + c_offset
/// ```
///
/// All coefficients are scaled by 256 for fixed-point integer arithmetic.
#[derive(Debug, Clone, Copy)]
pub struct ColorMatrix {
    // Y = (yr*R + yg*G + yb*B + 128) >> 8 + y_offset
    pub yr: i32,
    pub yg: i32,
    pub yb: i32,
    pub y_offset: i32,

    // Cb = (cbr*R + cbg*G + cbb*B + 128) >> 8 + c_offset
    pub cbr: i32,
    pub cbg: i32,
    pub cbb: i32,

    // Cr = (crr*R + crg*G + crb*B + 128) >> 8 + c_offset
    pub crr: i32,
    pub crg: i32,
    pub crb: i32,

    pub c_offset: i32,

    // Inverse coefficients for YCbCr → RGB
    // R = (y_scale*(Y - y_offset) + cr_r*(Cr - c_offset) + 128) >> 8
    // G = (y_scale*(Y - y_offset) + cb_g*(Cb - c_offset) + cr_g*(Cr - c_offset) + 128) >> 8
    // B = (y_scale*(Y - y_offset) + cb_b*(Cb - c_offset) + 128) >> 8
    pub inv_y_scale: i32,
    pub inv_cr_r: i32,
    pub inv_cb_g: i32,
    pub inv_cr_g: i32,
    pub inv_cb_b: i32,
}

impl ColorMatrix {
    /// ITU-R BT.601 — SD video, JPEG, VP8/WebP.
    ///
    /// Luma weights: 0.299 R + 0.587 G + 0.114 B
    pub const BT601: Self = Self {
        yr: 66,
        yg: 129,
        yb: 25,
        y_offset: 16,
        cbr: -38,
        cbg: -74,
        cbb: 112,
        crr: 112,
        crg: -94,
        crb: -18,
        c_offset: 128,
        // Inverse: derived from the forward matrix
        // Y' = Y - 16, Cb' = Cb - 128, Cr' = Cr - 128
        // R = 1.164*Y' + 1.596*Cr' → (298*Y' + 409*Cr' + 128) >> 8
        // G = 1.164*Y' - 0.392*Cb' - 0.813*Cr' → (298*Y' - 100*Cb' - 208*Cr' + 128) >> 8
        // B = 1.164*Y' + 2.017*Cb' → (298*Y' + 516*Cb' + 128) >> 8
        inv_y_scale: 298,
        inv_cr_r: 409,
        inv_cb_g: -100,
        inv_cr_g: -208,
        inv_cb_b: 516,
    };

    /// BT.601 Full Range — VP8/WebP, MJPEG.
    ///
    /// Same luma weights as BT.601 but Y uses full 0-255 range (no 16-235 headroom).
    /// VP8 (RFC 6386) operates on full-range Y internally.
    pub const BT601_FULL: Self = Self {
        yr: 77,      // 0.299 * 256 ≈ 77
        yg: 150,     // 0.587 * 256 ≈ 150
        yb: 29,      // 0.114 * 256 ≈ 29
        y_offset: 0, // FULL RANGE: Y ∈ [0, 255]
        cbr: -43,    // -0.169 * 256
        cbg: -85,    // -0.331 * 256
        cbb: 128,    // 0.500 * 256
        crr: 128,    // 0.500 * 256
        crg: -107,   // -0.419 * 256
        crb: -21,    // -0.081 * 256
        c_offset: 128,
        // Inverse (full range): no 1.164 scaling, direct
        // R = Y + 1.402*Cr' → (256*Y + 359*Cr' + 128) >> 8
        // G = Y - 0.344*Cb' - 0.714*Cr' → (256*Y - 88*Cb' - 183*Cr' + 128) >> 8
        // B = Y + 1.772*Cb' → (256*Y + 454*Cb' + 128) >> 8
        inv_y_scale: 256,
        inv_cr_r: 359,
        inv_cb_g: -88,
        inv_cr_g: -183,
        inv_cb_b: 454,
    };

    /// ITU-R BT.709 — HD video, H.264, H.265, AV1.
    ///
    /// Luma weights: 0.2126 R + 0.7152 G + 0.0722 B
    pub const BT709: Self = Self {
        yr: 47,
        yg: 157,
        yb: 16,
        y_offset: 16,
        cbr: -26,
        cbg: -87,
        cbb: 112,
        crr: 112,
        crg: -102,
        crb: -10,
        c_offset: 128,
        // R = 1.164*Y' + 1.793*Cr' → (298*Y' + 459*Cr' + 128) >> 8
        // G = 1.164*Y' - 0.213*Cb' - 0.533*Cr' → (298*Y' - 55*Cb' - 136*Cr' + 128) >> 8
        // B = 1.164*Y' + 2.112*Cb' → (298*Y' + 541*Cb' + 128) >> 8
        inv_y_scale: 298,
        inv_cr_r: 459,
        inv_cb_g: -55,
        inv_cr_g: -136,
        inv_cb_b: 541,
    };

    /// ITU-R BT.2020 — UHD/HDR video.
    ///
    /// Luma weights: 0.2627 R + 0.6780 G + 0.0593 B
    /// Scaled to studio range: Y' = round(Kr * 219) etc.
    ///   yr = round(0.2627 * 219) = 58
    ///   yg = round(0.6780 * 219) = 148
    ///   yb = round(0.0593 * 219) = 13 → sum = 219 ✓
    pub const BT2020: Self = Self {
        yr: 58,
        yg: 148,
        yb: 13,
        y_offset: 16,
        // Cb = round(-0.2627/(2*(1-0.0593)) * 224) * R + ...
        cbr: -31,
        cbg: -81,
        cbb: 112,
        crr: 112,
        crg: -103,
        crb: -9,
        c_offset: 128,
        // Inverse derived from forward:
        // R = (298*Y' + 459*Cr' + 128) >> 8  (Cr coeff = 256*1.4746/(219/256))
        // Approximation using standard inverse:
        inv_y_scale: 298,
        inv_cr_r: 430,
        inv_cb_g: -48,
        inv_cr_g: -166,
        inv_cb_b: 548,
    };
}
