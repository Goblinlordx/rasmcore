//! VP8 quantization and quality mapping (RFC 6386 Section 14.2).
//!
//! Maps quality (1-100) to VP8 quantizer parameters matching libwebp behavior.

/// Quantizer type — six distinct quantizer channels in VP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// Y-plane DC coefficient.
    YDc,
    /// Y-plane AC coefficients.
    YAc,
    /// Y2 (DC of DCs) DC coefficient.
    Y2Dc,
    /// Y2 AC coefficients.
    Y2Ac,
    /// UV-plane DC coefficient.
    UvDc,
    /// UV-plane AC coefficients.
    UvAc,
}

/// Quantization matrix for a single block type.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    /// Quantizer step sizes per coefficient position.
    pub q: [u16; 16],
    /// Inverse quantizer (fixed-point 1/q for fast division).
    pub iq: [u16; 16],
    /// Rounding bias per coefficient position.
    pub bias: [u32; 16],
    /// Zero threshold — skip quantization if |coeff| < threshold.
    pub zthresh: [u16; 16],
}

// TODO: Implement in webp-dct-quant track:
// pub fn build_matrix(qp: u8, qtype: QuantType) -> QuantMatrix
// pub fn quantize_block(coeffs: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16]) -> i32
// pub fn dequantize_block(quantized: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16])
// pub fn quality_to_qp(quality: u8) -> u8
