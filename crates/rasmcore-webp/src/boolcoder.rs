//! Boolean arithmetic encoder (RFC 6386 Section 7).
//!
//! General-purpose boolean arithmetic coder usable for VP8 and other
//! applications requiring adaptive binary entropy coding.

// TODO: Implement in webp-bool-coder track:
// pub struct BoolWriter { ... }
// impl BoolWriter {
//     pub fn new() -> Self
//     pub fn with_capacity(capacity: usize) -> Self
//     pub fn put_bit(&mut self, prob: u8, bit: bool)
//     pub fn put_bit_uniform(&mut self, bit: bool)
//     pub fn put_literal(&mut self, n_bits: u8, value: u32)
//     pub fn put_signed(&mut self, n_bits: u8, value: i32)
//     pub fn finish(self) -> Vec<u8>
//     pub fn size_estimate(&self) -> usize
// }
