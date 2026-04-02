# LUT Format Extension Guide

For step-by-step instructions on adding new LUT encoder and decoder
formats (`.spi1d`, `.spi3d`, `.mga`, etc.), see:

- **[docs/extending.md — Adding a LUT Encoder](../../docs/extending.md#adding-a-lut-encoder)**
- **[docs/extending.md — Adding a LUT Decoder](../../docs/extending.md#adding-a-lut-decoder)**

## Key patterns

- LUT encoders use `StaticLutEncoderRegistration` + `inventory::submit!`
  (not the `#[register_encoder]` macro used by image encoders)
- LUT decoders use `parse_*()` functions in `color_lut.rs` + format
  detection in `decoder/mod.rs` (not the `#[register_decoder]` macro)
- Pipeline short-circuit is automatic: `sink.rs` checks `is_lut_format()`
  and encodes the fused `ColorLut3D` directly when the output is a LUT format
