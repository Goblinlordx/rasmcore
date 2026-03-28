# JPEG Encoder Design Document — rasmcore-jpeg

## Overview

This document specifies the architecture for `rasmcore-jpeg`, a pure Rust JPEG
encoder implementing ITU-T T.81 (ISO/IEC 10918-1) with complete mode coverage.
Replaces the AGPL-licensed zenjpeg dependency. Zero runtime dependencies except
rasmcore-color (color conversion) and rasmcore-bitio (bit-level I/O).

**Design principles:**
- Complete ITU-T T.81 mode support (baseline, extended, progressive, arithmetic)
- Pure Rust, no C dependencies, compiles to wasm32-wasip2
- Each module is reusable with a clean public API
- Deterministic output (same input → identical bytes)
- Quality-per-byte parity with libjpeg-turbo, target parity with mozjpeg via trellis

---

## 1. JPEG Encoding Modes (ITU-T T.81 Complete Catalog)

### Frame Types and SOF Markers

| SOF | Mode | Precision | Entropy | Description |
|-----|------|-----------|---------|-------------|
| SOF0 (0xFFC0) | Baseline sequential | 8-bit | Huffman | Most common, required by all decoders |
| SOF1 (0xFFC1) | Extended sequential | 8 or 12-bit | Huffman | Extended precision |
| SOF2 (0xFFC2) | Progressive | 8 or 12-bit | Huffman | Multi-scan incremental display |
| SOF3 (0xFFC3) | Lossless | 2-16 bit | Huffman | Predictive, no DCT |
| SOF9 (0xFFC9) | Extended sequential | 8 or 12-bit | Arithmetic | Better compression |
| SOF10 (0xFFCA) | Progressive | 8 or 12-bit | Arithmetic | Best compression |
| SOF11 (0xFFCB) | Lossless | 2-16 bit | Arithmetic | Lossless + arithmetic |

### Parameters per Mode

- **Sample precision**: 8-bit (baseline), 8 or 12-bit (extended/progressive)
- **Components**: 1 (grayscale), 3 (YCbCr), 4 (CMYK/YCCK)
- **Chroma subsampling**: Sampling factors H×V per component
  - 4:4:4 → H=1,V=1 for all components
  - 4:2:2 → Y: H=2,V=1; Cb/Cr: H=1,V=1
  - 4:2:0 → Y: H=2,V=2; Cb/Cr: H=1,V=1
  - 4:1:1 → Y: H=4,V=1; Cb/Cr: H=1,V=1
- **Quantization tables**: Up to 4 tables (8-bit or 16-bit precision)
- **Huffman tables**: Up to 4 DC + 4 AC tables
- **Restart interval**: 0 (disabled) or N (MCU count between RST markers)
- **Progressive scans**: Spectral selection (Ss, Se) + successive approx (Ah, Al)

---

## 2. Encoding Pipeline

```
Input Pixels (RGB/RGBA/Gray)
    │
    ▼
┌─────────────────┐
│  Color Convert   │  RGB → YCbCr (BT.601) via rasmcore-color
│  (uses shared)   │  Gray passthrough for 1-component
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chroma          │  Downsample Cb/Cr per subsampling mode
│  Subsample       │  4:4:4, 4:2:2, 4:2:0, 4:1:1
│  (subsample.rs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MCU Iterator    │  Partition into Minimum Coded Units
│  (mcu.rs)        │  MCU size depends on subsampling
└────────┬────────┘
         │
    ┌────┴────────────┐  (per 8×8 block within MCU)
    ▼                 ▼
┌────────┐     ┌───────────┐
│  8×8    │     │  Quantize  │  Scale by quality-derived table
│  DCT    │────▶│  (quant.rs)│  Optional: trellis optimization
│ (dct.rs)│     └─────┬─────┘
└────────┘           │
                     ▼
              ┌───────────┐
              │  Zigzag +  │  DC: DPCM difference coding
              │  RLE       │  AC: (run, size) pairs + EOB/ZRL
              │  (scan.rs) │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │  Entropy   │  Huffman (default or optimized)
              │  Coding    │  OR Arithmetic (QM-coder)
              │(huffman.rs)│
              │(arith.rs)  │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │  Container │  JFIF markers: SOI, APP0, DQT,
              │  Assembly  │  SOF, DHT/DAC, SOS, RST, EOI
              │(container.rs)│
              └───────────┘
```

---

## 3. Module Structure — crates/rasmcore-jpeg/

```
src/
├── lib.rs              # Public API: encode(), EncodeConfig
├── config.rs           # EncodeConfig with ALL mode parameters
├── error.rs            # EncodeError type
├── dct.rs              # 8×8 forward/inverse DCT (AAN integer algorithm)
├── quant.rs            # Quantization tables, quality scaling, zigzag
├── huffman.rs          # Huffman table generation + entropy encoder
├── arith.rs            # QM-coder arithmetic entropy encoder
├── scan.rs             # DC DPCM + AC run-length encoding
├── progressive.rs      # Multi-scan progressive mode logic
├── subsample.rs        # Chroma subsampling (4:2:0, 4:2:2, 4:4:4, 4:1:1)
├── mcu.rs              # MCU partitioning and block iteration
├── container.rs        # JFIF marker assembly
├── trellis.rs          # Trellis quantization (rate-distortion optimization)
└── tables.rs           # Standard quantization + Huffman tables (Annex K)
```

Each module exposes a clean public API designed for reuse:
- `dct` provides general-purpose 8×8 integer DCT/IDCT
- `huffman` provides Huffman table generation usable by other codecs
- `arith` provides QM-coder usable for other JPEG-family codecs
- `quant` provides JPEG quantization with configurable tables

---

## 4. Standard Tables (ITU-T T.81 Annex K)

### 4.1 Standard Luminance Quantization Table (Table K.1)

```
 16  11  10  16  24  40  51  61
 12  12  14  19  26  58  60  55
 14  13  16  24  40  57  69  56
 14  17  22  29  51  87  80  62
 18  22  37  56  68 109 103  77
 24  35  55  64  81 104 113  92
 49  64  78  87 103 121 120 101
 72  92  95  98 112 100 103  99
```

### 4.2 Standard Chrominance Quantization Table (Table K.2)

```
 17  18  24  47  99  99  99  99
 18  21  26  66  99  99  99  99
 24  26  56  99  99  99  99  99
 47  66  99  99  99  99  99  99
 99  99  99  99  99  99  99  99
 99  99  99  99  99  99  99  99
 99  99  99  99  99  99  99  99
 99  99  99  99  99  99  99  99
```

### 4.3 Quality Scaling (libjpeg formula)

```
if quality < 50:
    scale_factor = 5000 / quality
else:
    scale_factor = 200 - 2 * quality

For each table entry:
    scaled = (base_value * scale_factor + 50) / 100
    final = clamp(scaled, 1, 255)  // 1-255 for 8-bit, 1-65535 for 16-bit
```

Quality 50 → scale=100 (use tables as-is). Quality 75 → scale=50 (halve values).
Quality 95 → scale=10 (divide by 10). Quality 100 → all ones (minimal quantization).

### 4.4 Zigzag Scan Order (64 elements)

```
 0  1  5  6 14 15 27 28
 2  4  7 13 16 26 29 42
 3  8 12 17 25 30 41 43
 9 11 18 24 31 40 44 53
10 19 23 32 39 45 52 54
20 22 33 38 46 51 55 60
21 34 37 47 50 56 59 61
35 36 48 49 57 58 62 63
```

Linear: [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

### 4.5 Standard Huffman Tables (Annex K, Tables K.3-K.6)

**DC Luminance (Table K.3):**
- Bits: [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
- Values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

**DC Chrominance (Table K.4):**
- Bits: [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
- Values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

**AC Luminance (Table K.5):**
- Bits: [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1]
- Values: [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA]

**AC Chrominance (Table K.6):**
- Bits: [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2]
- Values: [0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA]

---

## 5. Module Specifications

### 5.1 8×8 DCT (`dct.rs`)

**Public API:**
```rust
/// Forward 8×8 DCT on a block of pixels (level-shifted by -128 for 8-bit).
pub fn forward_dct_8x8(block: &[i16; 64], out: &mut [i32; 64]);

/// Inverse 8×8 DCT (for verification/reconstruction).
pub fn inverse_dct_8x8(coeffs: &[i32; 64], out: &mut [i16; 64]);
```

**Algorithm:** AAN (Arai, Agui, Nakajima) fast DCT.
- 1D transform: 5 multiplies, 29 additions
- 2D: row transform, then column transform
- Uses fixed-point scaled constants to avoid post-DCT division
- Integer version: 13-bit coefficients, 32-bit intermediate

**Mathematical definition (ITU-T T.81 Section A.3.3):**
```
F(u,v) = (1/4) * C(u) * C(v) * ΣΣ f(x,y) * cos((2x+1)*u*π/16) * cos((2y+1)*v*π/16)
```
Where C(0) = 1/√2, C(k) = 1 for k > 0.

### 5.2 Quantization (`quant.rs`)

**Public API:**
```rust
/// Build quantization table scaled by quality factor.
pub fn build_quant_table(base_table: &[u8; 64], quality: u8) -> [u16; 64];

/// Quantize a DCT block: out[i] = round(coeffs[i] / table[i]).
pub fn quantize(coeffs: &[i32; 64], table: &[u16; 64], out: &mut [i16; 64]);

/// Dequantize: out[i] = quantized[i] * table[i].
pub fn dequantize(quantized: &[i16; 64], table: &[u16; 64], out: &mut [i32; 64]);
```

### 5.3 Huffman Coding (`huffman.rs`)

**Public API:**
```rust
/// Standard Huffman table from Annex K.
pub fn default_dc_luma_table() -> HuffmanTable;
pub fn default_dc_chroma_table() -> HuffmanTable;
pub fn default_ac_luma_table() -> HuffmanTable;
pub fn default_ac_chroma_table() -> HuffmanTable;

/// Build optimized Huffman table from symbol frequencies.
pub fn build_optimized_table(frequencies: &[u32; 256]) -> HuffmanTable;

/// Encode a symbol using the given Huffman table.
pub fn encode_symbol(writer: &mut BitWriter, table: &HuffmanTable, symbol: u8);

/// Encode additional bits (raw value after Huffman symbol).
pub fn encode_extra_bits(writer: &mut BitWriter, value: i16, size: u8);
```

### 5.4 Arithmetic Coding (`arith.rs`)

**Public API:**
```rust
/// QM-coder arithmetic encoder (ITU-T T.81 Annex D).
pub struct ArithEncoder { ... }

impl ArithEncoder {
    pub fn new() -> Self;
    pub fn encode_bit(&mut self, context: &mut Context, bit: bool);
    pub fn encode_dc(&mut self, contexts: &mut DcContexts, diff: i16);
    pub fn encode_ac(&mut self, contexts: &mut AcContexts, coeffs: &[i16; 64]);
    pub fn finish(self) -> Vec<u8>;
}
```

**QM-coder:** Binary arithmetic coder with adaptive probability estimation.
113-entry probability estimation table (Annex D, Table D.3).
Renormalization via conditional exchange (MPS/LPS swap).

### 5.5 Container Assembly (`container.rs`)

**Public API:**
```rust
/// Write a complete JFIF file.
pub fn write_jfif(
    writer: &mut Vec<u8>,
    config: &EncodeConfig,
    quant_tables: &[QuantTable],
    huffman_tables: &[HuffmanTable],
    scan_data: &[ScanData],
    frame_info: &FrameInfo,
);
```

**Marker Bytes:**
| Marker | Code | Description |
|--------|------|-------------|
| SOI | 0xFFD8 | Start of image |
| APP0 | 0xFFE0 | JFIF header (version, density, thumbnail) |
| DQT | 0xFFDB | Define quantization table(s) |
| SOF0 | 0xFFC0 | Start of frame (baseline) |
| SOF1 | 0xFFC1 | Start of frame (extended) |
| SOF2 | 0xFFC2 | Start of frame (progressive) |
| SOF9 | 0xFFC9 | Start of frame (arithmetic sequential) |
| SOF10 | 0xFFCA | Start of frame (arithmetic progressive) |
| DHT | 0xFFC4 | Define Huffman table(s) |
| DAC | 0xFFCC | Define arithmetic conditioning |
| SOS | 0xFFDA | Start of scan |
| DRI | 0xFFDD | Define restart interval |
| RST0-7 | 0xFFD0-D7 | Restart markers (cycle every 8) |
| EOI | 0xFFD9 | End of image |
| COM | 0xFFFE | Comment |

### 5.6 Progressive Mode (`progressive.rs`)

**Scan configurations:**
```rust
pub struct ScanSpec {
    pub components: Vec<u8>,     // Component indices in this scan
    pub ss: u8,                  // Spectral selection start (0-63)
    pub se: u8,                  // Spectral selection end (0-63)
    pub ah: u8,                  // Successive approximation high bit
    pub al: u8,                  // Successive approximation low bit
}
```

**Default progressive scan order (matches libjpeg):**
1. DC all components: Ss=0, Se=0, Ah=0, Al=0
2. Y AC 1-5: Ss=1, Se=5, Ah=0, Al=2
3. Cb AC 1-63: Ss=1, Se=63, Ah=0, Al=1
4. Cr AC 1-63: Ss=1, Se=63, Ah=0, Al=1
5. Y AC 6-63: Ss=6, Se=63, Ah=0, Al=2
6. Y AC 1-63 refine: Ss=1, Se=63, Ah=2, Al=1
7. Y AC 1-63 final: Ss=1, Se=63, Ah=1, Al=0
...etc (total ~10-15 scans for optimal progressive display)

---

## 6. C-Specific Patterns → Rust Alternatives

| C Pattern (libjpeg-turbo) | Rust Alternative |
|---|---|
| `JSAMPLE` typedef (unsigned char) | `u8` for 8-bit, `u16` for 12-bit |
| Global `jpeg_compress_struct` | `Encoder` struct with owned state |
| `jpeg_memory_mgr` pool allocator | `Vec<T>` with standard allocator |
| Function pointer dispatch (SIMD) | Direct functions + `#[cfg]` for SIMD |
| `JMETHOD` macro for vtable | Trait or enum dispatch |
| `setjmp`/`longjmp` error handling | `Result<T, EncodeError>` |
| Byte stuffing via pointer manipulation | Iterator-based byte stuffing |
| MCU buffer as flat array with stride | `&[u8]` slices with explicit width |
| Temporary coefficient buffers on stack | Stack arrays `[i16; 64]` or `[i32; 64]` |
| `#define DCTSIZE 8` | `const BLOCK_SIZE: usize = 8` |

---

## 7. Public API — crates/rasmcore-jpeg/src/lib.rs

```rust
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError>;

pub enum PixelFormat { Rgb8, Rgba8, Gray8 }

pub struct EncodeConfig {
    pub quality: u8,                          // 1-100
    pub progressive: bool,                    // Multi-scan progressive
    pub subsampling: ChromaSubsampling,       // 4:4:4, 4:2:2, 4:2:0, 4:1:1
    pub arithmetic_coding: bool,              // Use arithmetic instead of Huffman
    pub optimize_huffman: bool,               // Two-pass for optimal tables
    pub trellis: bool,                        // Rate-distortion optimization
    pub restart_interval: Option<u16>,        // MCU count between RST markers
    pub sample_precision: SamplePrecision,    // 8-bit or 12-bit
}

pub enum ChromaSubsampling { Cs444, Cs422, Cs420, Cs411 }
pub enum SamplePrecision { Eight, Twelve }
```

---

## 8. Complexity Estimates

| Module | Relative Complexity | Key Challenge |
|---|---|---|
| `dct.rs` | Medium | AAN algorithm correctness, fixed-point precision |
| `quant.rs` | Low | Standard table scaling, zigzag ordering |
| `huffman.rs` | Medium | Optimized table generation, canonical codes |
| `arith.rs` | High | QM-coder probability estimation, carry handling |
| `scan.rs` | Low | DC DPCM, AC run-length, straightforward |
| `progressive.rs` | High | Multi-scan state, successive approximation refinement |
| `subsample.rs` | Low | Averaging filters, straightforward |
| `mcu.rs` | Medium | Block extraction for all subsampling modes |
| `container.rs` | Medium | Marker assembly, byte stuffing |
| `trellis.rs` | High | Dynamic programming, rate-distortion cost model |
| `tables.rs` | Low | Static data transcription |

**Total estimated LOC:** ~4,000-5,000 lines of Rust (excluding tests).

---

## 9. SIMD Strategy (Future)

Phase 1 (these tracks): scalar-only for correctness.

Phase 2 (future track): SIMD optimization targets:
- `forward_dct_8x8` — 8-wide butterfly operations
- `quantize` — parallel divide/multiply
- Color conversion — 4-wide RGB→YCbCr
- Chroma subsampling — 8-wide averaging

WASM: SIMD128 (same approach as rasmcore-webp).
Native: auto-vectorization handles many cases in release builds.
