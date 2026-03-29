//! JPEG type definitions covering ALL modes from ITU-T T.81.

/// Input pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 bytes per pixel (R, G, B).
    Rgb8,
    /// 4 bytes per pixel (R, G, B, A) — alpha discarded during encode.
    Rgba8,
    /// 1 byte per pixel (grayscale).
    Gray8,
}

/// Chroma subsampling mode.
///
/// Determines how color (Cb, Cr) channels are sampled relative to luma (Y).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:2:0 — Cb/Cr sampled at half resolution in both dimensions.
    /// Most common for photographic content. Default.
    Quarter420,
    /// 4:2:2 — Cb/Cr sampled at half horizontal, full vertical resolution.
    /// Better for horizontal edges.
    Half422,
    /// 4:4:4 — No subsampling. Full color resolution.
    /// Best quality, largest files.
    None444,
    /// 4:1:1 — Cb/Cr sampled at quarter horizontal, full vertical.
    /// Rarely used in JPEG.
    Quarter411,
}

/// Sample precision (bits per component).
///
/// ITU-T T.81 supports 8-bit (baseline) and 12-bit (extended) precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplePrecision {
    /// 8-bit precision (baseline JPEG). Default.
    Eight,
    /// 12-bit precision (extended JPEG). Rare, used in medical imaging.
    Twelve,
}

/// Custom quantization table (8x8 = 64 values).
///
/// Values are in zigzag order as stored in the JPEG stream.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// 64 quantization values in zigzag scan order.
    pub values: [u16; 64],
}

/// Custom quantization table pair (luminance + chrominance).
#[derive(Debug, Clone)]
pub struct CustomQuantTables {
    /// Luminance (Y) quantization table.
    pub luminance: QuantTable,
    /// Chrominance (Cb, Cr) quantization table.
    pub chrominance: QuantTable,
}

/// JPEG encode configuration — covers ALL modes from ITU-T T.81.
///
/// Designed to expose every JPEG parameter upfront so subsequent
/// implementation tracks can add support incrementally without API changes.
///
/// **Default behavior** (turbo=false): enables trellis quantization,
/// optimized Huffman tables, and EOB optimization for mozjpeg-quality output.
///
/// **Turbo mode** (turbo=true): disables all multi-pass optimizations for
/// 3-10x faster encoding. Produces libjpeg-equivalent output.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// Quality level 1-100. Maps to quantization table scaling.
    /// Default: 85.
    pub quality: u8,

    /// Turbo mode: skip all multi-pass optimizations for maximum speed.
    ///
    /// When `true`: forces single-pass encode with standard Huffman tables,
    /// no trellis, no EOB optimization. Overrides `trellis`, `optimize_huffman`,
    /// and `eob_optimize` to `false`.
    ///
    /// When `false` (default): respects individual optimization flags.
    /// Default: false.
    pub turbo: bool,

    /// Emit progressive JPEG (spectral selection + successive approximation).
    /// Default: false (sequential/baseline).
    pub progressive: bool,

    /// Chroma subsampling mode.
    /// Default: Quarter420 (4:2:0).
    pub subsampling: ChromaSubsampling,

    /// Use arithmetic coding instead of Huffman.
    /// Default: false (Huffman).
    pub arithmetic_coding: bool,

    /// Restart interval in MCU count. None = no restart markers.
    /// Restart markers enable error resilience and parallel decode.
    pub restart_interval: Option<u16>,

    /// Two-pass Huffman optimization (compute optimal tables from actual data).
    /// Default: true (mozjpeg-quality output).
    pub optimize_huffman: bool,

    /// Trellis quantization (rate-distortion optimization per DCT block).
    /// Produces smaller files at same quality but slower encode.
    /// Default: true (mozjpeg-quality output).
    pub trellis: bool,

    /// EOB block-level optimization — zero entire blocks where encoding cost
    /// exceeds distortion penalty. Requires `optimize_huffman: true`.
    /// Default: true (mozjpeg-quality output).
    pub eob_optimize: bool,

    /// Sample precision (8-bit baseline or 12-bit extended).
    /// Default: Eight.
    pub sample_precision: SamplePrecision,

    /// Quantization table preset. Default: Robidoux (mozjpeg/ImageMagick).
    /// Use `QuantPreset::AnnexK` for standard libjpeg compatibility.
    pub quant_preset: crate::quantize::QuantPreset,

    /// Custom quantization tables override. When set, overrides quant_preset.
    pub custom_quant_tables: Option<CustomQuantTables>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            turbo: false,
            progressive: false,
            subsampling: ChromaSubsampling::Quarter420,
            arithmetic_coding: false,
            restart_interval: None,
            optimize_huffman: true,
            trellis: true,
            eob_optimize: true,
            sample_precision: SamplePrecision::Eight,
            quant_preset: crate::quantize::QuantPreset::default(),
            custom_quant_tables: None,
        }
    }
}

impl EncodeConfig {
    /// Turbo mode: maximum encode throughput.
    ///
    /// Disables all optional quality optimizations (trellis, progressive,
    /// Huffman optimization, arithmetic coding). Uses standard quantization
    /// tables and 4:2:0 subsampling for fastest possible encode.
    ///
    /// Typical speedup: 3-10x over default (trellis + optimize_huffman).
    pub fn turbo(quality: u8) -> Self {
        Self {
            quality,
            turbo: true,
            quant_preset: crate::quantize::QuantPreset::AnnexK,
            ..Default::default()
        }
    }

    /// Returns the effective config after applying turbo overrides.
    ///
    /// When `turbo=true`, forces `trellis=false`, `optimize_huffman=false`,
    /// `eob_optimize=false`.
    pub fn effective(&self) -> Self {
        let mut c = self.clone();
        if c.turbo {
            c.trellis = false;
            c.optimize_huffman = false;
            c.eob_optimize = false;
        }
        c
    }

    /// Quality preset: balanced quality with optimizations.
    ///
    /// Enables trellis quantization and Huffman optimization for
    /// best quality-to-size ratio. Uses Robidoux quant tables.
    pub fn quality(quality: u8) -> Self {
        Self {
            quality,
            trellis: true,
            optimize_huffman: true,
            ..Default::default()
        }
    }
}
