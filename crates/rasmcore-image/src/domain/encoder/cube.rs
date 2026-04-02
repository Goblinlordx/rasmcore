//! .cube 3D LUT encoder — serialize a ColorLut3D to Adobe/Resolve .cube format.
//!
//! This encoder short-circuits the pixel pipeline: instead of executing the
//! graph and encoding pixels, it extracts the fused ColorLut3D from the
//! color op chain and serializes it directly to .cube text format.

use crate::domain::color_lut::ColorLut3D;
use crate::domain::error::ImageError;

/// Configuration for .cube LUT export.
#[derive(Debug, Clone)]
pub struct CubeExportConfig {
    /// Grid size for the 3D LUT (default: 33, industry standard for 8-bit SDR).
    pub grid_size: usize,
    /// Optional title line in the .cube file.
    pub title: Option<String>,
}

impl Default for CubeExportConfig {
    fn default() -> Self {
        Self {
            grid_size: 33,
            title: None,
        }
    }
}

/// Serialize a ColorLut3D to .cube text format.
///
/// Output follows the Adobe/Resolve .cube specification:
/// - Optional TITLE line
/// - LUT_3D_SIZE N
/// - N^3 lines of "R G B" triplets (6 decimal places, values in [0.0, 1.0])
/// - R varies fastest, then G, then B (matches internal storage order)
pub fn serialize_cube(lut: &ColorLut3D, config: &CubeExportConfig) -> String {
    let n = lut.grid_size;
    // Pre-allocate: header ~50 bytes + ~25 bytes per entry (e.g., "0.123456 0.654321 0.987654\n")
    let capacity = 100 + n * n * n * 28;
    let mut out = String::with_capacity(capacity);

    // Header
    if let Some(title) = &config.title {
        out.push_str("TITLE \"");
        out.push_str(title);
        out.push_str("\"\n");
    }
    out.push_str(&format!("LUT_3D_SIZE {n}\n"));
    out.push('\n');

    // Data: B outer, G middle, R inner (matches ColorLut3D storage order)
    for entry in &lut.data {
        out.push_str(&format!(
            "{:.6} {:.6} {:.6}\n",
            entry[0], entry[1], entry[2]
        ));
    }

    out
}

/// Serialize a ColorLut3D to .cube format as bytes (UTF-8).
pub fn serialize_cube_bytes(lut: &ColorLut3D, config: &CubeExportConfig) -> Vec<u8> {
    serialize_cube(lut, config).into_bytes()
}

/// Encode a ColorLut3D to .cube format. This is the encoder entry point
/// called by the pipeline write path when format == "cube".
pub fn encode(lut: &ColorLut3D) -> Result<Vec<u8>, ImageError> {
    let config = CubeExportConfig::default();
    Ok(serialize_cube_bytes(lut, &config))
}

inventory::submit! {
    &crate::domain::encoder::StaticLutEncoderRegistration {
        format: "cube",
        extensions: &["cube"],
        encode_fn: encode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::color_lut::parse_cube_lut;

    #[test]
    fn identity_roundtrip_exact() {
        let identity = ColorLut3D::identity(17);
        let cube_text = serialize_cube(&identity, &CubeExportConfig::default());

        // Parse it back
        let parsed = parse_cube_lut(&cube_text).unwrap();
        assert_eq!(parsed.grid_size, 17);
        assert_eq!(parsed.data.len(), 17 * 17 * 17);

        // Should be near-identity (float formatting precision)
        for (i, (orig, back)) in identity.data.iter().zip(parsed.data.iter()).enumerate() {
            for c in 0..3 {
                let diff = (orig[c] - back[c]).abs();
                assert!(
                    diff < 1e-5,
                    "entry {i} channel {c}: orig={}, back={}, diff={diff}",
                    orig[c],
                    back[c]
                );
            }
        }
    }

    #[test]
    fn cube_format_has_correct_header() {
        let lut = ColorLut3D::identity(9);
        let config = CubeExportConfig {
            grid_size: 9,
            title: Some("Test LUT".to_string()),
        };
        let text = serialize_cube(&lut, &config);

        assert!(text.starts_with("TITLE \"Test LUT\"\n"));
        assert!(text.contains("LUT_3D_SIZE 9\n"));

        // Count data lines (non-empty, non-header)
        let data_lines: Vec<&str> = text
            .lines()
            .filter(|l| {
                let l = l.trim();
                !l.is_empty() && !l.starts_with("TITLE") && !l.starts_with("LUT_3D_SIZE")
            })
            .collect();
        assert_eq!(data_lines.len(), 9 * 9 * 9);
    }

    #[test]
    fn composed_ops_produce_valid_cube() {
        use crate::domain::color_lut::compose_cluts;

        // Hue rotate by building a simple CLUT
        let hue_lut = ColorLut3D::from_fn(17, |r, g, b| {
            // Simple channel swap: R→G, G→B, B→R
            (b, r, g)
        });
        let saturate_lut = ColorLut3D::from_fn(17, |r, g, b| {
            // Desaturate 50%: blend toward gray
            let gray = r * 0.299 + g * 0.587 + b * 0.114;
            let f = 0.5;
            (r * f + gray * (1.0 - f), g * f + gray * (1.0 - f), b * f + gray * (1.0 - f))
        });

        let composed = compose_cluts(&hue_lut, &saturate_lut);
        let cube_text = serialize_cube(&composed, &CubeExportConfig::default());
        let parsed = parse_cube_lut(&cube_text).unwrap();

        assert_eq!(parsed.grid_size, 17);

        // Verify the composed result matches: apply hue_lut then saturate_lut
        for (i, (orig, back)) in composed.data.iter().zip(parsed.data.iter()).enumerate() {
            for c in 0..3 {
                let diff = (orig[c] - back[c]).abs();
                assert!(
                    diff < 1e-5,
                    "entry {i} channel {c}: composed={}, parsed={}, diff={diff}",
                    orig[c],
                    back[c]
                );
            }
        }
    }

    #[test]
    fn encode_returns_valid_bytes() {
        let lut = ColorLut3D::identity(9);
        let bytes = encode(&lut).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("LUT_3D_SIZE 9"));
    }
}
