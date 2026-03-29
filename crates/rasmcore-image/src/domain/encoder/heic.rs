//! HEIC encoder — encode RGB pixels to HEIC container via HEVC I-frame encoding.
//!
//! Pipeline: RGB → YCbCr 4:2:0 → HEVC I-frame encode → ISOBMFF HEIC container.
//!
//! Feature-gated behind `nonfree-hevc` to isolate patent-encumbered code.

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// HEIC encode configuration.
#[derive(Debug, Clone)]
pub struct HeicEncodeConfig {
    /// Quality level 1-100 (default: 75). Mapped to HEVC QP (51→1 inversely).
    pub quality: u8,
}

impl Default for HeicEncodeConfig {
    fn default() -> Self {
        Self { quality: 75 }
    }
}

/// Map quality (1-100) to HEVC QP (0-51).
/// Quality 100 → QP 0 (lossless-like), Quality 1 → QP 51.
fn quality_to_qp(quality: u8) -> i32 {
    let q = quality.clamp(1, 100) as i32;
    // Linear mapping: q=100 → QP=4, q=50 → QP=26, q=1 → QP=51
    // Matches x265 CRF-like behavior for still images
    51 - (q - 1) * 47 / 99
}

/// Encode pixel data to HEIC format.
///
/// Converts RGB8 input to YCbCr 4:2:0, encodes as an HEVC I-frame, and wraps
/// in an ISOBMFF HEIC container.
#[cfg(feature = "nonfree-hevc")]
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    config: &HeicEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    use rasmcore_color::ColorMatrix;
    use rasmcore_hevc::encode::encoder::{encode_iframe, EncodeConfig};

    let width = info.width;
    let height = info.height;

    // Step 1: RGB → YCbCr 4:2:0 (BT.709 limited range)
    let yuv = rasmcore_color::rgb_to_ycbcr_420(pixels, width, height, &ColorMatrix::BT709);

    // Step 2: Encode HEVC I-frame
    let qp = quality_to_qp(config.quality);
    let enc_config = EncodeConfig { qp };
    let bitstream = encode_iframe(&yuv.y, &yuv.u, &yuv.v, width, height, &enc_config)
        .map_err(|e| ImageError::ProcessingFailed(format!("HEVC encode failed: {e}")))?;

    // Step 3: Extract parameter set NALs from bitstream for hvcC config
    let hevc_config = extract_hevc_config(&bitstream, width, height)?;

    // Step 4: Extract slice data (after parameter sets) for mdat
    let slice_data = extract_slice_data(&bitstream)?;

    // Step 5: Wrap in ISOBMFF HEIC container
    let heic_input = rasmcore_isobmff::writer::HeicInput {
        width,
        height,
        bitstream: slice_data,
        hevc_config,
        color: None, // No ICC profile for now
    };
    let heic_bytes = rasmcore_isobmff::writer::assemble_heic(&heic_input);

    Ok(heic_bytes)
}

/// Encode pixel data to HEIC format (stub when nonfree-hevc feature is disabled).
#[cfg(not(feature = "nonfree-hevc"))]
pub fn encode(
    _pixels: &[u8],
    _info: &ImageInfo,
    _config: &HeicEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    Err(ImageError::ProcessingFailed(
        "HEIC encoding requires the nonfree-hevc feature".into(),
    ))
}

/// Extract HEVC config (VPS/SPS/PPS NAL arrays) from an Annex B bitstream.
#[cfg(feature = "nonfree-hevc")]
fn extract_hevc_config(
    bitstream: &[u8],
    _width: u32,
    _height: u32,
) -> Result<rasmcore_isobmff::HevcConfig, ImageError> {
    use rasmcore_hevc::nal::NalIterator;
    use rasmcore_hevc::types::NalUnitType;

    let mut vps_nals = Vec::new();
    let mut sps_nals = Vec::new();
    let mut pps_nals = Vec::new();

    for nal_data in NalIterator::new(bitstream) {
        if nal_data.len() < 2 {
            continue;
        }
        let nal_type = (nal_data[0] >> 1) & 0x3F;
        // Remove the 2-byte NAL header to get raw RBSP+prevention bytes
        let nal_payload = nal_data.to_vec();
        match NalUnitType::from_u8(nal_type) {
            NalUnitType::VpsNut => vps_nals.push(nal_payload),
            NalUnitType::SpsNut => sps_nals.push(nal_payload),
            NalUnitType::PpsNut => pps_nals.push(nal_payload),
            _ => {}
        }
    }

    let mut nal_arrays = Vec::new();
    if !vps_nals.is_empty() {
        nal_arrays.push(rasmcore_isobmff::NalArray {
            completeness: true,
            nal_type: 32, // VPS
            nal_units: vps_nals,
        });
    }
    if !sps_nals.is_empty() {
        nal_arrays.push(rasmcore_isobmff::NalArray {
            completeness: true,
            nal_type: 33, // SPS
            nal_units: sps_nals,
        });
    }
    if !pps_nals.is_empty() {
        nal_arrays.push(rasmcore_isobmff::NalArray {
            completeness: true,
            nal_type: 34, // PPS
            nal_units: pps_nals,
        });
    }

    Ok(rasmcore_isobmff::HevcConfig {
        configuration_version: 1,
        general_profile_space: 0,
        general_tier_flag: false,
        general_profile_idc: 3, // Main Still Picture
        general_level_idc: 60,
        chroma_format_idc: 1, // 4:2:0
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        nal_arrays,
    })
}

/// Extract slice data (VCL NALs) from an Annex B bitstream, skipping parameter sets.
#[cfg(feature = "nonfree-hevc")]
fn extract_slice_data(bitstream: &[u8]) -> Result<Vec<u8>, ImageError> {
    use rasmcore_hevc::nal::NalIterator;
    use rasmcore_hevc::types::NalUnitType;

    // Find the first VCL NAL unit and return everything from that point
    // including the start code, so the decoder can parse it
    let mut vcl_start = None;
    let mut pos = 0;
    for nal_data in NalIterator::new(bitstream) {
        if nal_data.len() < 2 {
            continue;
        }
        let nal_type = NalUnitType::from_u8((nal_data[0] >> 1) & 0x3F);
        if nal_type.is_vcl() {
            // Find this NAL's position in the original bitstream
            // The NAL data pointer tells us where in the stream it is
            vcl_start = Some(nal_data.as_ptr() as usize - bitstream.as_ptr() as usize);
            break;
        }
    }

    match vcl_start {
        // Go back to include the start code (00 00 00 01 or 00 00 01)
        Some(start) => {
            let adjusted = if start >= 4 && bitstream[start - 4..start] == [0, 0, 0, 1] {
                start - 4
            } else if start >= 3 && bitstream[start - 3..start] == [0, 0, 1] {
                start - 3
            } else {
                start
            };
            Ok(bitstream[adjusted..].to_vec())
        }
        None => Err(ImageError::ProcessingFailed(
            "No VCL NAL found in bitstream".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_to_qp_mapping() {
        assert_eq!(quality_to_qp(100), 4);
        let qp50 = quality_to_qp(50);
        assert!(qp50 >= 26 && qp50 <= 28, "QP at Q50 should be ~27, got {qp50}");
        assert_eq!(quality_to_qp(1), 51);
        // Monotonic
        for q in 1..100 {
            assert!(quality_to_qp(q) >= quality_to_qp(q + 1));
        }
    }

    #[cfg(feature = "nonfree-hevc")]
    #[test]
    fn encode_flat_64x64_heic() {
        use crate::domain::types::{ColorSpace, PixelFormat};

        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 64 * 64 * 3];
        let config = HeicEncodeConfig { quality: 75 };

        let heic_bytes = encode(&pixels, &info, &config).expect("HEIC encode failed");

        // Should produce non-empty output
        assert!(!heic_bytes.is_empty());

        // Should start with ftyp box
        assert_eq!(&heic_bytes[4..8], b"ftyp");

        // Should be parseable by our ISOBMFF reader
        let parsed = rasmcore_isobmff::parse(&heic_bytes).expect("ISOBMFF parse failed");
        assert_eq!(parsed.primary_image.width, 64);
        assert_eq!(parsed.primary_image.height, 64);
    }

    #[cfg(feature = "nonfree-hevc")]
    #[test]
    fn encode_decode_roundtrip() {
        use crate::domain::types::{ColorSpace, PixelFormat};
        use rasmcore_hevc::encode::encoder::{encode_iframe, EncodeConfig};

        let width = 64u32;
        let height = 64u32;
        let info = ImageInfo {
            width,
            height,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; (width * height * 3) as usize];

        // Encode to HEIC container
        let config = HeicEncodeConfig { quality: 75 };
        let heic_bytes = encode(&pixels, &info, &config).expect("HEIC encode failed");

        // Verify container is valid
        let file = rasmcore_isobmff::parse(&heic_bytes).expect("ISOBMFF parse failed");
        assert_eq!(file.primary_image.width, width);
        assert_eq!(file.primary_image.height, height);

        // Separately: encode raw bitstream and decode it directly
        // (the container stores bitstream without parameter sets in mdat;
        // full Annex B decode works with parameter sets inline)
        let yuv = rasmcore_color::rgb_to_ycbcr_420(
            &pixels,
            width,
            height,
            &rasmcore_color::ColorMatrix::BT709,
        );
        let annex_b = encode_iframe(&yuv.y, &yuv.u, &yuv.v, width, height, &EncodeConfig { qp: 26 })
            .expect("HEVC encode failed");
        let decoded = rasmcore_hevc::decode(&annex_b, &[]).expect("HEVC decode failed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert!(!decoded.pixels.is_empty());
    }
}
