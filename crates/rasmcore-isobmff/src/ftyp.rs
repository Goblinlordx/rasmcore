//! File type (ftyp) box parser and format detection.

use crate::boxreader::read_box_header;
use crate::error::IsobmffError;
use crate::types::{Brand, Ftyp};

/// Detect whether the given data starts with an ISOBMFF ftyp box
/// containing a recognized HEIF or AVIF brand.
///
/// Returns the major brand if recognized, `None` otherwise.
/// Only needs the first ~32 bytes of the file.
pub fn detect(header: &[u8]) -> Option<Brand> {
    let ftyp = parse_ftyp(header).ok()?;
    let brand = ftyp.major_brand;
    if brand.is_heif() || brand.is_avif() {
        Some(brand)
    } else {
        // Check compatible brands — some files use `mif1` as major
        // but list `heic` or `avif` as compatible
        ftyp.compatible_brands
            .iter()
            .find(|b| b.is_heif() || b.is_avif())
            .copied()
    }
}

/// Parse the ftyp box from the beginning of the data.
///
/// The ftyp box must be the first box in the file.
pub fn parse_ftyp(data: &[u8]) -> Result<Ftyp, IsobmffError> {
    let header = read_box_header(data, 0)?;

    if &header.box_type != b"ftyp" {
        return Err(IsobmffError::NotIsobmff);
    }

    let content_start = header.content_offset;
    let content_len = match header.content_size {
        Some(s) => s as usize,
        None => data.len().saturating_sub(content_start),
    };

    if content_len < 8 {
        return Err(IsobmffError::Truncated {
            expected: 8,
            available: content_len,
        });
    }

    let content = &data[content_start..content_start + content_len];

    let mut major = [0u8; 4];
    major.copy_from_slice(&content[0..4]);
    let major_brand = Brand::from_fourcc(major);

    let minor_version = u32::from_be_bytes([content[4], content[5], content[6], content[7]]);

    let mut compatible_brands = Vec::new();
    let mut pos = 8;
    while pos + 4 <= content_len {
        let mut cc = [0u8; 4];
        cc.copy_from_slice(&content[pos..pos + 4]);
        compatible_brands.push(Brand::from_fourcc(cc));
        pos += 4;
    }

    Ok(Ftyp {
        major_brand,
        minor_version,
        compatible_brands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ftyp(major: &[u8; 4], minor: u32, compat: &[&[u8; 4]]) -> Vec<u8> {
        let content_len = 8 + compat.len() * 4;
        let box_size = (8 + content_len) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&box_size.to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(major);
        buf.extend_from_slice(&minor.to_be_bytes());
        for c in compat {
            buf.extend_from_slice(*c);
        }
        buf
    }

    #[test]
    fn parse_heic_ftyp() {
        let data = make_ftyp(b"heic", 0, &[b"heic", b"mif1"]);
        let ftyp = parse_ftyp(&data).unwrap();
        assert_eq!(ftyp.major_brand, Brand::Heic);
        assert_eq!(ftyp.minor_version, 0);
        assert_eq!(ftyp.compatible_brands.len(), 2);
        assert_eq!(ftyp.compatible_brands[0], Brand::Heic);
        assert_eq!(ftyp.compatible_brands[1], Brand::Mif1);
    }

    #[test]
    fn parse_avif_ftyp() {
        let data = make_ftyp(b"avif", 0, &[b"avif", b"mif1"]);
        let ftyp = parse_ftyp(&data).unwrap();
        assert_eq!(ftyp.major_brand, Brand::Avif);
        assert!(ftyp.major_brand.is_avif());
    }

    #[test]
    fn detect_heic() {
        let data = make_ftyp(b"heic", 0, &[b"heic"]);
        let brand = detect(&data);
        assert_eq!(brand, Some(Brand::Heic));
    }

    #[test]
    fn detect_avif() {
        let data = make_ftyp(b"avif", 0, &[b"avif", b"mif1"]);
        let brand = detect(&data);
        assert_eq!(brand, Some(Brand::Avif));
    }

    #[test]
    fn detect_mif1_with_heic_compat() {
        // Major brand is mif1 but heic is in compatible brands
        let data = make_ftyp(b"mif1", 0, &[b"mif1", b"heic"]);
        let brand = detect(&data);
        // Should detect via compatible brand check
        assert!(brand.is_some());
    }

    #[test]
    fn detect_unknown_brand() {
        let data = make_ftyp(b"isom", 0, &[b"isom", b"mp41"]);
        let brand = detect(&data);
        assert_eq!(brand, None);
    }

    #[test]
    fn detect_not_ftyp() {
        let data = vec![0u8; 100];
        let brand = detect(&data);
        assert_eq!(brand, None);
    }

    #[test]
    fn ftyp_multiple_compatible_brands() {
        let data = make_ftyp(b"heic", 1, &[b"heic", b"heix", b"mif1", b"MiPr"]);
        let ftyp = parse_ftyp(&data).unwrap();
        assert_eq!(ftyp.compatible_brands.len(), 4);
        assert_eq!(ftyp.minor_version, 1);
    }

    #[test]
    fn not_isobmff() {
        // PNG magic bytes
        let data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        let err = parse_ftyp(data).unwrap_err();
        assert!(matches!(err, IsobmffError::NotIsobmff));
    }
}
