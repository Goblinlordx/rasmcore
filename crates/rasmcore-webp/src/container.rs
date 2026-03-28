//! WebP RIFF container wrapping.
//!
//! Wraps VP8 frame data in a standard WebP file:
//! `RIFF` + file_size(LE32) + `WEBP` + `VP8 ` + chunk_size(LE32) + vp8_data

/// Wrap raw VP8 frame data in a complete WebP RIFF container.
///
/// Returns a valid `.webp` file that can be opened by any WebP decoder.
pub fn wrap_vp8(vp8_data: &[u8]) -> Vec<u8> {
    let chunk_size = vp8_data.len() as u32;
    // RIFF file size = "WEBP" (4) + "VP8 " (4) + chunk_size_field (4) + data
    let file_size = 4 + 4 + 4 + chunk_size;
    // Pad to even size per RIFF spec
    let padding = chunk_size & 1;

    let mut out = Vec::with_capacity(12 + 8 + vp8_data.len() + padding as usize);

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(file_size + padding).to_le_bytes());
    out.extend_from_slice(b"WEBP");

    // VP8 chunk
    out.extend_from_slice(b"VP8 ");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(vp8_data);

    // Pad to even length if needed
    if padding != 0 {
        out.push(0);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_vp8_produces_riff_header() {
        let data = vec![0u8; 10];
        let webp = wrap_vp8(&data);
        assert_eq!(&webp[0..4], b"RIFF");
        assert_eq!(&webp[8..12], b"WEBP");
        assert_eq!(&webp[12..16], b"VP8 ");
    }

    #[test]
    fn wrap_vp8_correct_sizes() {
        let data = vec![0u8; 100];
        let webp = wrap_vp8(&data);
        let file_size = u32::from_le_bytes(webp[4..8].try_into().unwrap());
        let chunk_size = u32::from_le_bytes(webp[16..20].try_into().unwrap());
        assert_eq!(chunk_size, 100);
        assert_eq!(file_size, 4 + 4 + 4 + 100);
    }

    #[test]
    fn wrap_vp8_odd_size_padded() {
        let data = vec![0u8; 11]; // odd
        let webp = wrap_vp8(&data);
        assert_eq!(webp.len() % 2, 0); // padded to even
    }
}
