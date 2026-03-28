//! CRC32 and Adler32 checksums for PNG and zlib.

/// Compute CRC32 (ISO 3309 / ITU-T V.42) used by PNG chunks.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

/// Update a running CRC32 with additional data.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut c = crc ^ 0xFFFFFFFF;
    for &byte in data {
        let index = ((c ^ byte as u32) & 0xFF) as usize;
        c = CRC32_TABLE[index] ^ (c >> 8);
    }
    c ^ 0xFFFFFFFF
}

/// Compute Adler32 checksum used by zlib streams.
pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Update a running Adler32 with additional data.
pub fn adler32_update(adler: u32, data: &[u8]) -> u32 {
    let mut a = adler & 0xFFFF;
    let mut b = adler >> 16;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// Pre-computed CRC32 table (polynomial 0xEDB88320)
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(&[]), 0x00000000);
    }

    #[test]
    fn crc32_known_value() {
        // CRC32 of "123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn crc32_incremental_matches_one_shot() {
        let data = b"Hello, World!";
        let one_shot = crc32(data);
        let incremental = crc32_update(crc32(&data[..5]), &data[5..]);
        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn adler32_empty() {
        assert_eq!(adler32(&[]), 0x00000001);
    }

    #[test]
    fn adler32_known_value() {
        // Adler32 of "Wikipedia" = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn adler32_incremental_matches_one_shot() {
        let data = b"Hello, World!";
        let one_shot = adler32(data);
        let incremental = adler32_update(adler32(&data[..5]), &data[5..]);
        assert_eq!(one_shot, incremental);
    }
}
