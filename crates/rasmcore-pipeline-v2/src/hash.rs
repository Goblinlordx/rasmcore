//! Content hashing for cache keys.

/// Hash type for content-addressed caching (32 bytes, blake3).
pub type ContentHash = [u8; 32];

/// Zero hash — used as a sentinel for "no hash computed".
pub const ZERO_HASH: ContentHash = [0u8; 32];

/// Compute a content hash from a parent hash, operation name, and parameters.
///
/// This produces a deterministic cache key for a node's output given its
/// upstream hash and its own configuration. Two identical operations on
/// identical inputs produce the same hash.
pub fn content_hash(parent: &ContentHash, op_name: &str, params: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(parent);
    hasher.update(op_name.as_bytes());
    hasher.update(params);
    *hasher.finalize().as_bytes()
}

/// Compute a source hash from raw input data.
///
/// Hashes first 4KB + last 4KB + length for speed on large inputs.
/// Two identical source byte sequences produce the same hash.
pub fn source_hash(data: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"source");
    hasher.update(&(data.len() as u64).to_le_bytes());
    let prefix = &data[..data.len().min(4096)];
    hasher.update(prefix);
    if data.len() > 4096 {
        let suffix_start = data.len().saturating_sub(4096);
        hasher.update(&data[suffix_start..]);
    }
    *hasher.finalize().as_bytes()
}

/// Compute a source hash from host-decoded pixel bytes.
///
/// Includes a format-specific domain separator so that the same raw bytes
/// in different formats produce distinct hashes. Uses the same prefix/suffix
/// strategy as `source_hash` for speed on large pixel buffers.
pub fn source_hash_pixels(format_tag: &str, pixels: &[u8], width: u32, height: u32) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(format_tag.as_bytes());
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(&(pixels.len() as u64).to_le_bytes());
    let prefix = &pixels[..pixels.len().min(4096)];
    hasher.update(prefix);
    if pixels.len() > 4096 {
        let suffix_start = pixels.len().saturating_sub(4096);
        hasher.update(&pixels[suffix_start..]);
    }
    *hasher.finalize().as_bytes()
}
