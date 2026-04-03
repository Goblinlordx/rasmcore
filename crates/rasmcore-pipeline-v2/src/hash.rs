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
