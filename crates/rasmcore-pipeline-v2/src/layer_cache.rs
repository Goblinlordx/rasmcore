//! Content-addressed layer cache for cross-pipeline result reuse.
//!
//! The LayerCache lives outside any single pipeline's lifetime. Each node's
//! output is keyed by a content hash that encodes the full computation lineage:
//! hash(upstream_hash || operation_name || param_bytes).
//!
//! Pipelines check the cache before execution and push results after completion.
//! The SpatialCache is NOT modified — LayerCache operates at the graph level,
//! above tiles.
//!
//! ## V2 Design
//!
//! All pixel data is `Vec<f32>` — 4 channels (RGBA) per pixel. The cache
//! stores f32 data directly, with optional quantization to u16 or u8 for
//! memory savings. Consumers always receive f32 — quantization is transparent.
//!
//! ## Cache Quality (Quantization)
//!
//! - `Full`: Store as f32 (16 bytes/pixel)
//! - `Q16`: Quantize to u16 on store, promote to f32 on read (8 bytes/pixel)
//! - `Q8`: Quantize to u8 on store, promote to f32 on read (4 bytes/pixel)

use std::collections::HashMap;
use crate::hash::ContentHash;

/// Cache storage quality — controls memory/precision tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheQuality {
    /// Full precision — store pixels as f32 (16 bytes/pixel).
    Full,
    /// Quantize to u16 on store, promote to f32 on read (8 bytes/pixel, 2x saving).
    Q16,
    /// Quantize to u8 on store, promote to f32 on read (4 bytes/pixel, 4x saving).
    Q8,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: u32,
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
}

/// A cached layer entry.
struct CachedLayer {
    /// Raw stored bytes (f32, u16, or u8 depending on quality).
    data: Vec<u8>,
    width: u32,
    height: u32,
    /// How this entry was stored.
    stored_quality: CacheQuality,
    /// Whether this entry was accessed during the current pipeline run.
    referenced: bool,
}

/// Default maximum cap: 1 GB.
const DEFAULT_MAX_BUDGET: usize = 1024 * 1024 * 1024;

/// Content-addressed layer cache that persists across pipeline lifetimes.
///
/// All data in/out is `Vec<f32>` (RGBA, 4 channels). Quantization is
/// transparent — consumers always see f32.
///
/// Starts at the initial budget and auto-grows (doubling) up to `max_budget`
/// when a store would require evicting referenced entries.
pub struct LayerCache {
    entries: HashMap<ContentHash, CachedLayer>,
    memory_budget: usize,
    max_budget: usize,
    memory_used: usize,
    stats: CacheStats,
    quality: CacheQuality,
}

impl LayerCache {
    /// Create a new layer cache with the given initial memory budget in bytes.
    pub fn new(memory_budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            memory_budget,
            max_budget: DEFAULT_MAX_BUDGET,
            memory_used: 0,
            stats: CacheStats::default(),
            quality: CacheQuality::Full,
        }
    }

    /// Create with explicit initial and maximum budgets.
    pub fn with_max_budget(initial_budget: usize, max_budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            memory_budget: initial_budget,
            max_budget,
            memory_used: 0,
            stats: CacheStats::default(),
            quality: CacheQuality::Full,
        }
    }

    /// Set cache quality. Affects future stores only.
    pub fn set_cache_quality(&mut self, quality: CacheQuality) {
        self.quality = quality;
    }

    /// Get the current cache quality setting.
    pub fn cache_quality(&self) -> CacheQuality {
        self.quality
    }

    /// Retrieve cached f32 pixel data by content hash.
    ///
    /// Returns (pixels, width, height) with pixels always as f32 RGBA.
    /// Marks the entry as referenced.
    pub fn get(&mut self, hash: &ContentHash) -> Option<(Vec<f32>, u32, u32)> {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.referenced = true;
            self.stats.hits += 1;
            let pixels = match entry.stored_quality {
                CacheQuality::Full => bytes_to_f32(&entry.data),
                CacheQuality::Q16 => promote_u16_to_f32(&entry.data),
                CacheQuality::Q8 => promote_u8_to_f32(&entry.data),
            };
            Some((pixels, entry.width, entry.height))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store f32 pixel data in the cache.
    ///
    /// Quantizes according to the current quality setting.
    pub fn store(
        &mut self,
        hash: ContentHash,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) {
        let (stored_data, stored_quality) = match self.quality {
            CacheQuality::Full => (f32_to_bytes(pixels), CacheQuality::Full),
            CacheQuality::Q16 => (quantize_f32_to_u16(pixels), CacheQuality::Q16),
            CacheQuality::Q8 => (quantize_f32_to_u8(pixels), CacheQuality::Q8),
        };

        let size = stored_data.len();

        // Evict unreferenced entries first
        while self.memory_used + size > self.memory_budget && !self.entries.is_empty() {
            let evict_key = self
                .entries
                .iter()
                .find(|(_, v)| !v.referenced)
                .map(|(k, _)| *k);
            if let Some(key) = evict_key {
                if let Some(removed) = self.entries.remove(&key) {
                    self.memory_used -= removed.data.len();
                }
            } else {
                // All entries referenced — try to grow
                if self.memory_budget < self.max_budget {
                    let new_budget = (self.memory_budget * 2).min(self.max_budget);
                    let needed = self.memory_used + size;
                    self.memory_budget = new_budget.max(needed).min(self.max_budget);
                }
                break;
            }
        }

        // Grow if still doesn't fit
        if self.memory_used + size > self.memory_budget && self.memory_budget < self.max_budget {
            let needed = self.memory_used + size;
            self.memory_budget = needed.min(self.max_budget);
        }

        // Store if fits
        if self.memory_used + size <= self.memory_budget || self.entries.is_empty() {
            self.memory_used += size;
            self.entries.insert(
                hash,
                CachedLayer {
                    data: stored_data,
                    width,
                    height,
                    stored_quality,
                    referenced: true,
                },
            );
        }
    }

    /// Mark a hash as referenced.
    pub fn mark_referenced(&mut self, hash: &ContentHash) {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.referenced = true;
        }
    }

    /// Reset all referenced flags (call at start of new pipeline run).
    pub fn reset_references(&mut self) {
        for entry in self.entries.values_mut() {
            entry.referenced = false;
        }
    }

    /// Remove all unreferenced entries.
    pub fn cleanup_unreferenced(&mut self) {
        self.entries.retain(|_, v| {
            if v.referenced {
                true
            } else {
                self.memory_used -= v.data.len();
                false
            }
        });
        self.stats.entries = self.entries.len() as u32;
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.memory_used = 0;
        self.stats = CacheStats::default();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len() as u32,
            size_bytes: self.memory_used as u64,
            ..self.stats
        }
    }

    /// Check if a hash exists without marking it referenced.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.entries.contains_key(hash)
    }
}

// ─── Conversion helpers ─────────────────────────────────────────────────────

/// Convert f32 slice to raw bytes (zero-copy layout).
fn f32_to_bytes(pixels: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for &v in pixels {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Convert raw bytes back to f32 slice.
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Quantize f32 RGBA to u16 RGBA (8 bytes/pixel → 2 bytes/channel).
fn quantize_f32_to_u16(pixels: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() * 2);
    for &v in pixels {
        let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

/// Quantize f32 RGBA to u8 RGBA (4 bytes/pixel → 1 byte/channel).
fn quantize_f32_to_u8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Promote u16 RGBA back to f32.
fn promote_u16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0)
        .collect()
}

/// Promote u8 RGBA back to f32.
fn promote_u8_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| b as f32 / 255.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{content_hash, source_hash};

    #[test]
    fn store_and_retrieve() {
        let mut cache = LayerCache::new(1024 * 1024);
        let hash = source_hash(b"test");
        let pixels = vec![0.5, 0.3, 0.1, 1.0]; // 1 pixel
        cache.store(hash, &pixels, 1, 1);
        let (result, w, h) = cache.get(&hash).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(result, pixels);
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = LayerCache::new(1024);
        let hash = source_hash(b"not stored");
        assert!(cache.get(&hash).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn hash_chain_deterministic() {
        let src = source_hash(b"test image data");
        let h1 = content_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = content_hash(&src, "blur", &2.0f32.to_le_bytes());
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_chain_different_params() {
        let src = source_hash(b"test image data");
        let h1 = content_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = content_hash(&src, "blur", &3.0f32.to_le_bytes());
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_chain_upstream_propagates() {
        let src1 = source_hash(b"image A");
        let src2 = source_hash(b"image B");
        let h1 = content_hash(&src1, "blur", &2.0f32.to_le_bytes());
        let h2 = content_hash(&src2, "blur", &2.0f32.to_le_bytes());
        assert_ne!(h1, h2);
    }

    #[test]
    fn cleanup_removes_unreferenced() {
        let mut cache = LayerCache::new(1024 * 1024);
        let h1 = source_hash(b"keep");
        let h2 = source_hash(b"remove");
        cache.store(h1, &[0.5; 4], 1, 1);
        cache.store(h2, &[0.3; 4], 1, 1);

        cache.reset_references();
        cache.mark_referenced(&h1);
        cache.cleanup_unreferenced();

        assert!(cache.contains(&h1));
        assert!(!cache.contains(&h2));
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn memory_budget_enforced() {
        // Each 1-pixel f32 entry = 16 bytes stored
        let mut cache = LayerCache::new(40); // fits ~2 entries
        let h1 = source_hash(b"first");
        let h2 = source_hash(b"second");
        cache.store(h1, &[0.5; 4], 1, 1);
        cache.store(h2, &[0.3; 4], 1, 1);
        assert!(cache.contains(&h1));
        assert!(cache.contains(&h2));

        // Third should evict unreferenced
        cache.reset_references();
        cache.mark_referenced(&h2);
        let h3 = source_hash(b"third");
        cache.store(h3, &[0.1; 4], 1, 1);

        assert!(!cache.contains(&h1));
        assert!(cache.contains(&h2));
        assert!(cache.contains(&h3));
    }

    #[test]
    fn clear_removes_everything() {
        let mut cache = LayerCache::new(1024);
        cache.store(source_hash(b"a"), &[0.5; 4], 1, 1);
        cache.store(source_hash(b"b"), &[0.3; 4], 1, 1);
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().size_bytes, 0);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut cache = LayerCache::new(1024);
        let hash = source_hash(b"test");
        cache.store(hash, &[0.5, 0.3, 0.1, 1.0], 1, 1);

        cache.get(&hash); // hit
        cache.get(&hash); // hit
        cache.get(&source_hash(b"miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn q16_round_trip() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q16);

        let hash = source_hash(b"q16 test");
        let pixels = vec![0.0, 0.5, 1.0, 1.0];
        cache.store(hash, &pixels, 1, 1);

        // Q16: 4 channels * 2 bytes = 8 bytes stored
        assert_eq!(cache.stats().size_bytes, 8);

        let (result, w, h) = cache.get(&hash).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.0).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 1.0).abs() < 0.001);
        assert!((result[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn q8_round_trip() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q8);

        let hash = source_hash(b"q8 test");
        let pixels = vec![0.0, 0.5, 1.0, 1.0];
        cache.store(hash, &pixels, 1, 1);

        // Q8: 4 channels * 1 byte = 4 bytes stored
        assert_eq!(cache.stats().size_bytes, 4);

        let (result, w, h) = cache.get(&hash).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.0).abs() < 0.005);
        assert!((result[1] - 0.5).abs() < 0.005);
        assert!((result[2] - 1.0).abs() < 0.005);
        assert!((result[3] - 1.0).abs() < 0.005);
    }

    #[test]
    fn full_quality_bit_exact() {
        let mut cache = LayerCache::new(1024 * 1024);
        assert_eq!(cache.cache_quality(), CacheQuality::Full);

        let hash = source_hash(b"full test");
        let pixels = vec![0.123456, 0.654321, 0.999999, 0.000001];
        cache.store(hash, &pixels, 1, 1);

        assert_eq!(cache.stats().size_bytes, 16);
        let (result, _, _) = cache.get(&hash).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn content_hash_chain_for_cache_reuse() {
        let mut cache = LayerCache::new(1024 * 1024);

        // Simulate: source → brightness(+0.5) → blur(2.0)
        let src_hash = source_hash(b"image data");
        let bright_hash = content_hash(&src_hash, "brightness", &0.5f32.to_le_bytes());
        let blur_hash = content_hash(&bright_hash, "blur", &2.0f32.to_le_bytes());

        // Store results
        cache.store(src_hash, &[0.5; 4], 1, 1);
        cache.store(bright_hash, &[1.0; 4], 1, 1);
        cache.store(blur_hash, &[0.8; 4], 1, 1);

        // Same chain on second run — all hits
        assert!(cache.get(&src_hash).is_some());
        assert!(cache.get(&bright_hash).is_some());
        assert!(cache.get(&blur_hash).is_some());
        assert_eq!(cache.stats().hits, 3);

        // Change blur params — blur miss, others still hit
        let blur_hash2 = content_hash(&bright_hash, "blur", &3.0f32.to_le_bytes());
        assert!(cache.get(&blur_hash2).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Change brightness — everything downstream misses
        let bright_hash2 = content_hash(&src_hash, "brightness", &0.8f32.to_le_bytes());
        let blur_hash3 = content_hash(&bright_hash2, "blur", &2.0f32.to_le_bytes());
        assert!(cache.get(&bright_hash2).is_none());
        assert!(cache.get(&blur_hash3).is_none());
    }
}
