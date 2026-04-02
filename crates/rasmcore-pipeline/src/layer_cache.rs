//! Content-addressed layer cache for cross-pipeline result reuse.
//!
//! The LayerCache lives outside any single pipeline's lifetime. Each node's
//! output is keyed by a content hash that encodes the full computation lineage:
//! hash(upstream_hash || operation_name || param_bytes).
//!
//! Pipelines check the cache before execution and push results after completion.
//! The SpatialCache is NOT modified — LayerCache operates at the graph level.
//!
//! ## Cache Quality (Quantization)
//!
//! By default, cached entries store pixels at full precision (f32). For memory-
//! constrained environments (e.g., WASM with 4GB address space), opt-in
//! quantization reduces storage by 2x (Q16) or 4x (Q8):
//!
//! - `Full`: Store as-is (16 bytes/pixel for Rgba32f)
//! - `Q16`: Quantize to u16 on store, promote to f32 on read (8 bytes/pixel)
//! - `Q8`: Quantize to u8 on store, promote to f32 on read (4 bytes/pixel)
//!
//! Filters always see f32 — quantization is transparent at cache boundaries.

use std::collections::HashMap;

/// Content hash — 32-byte blake3 digest encoding the full computation lineage.
pub type ContentHash = [u8; 32];

/// A zero hash used as the initial upstream for source nodes.
pub const ZERO_HASH: ContentHash = [0u8; 32];

/// Compute a content hash from an upstream hash, operation name, and parameters.
///
/// hash = blake3(upstream_hash || op_name || param_bytes)
pub fn compute_hash(upstream: &ContentHash, op: &str, params: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(upstream);
    hasher.update(op.as_bytes());
    hasher.update(params);
    *hasher.finalize().as_bytes()
}

/// Compute a source hash from input data (first 4KB + length for speed).
pub fn compute_source_hash(data: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"source");
    hasher.update(&(data.len() as u64).to_le_bytes());
    // Hash first 4KB + last 4KB for speed on large inputs
    let prefix = &data[..data.len().min(4096)];
    hasher.update(prefix);
    if data.len() > 4096 {
        let suffix_start = data.len().saturating_sub(4096);
        hasher.update(&data[suffix_start..]);
    }
    *hasher.finalize().as_bytes()
}

/// Cache storage quality — controls memory/precision tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheQuality {
    /// Full precision — store pixels as-is (16 bytes/pixel for Rgba32f).
    Full,
    /// Quantize to u16 on store, promote to f32 on read (8 bytes/pixel, 2x saving).
    Q16,
    /// Quantize to u8 on store, promote to f32 on read (4 bytes/pixel, 4x saving).
    Q8,
}

/// A cached layer entry: pixels + metadata.
struct CachedLayer {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    /// Bytes per pixel of the STORED data (may differ from original if quantized).
    stored_bpp: u32,
    /// Bytes per pixel of the ORIGINAL data before quantization.
    original_bpp: u32,
    referenced: bool,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: u32,
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
}

/// Default maximum cap: 1 GB. Leaves headroom in 4 GB WASM address space.
const DEFAULT_MAX_BUDGET: usize = 1024 * 1024 * 1024;

/// Content-addressed layer cache that persists across pipeline lifetimes.
///
/// Starts at the initial budget and auto-grows (doubling) up to `max_budget`
/// when a store would require evicting referenced entries. This avoids
/// thrashing on large images or long chains while keeping a safe cap.
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
    /// Auto-grows up to 1 GB when needed.
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

    /// Set cache quality. Affects future stores only — existing entries are not re-quantized.
    pub fn set_cache_quality(&mut self, quality: CacheQuality) {
        self.quality = quality;
    }

    /// Get the current cache quality setting.
    pub fn cache_quality(&self) -> CacheQuality {
        self.quality
    }

    /// Check if a hash is cached. Returns pixels (promoted to original precision if
    /// quantized), width, height, and original bpp. Marks as referenced.
    pub fn get(&mut self, hash: &ContentHash) -> Option<(Vec<u8>, u32, u32, u32)> {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.referenced = true;
            self.stats.hits += 1;
            let pixels = if entry.stored_bpp == entry.original_bpp {
                // Full precision — no promotion needed
                entry.pixels.clone()
            } else {
                // Quantized — promote back to original precision
                promote_pixels(&entry.pixels, entry.stored_bpp, entry.original_bpp)
            };
            Some((pixels, entry.width, entry.height, entry.original_bpp))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store a layer's output in the cache. If cache quality is Q16 or Q8,
    /// the pixels are quantized before storage to save memory.
    pub fn store(
        &mut self,
        hash: ContentHash,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        bpp: u32,
    ) {
        // Quantize if quality setting demands it and source is f32 (bpp=16 for Rgba32f)
        let (stored_pixels, stored_bpp) = if bpp == 16 && self.quality != CacheQuality::Full {
            match self.quality {
                CacheQuality::Q16 => (quantize_f32_to_u16(&pixels), 8u32),
                CacheQuality::Q8 => (quantize_f32_to_u8(&pixels), 4u32),
                CacheQuality::Full => unreachable!(),
            }
        } else {
            (pixels, bpp)
        };

        let size = stored_pixels.len();

        // Evict unreferenced entries first
        while self.memory_used + size > self.memory_budget && !self.entries.is_empty() {
            let evict_key = self
                .entries
                .iter()
                .find(|(_, v)| !v.referenced)
                .map(|(k, _)| *k);
            if let Some(key) = evict_key {
                if let Some(removed) = self.entries.remove(&key) {
                    self.memory_used -= removed.pixels.len();
                }
            } else {
                // All entries referenced — try to grow instead of dropping
                if self.memory_budget < self.max_budget {
                    let new_budget = (self.memory_budget * 2).min(self.max_budget);
                    let needed = self.memory_used + size;
                    self.memory_budget = new_budget.max(needed).min(self.max_budget);
                }
                break;
            }
        }

        // If still doesn't fit after eviction, grow if under cap
        if self.memory_used + size > self.memory_budget && self.memory_budget < self.max_budget {
            let needed = self.memory_used + size;
            self.memory_budget = needed.min(self.max_budget);
        }

        // Store if fits (after possible growth)
        if self.memory_used + size <= self.memory_budget || self.entries.is_empty() {
            self.memory_used += size;
            self.entries.insert(
                hash,
                CachedLayer {
                    pixels: stored_pixels,
                    width,
                    height,
                    stored_bpp,
                    original_bpp: bpp,
                    referenced: true,
                },
            );
        }
    }

    /// Mark a hash as referenced (even if it was a cache hit — still "used").
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

    /// Remove all unreferenced entries. Call after pipeline completion.
    pub fn cleanup_unreferenced(&mut self) {
        self.entries.retain(|_, v| {
            if v.referenced {
                true
            } else {
                self.memory_used -= v.pixels.len();
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

// ─── Quantization helpers ────────────────────────────────────────────────

/// Quantize Rgba32f (16 bytes/pixel) to Rgba16 (8 bytes/pixel).
fn quantize_f32_to_u16(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 8);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..4 {
            let v = f32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
            let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    out
}

/// Quantize Rgba32f (16 bytes/pixel) to Rgba8 (4 bytes/pixel).
fn quantize_f32_to_u8(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 4);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..4 {
            let v = f32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
            out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// Promote quantized pixels back to original precision.
fn promote_pixels(pixels: &[u8], stored_bpp: u32, original_bpp: u32) -> Vec<u8> {
    if stored_bpp == original_bpp {
        return pixels.to_vec();
    }
    // Promote to Rgba32f (bpp=16)
    if original_bpp == 16 {
        match stored_bpp {
            8 => promote_u16_to_f32(pixels),
            4 => promote_u8_to_f32(pixels),
            _ => pixels.to_vec(), // unknown — pass through
        }
    } else {
        pixels.to_vec() // non-f32 original — pass through
    }
}

/// Promote Rgba16 (8 bytes/pixel) to Rgba32f (16 bytes/pixel).
fn promote_u16_to_f32(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 8; // 8 bytes per pixel (4 channels * 2 bytes)
    let mut out = Vec::with_capacity(count * 16);
    for chunk in pixels.chunks_exact(8) {
        for i in 0..4 {
            let u = u16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]);
            let v = u as f32 / 65535.0;
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Promote Rgba8 (4 bytes/pixel) to Rgba32f (16 bytes/pixel).
fn promote_u8_to_f32(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 4; // 4 bytes per pixel (4 channels)
    let mut out = Vec::with_capacity(count * 16);
    for chunk in pixels.chunks_exact(4) {
        for &b in chunk {
            let v = b as f32 / 255.0;
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_chain_deterministic() {
        let src = compute_source_hash(b"test image data");
        let h1 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        assert_eq!(h1, h2, "same inputs must produce same hash");
    }

    #[test]
    fn hash_chain_different_params() {
        let src = compute_source_hash(b"test image data");
        let h1 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src, "blur", &3.0f32.to_le_bytes());
        assert_ne!(h1, h2, "different params must produce different hash");
    }

    #[test]
    fn hash_chain_upstream_propagates() {
        let src1 = compute_source_hash(b"image A");
        let src2 = compute_source_hash(b"image B");
        let h1 = compute_hash(&src1, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src2, "blur", &2.0f32.to_le_bytes());
        assert_ne!(h1, h2, "different upstream must produce different hash");
    }

    #[test]
    fn store_and_retrieve() {
        let mut cache = LayerCache::new(1024 * 1024);
        let hash = compute_source_hash(b"test");
        cache.store(hash, vec![1, 2, 3, 4], 2, 2, 1);
        let result = cache.get(&hash);
        assert!(result.is_some());
        let (pixels, w, h, bpp) = result.unwrap();
        assert_eq!(pixels, &[1, 2, 3, 4]);
        assert_eq!((w, h, bpp), (2, 2, 1));
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = LayerCache::new(1024);
        let hash = compute_source_hash(b"not stored");
        assert!(cache.get(&hash).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn cleanup_removes_unreferenced() {
        let mut cache = LayerCache::new(1024 * 1024);
        let h1 = compute_source_hash(b"keep");
        let h2 = compute_source_hash(b"remove");
        cache.store(h1, vec![1; 100], 10, 10, 1);
        cache.store(h2, vec![2; 100], 10, 10, 1);

        // Reset and only reference h1
        cache.reset_references();
        cache.mark_referenced(&h1);
        cache.cleanup_unreferenced();

        assert!(cache.contains(&h1));
        assert!(!cache.contains(&h2));
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn memory_budget_enforced() {
        let mut cache = LayerCache::new(200); // tight budget
        let h1 = compute_source_hash(b"first");
        let h2 = compute_source_hash(b"second");
        cache.store(h1, vec![0; 100], 10, 10, 1);
        cache.store(h2, vec![0; 100], 10, 10, 1);

        // Both fit in 200 bytes
        assert!(cache.contains(&h1));
        assert!(cache.contains(&h2));

        // Third should evict unreferenced
        cache.reset_references();
        cache.mark_referenced(&h2); // only h2 referenced
        let h3 = compute_source_hash(b"third");
        cache.store(h3, vec![0; 100], 10, 10, 1);

        // h1 should be evicted (unreferenced), h2 and h3 should remain
        assert!(!cache.contains(&h1));
        assert!(cache.contains(&h2));
        assert!(cache.contains(&h3));
    }

    #[test]
    fn clear_removes_everything() {
        let mut cache = LayerCache::new(1024);
        cache.store(compute_source_hash(b"a"), vec![1], 1, 1, 1);
        cache.store(compute_source_hash(b"b"), vec![2], 1, 1, 1);
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().size_bytes, 0);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut cache = LayerCache::new(1024);
        let hash = compute_source_hash(b"test");
        cache.store(hash, vec![1, 2, 3], 1, 1, 3);

        cache.get(&hash); // hit
        cache.get(&hash); // hit
        cache.get(&compute_source_hash(b"miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    // ── Cache Quality Tests ─────────────────────────────────────────────

    /// Helper: build a 1-pixel Rgba32f buffer [R, G, B, A] from f32 values.
    fn make_rgba32f(r: f32, g: f32, b: f32, a: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&r.to_le_bytes());
        v.extend_from_slice(&g.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        v.extend_from_slice(&a.to_le_bytes());
        v
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn q16_round_trip_preserves_quality() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q16);

        let hash = compute_source_hash(b"q16 test");
        let pixels = make_rgba32f(0.0, 0.5, 1.0, 1.0);
        cache.store(hash, pixels.clone(), 1, 1, 16);

        // Stored size should be 8 bytes (Q16), not 16 (f32)
        assert_eq!(cache.stats().size_bytes, 8);

        // Get should promote back to f32
        let (result, w, h, bpp) = cache.get(&hash).unwrap();
        assert_eq!((w, h, bpp), (1, 1, 16));
        assert_eq!(result.len(), 16); // promoted to f32

        // Check values — Q16 should be very close (within 1/65535)
        let r = read_f32(&result, 0);
        let g = read_f32(&result, 4);
        let b = read_f32(&result, 8);
        let a = read_f32(&result, 12);
        assert!((r - 0.0).abs() < 0.001, "R: expected ~0.0, got {r}");
        assert!((g - 0.5).abs() < 0.001, "G: expected ~0.5, got {g}");
        assert!((b - 1.0).abs() < 0.001, "B: expected ~1.0, got {b}");
        assert!((a - 1.0).abs() < 0.001, "A: expected ~1.0, got {a}");
    }

    #[test]
    fn q8_round_trip_preserves_quality() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q8);

        let hash = compute_source_hash(b"q8 test");
        let pixels = make_rgba32f(0.0, 0.5, 1.0, 1.0);
        cache.store(hash, pixels, 1, 1, 16);

        // Stored size should be 4 bytes (Q8), not 16 (f32)
        assert_eq!(cache.stats().size_bytes, 4);

        let (result, w, h, bpp) = cache.get(&hash).unwrap();
        assert_eq!((w, h, bpp), (1, 1, 16));
        assert_eq!(result.len(), 16);

        let r = read_f32(&result, 0);
        let g = read_f32(&result, 4);
        let b = read_f32(&result, 8);
        let a = read_f32(&result, 12);
        // Q8 has lower precision — within 1/255
        assert!((r - 0.0).abs() < 0.005, "R: expected ~0.0, got {r}");
        assert!((g - 0.5).abs() < 0.005, "G: expected ~0.5, got {g}");
        assert!((b - 1.0).abs() < 0.005, "B: expected ~1.0, got {b}");
        assert!((a - 1.0).abs() < 0.005, "A: expected ~1.0, got {a}");
    }

    #[test]
    fn full_quality_no_quantization() {
        let mut cache = LayerCache::new(1024 * 1024);
        // Default is Full
        assert_eq!(cache.cache_quality(), CacheQuality::Full);

        let hash = compute_source_hash(b"full test");
        let pixels = make_rgba32f(0.123456, 0.654321, 0.999999, 0.000001);
        cache.store(hash, pixels.clone(), 1, 1, 16);

        // Full precision — stored size = 16 bytes
        assert_eq!(cache.stats().size_bytes, 16);

        let (result, _, _, _) = cache.get(&hash).unwrap();
        assert_eq!(result, pixels, "Full quality should be bit-exact");
    }

    #[test]
    fn q16_memory_savings() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q16);

        // Store 100 pixels at f32 (1600 bytes original)
        let mut pixels = Vec::with_capacity(100 * 16);
        for i in 0..100u32 {
            let v = i as f32 / 99.0;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let hash = compute_source_hash(b"100px");
        cache.store(hash, pixels, 10, 10, 16);

        // Q16: 100 pixels * 8 bytes = 800 bytes (2x saving)
        assert_eq!(cache.stats().size_bytes, 800);
    }

    #[test]
    fn q8_memory_savings() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q8);

        let mut pixels = Vec::with_capacity(100 * 16);
        for i in 0..100u32 {
            let v = i as f32 / 99.0;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let hash = compute_source_hash(b"100px q8");
        cache.store(hash, pixels, 10, 10, 16);

        // Q8: 100 pixels * 4 bytes = 400 bytes (4x saving)
        assert_eq!(cache.stats().size_bytes, 400);
    }

    #[test]
    fn non_f32_data_not_quantized() {
        let mut cache = LayerCache::new(1024 * 1024);
        cache.set_cache_quality(CacheQuality::Q8);

        // Store Rgba8 data (bpp=4) — should NOT be quantized further
        let hash = compute_source_hash(b"rgba8");
        let pixels = vec![128, 64, 32, 255]; // 1 pixel, bpp=4
        cache.store(hash, pixels.clone(), 1, 1, 4);

        assert_eq!(cache.stats().size_bytes, 4); // unchanged
        let (result, _, _, bpp) = cache.get(&hash).unwrap();
        assert_eq!(bpp, 4);
        assert_eq!(result, pixels);
    }
}
