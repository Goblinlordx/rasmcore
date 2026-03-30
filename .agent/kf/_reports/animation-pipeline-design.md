# Animation Pipeline Design — GIF / WebP / APNG

**Status:** Research complete
**Date:** 2026-03-30
**Track:** animation-pipeline-research_20260330041802Z

---

## 1. Format Survey

### 1.1 GIF Animation Model

- **Frames:** Sequence of sub-images composited onto a logical canvas
- **Canvas:** Defined in Logical Screen Descriptor (width, height, background color)
- **Per-frame metadata:**
  - `delay`: centiseconds (1/100s) between frames
  - `left`, `top`: frame offset on canvas (sub-region positioning)
  - `width`, `height`: frame dimensions (can be smaller than canvas)
  - `disposal_method`: what to do with the canvas before rendering next frame
    - `None` (0): leave as-is
    - `DoNotDispose` (1): leave as-is (same as None)
    - `RestoreBackground` (2): clear frame region to background color
    - `RestorePrevious` (3): restore frame region to previous state
  - `transparent`: optional transparent color index
- **Loop count:** Netscape extension (0 = infinite, N = play N times)
- **Color:** Palette-based (256 colors per frame, local or global table)
- **Max colors:** 256 per frame (local color table) — no true color

### 1.2 WebP Animation Model

- **Container:** RIFF with ANIM chunk (global) + ANMF chunks (per-frame)
- **ANIM chunk:** background color (RGBA), loop count
- **ANMF chunk (per-frame):**
  - `offset_x`, `offset_y`: frame position (even numbers, mul of 2)
  - `width`, `height`: frame dimensions
  - `duration`: milliseconds
  - `blend`: `alpha_blend` or `no_blend` (composite over previous or replace)
  - `dispose`: `none` or `background` (no "restore previous" like GIF)
  - Payload: VP8/VP8L bitstream (lossy or lossless, per-frame choice)
- **Loop count:** 0 = infinite
- **Color:** True color RGBA (8-bit per channel)
- **Compression:** Each frame independently compressed (lossy or lossless)

### 1.3 APNG Model

- **Extension of PNG:** Uses additional chunk types within standard PNG container
- **Chunks:**
  - `acTL` (Animation Control): num_frames, num_plays (loop count)
  - `fcTL` (Frame Control): sequence number, width, height, x_offset, y_offset,
    delay_num, delay_den (rational delay), dispose_op, blend_op
  - `fdAT` (Frame Data): sequence number + PNG-compressed pixel data
- **Disposal operations:**
  - `APNG_DISPOSE_OP_NONE` (0): leave as-is
  - `APNG_DISPOSE_OP_BACKGROUND` (1): clear to transparent black
  - `APNG_DISPOSE_OP_PREVIOUS` (2): restore to previous
- **Blend operations:**
  - `APNG_BLEND_OP_SOURCE` (0): replace (overwrite)
  - `APNG_BLEND_OP_OVER` (1): alpha composite
- **Color:** True color, same depth as base PNG (8-bit or 16-bit, with alpha)
- **First frame:** Can be the default PNG IDAT (backward compatible) or separate

### 1.4 Crate API Survey

| Crate | Version | Multi-frame decode | Multi-frame encode | Animation metadata |
|-------|---------|-------------------|-------------------|-------------------|
| `gif` | 0.14 | `read_next_frame()` iterator | `write_frame()` per frame | delay, disposal, offset, transparent |
| `image-webp` | 0.2 | `next_frame()` + duration_ms | Not supported | frame count, loop count, duration |
| `png` | 0.17+ | `next_frame()` for APNG | Not for APNG | num_frames, delay via `fcTL` info |

**gif 0.14:** Full API. `Decoder` has `read_next_frame() -> Option<&Frame>` with
all metadata. `Encoder` has `write_frame(&Frame)`. Best supported.

**image-webp 0.2:** Decode API exists: `next_frame(&mut buf) -> Result<duration_ms>`,
`reset_animation()`, total duration. No encode API for animation.

**png 0.17+:** APNG decode via repeated `next_frame()` calls. The `Reader::info()`
provides animation control info. No APNG encode in the standard png crate;
the separate `apng` crate (0.3) provides encoding.

---

## 2. Architecture Design

### 2.1 Proposed Domain Types

```rust
/// Metadata for a single animation frame.
pub struct FrameMetadata {
    /// Delay before next frame (milliseconds).
    pub delay_ms: u32,
    /// Frame offset on the logical canvas.
    pub offset_x: u32,
    pub offset_y: u32,
    /// How to handle the canvas before rendering the next frame.
    pub dispose: DisposeOp,
    /// How to composite this frame onto the canvas.
    pub blend: BlendOp,
}

pub enum DisposeOp {
    /// Leave canvas as-is after rendering this frame.
    None,
    /// Clear the frame region to background/transparent.
    Background,
    /// Restore the frame region to the state before this frame.
    Previous,
}

pub enum BlendOp {
    /// Replace the frame region (overwrite).
    Source,
    /// Alpha-composite over the existing canvas.
    Over,
}

/// A single decoded animation frame.
pub struct AnimationFrame {
    /// Pixel data for this frame (may be sub-region of canvas).
    pub pixels: Vec<u8>,
    /// Frame dimensions and format.
    pub info: ImageInfo,
    /// Animation-specific metadata.
    pub meta: FrameMetadata,
}

/// A decoded animated image.
pub struct AnimatedImage {
    /// Logical canvas dimensions.
    pub canvas_width: u32,
    pub canvas_height: u32,
    /// Pixel format for all frames.
    pub format: PixelFormat,
    /// Color space.
    pub color_space: ColorSpace,
    /// Loop count (0 = infinite).
    pub loop_count: u32,
    /// All frames in order.
    pub frames: Vec<AnimationFrame>,
    /// ICC profile (shared across all frames).
    pub icc_profile: Option<Vec<u8>>,
}
```

**Design decisions:**
- `AnimatedImage` is separate from `DecodedImage` — no enum wrapper. This keeps
  the single-frame path zero-cost and avoids `match` everywhere.
- Frames store their own sub-region pixels and offsets. The caller decides whether
  to composite to full canvas or process sub-regions directly.
- `DisposeOp` and `BlendOp` are unified across formats (GIF/WebP/APNG all map cleanly).
- Delay is in milliseconds (GIF centiseconds × 10, WebP already ms, APNG rational converted).

### 2.2 Proposed Decoder API

```rust
/// Decode all frames of an animated image.
///
/// Returns `AnimatedImage` if the input contains multiple frames,
/// or falls back to single-frame `DecodedImage` wrapped in a 1-frame animation.
pub fn decode_animated(data: &[u8]) -> Result<AnimatedImage, ImageError>;

/// Check if an image is animated without fully decoding.
pub fn is_animated(data: &[u8]) -> Result<bool, ImageError>;

/// Streaming: decode frames one at a time (for memory-constrained environments).
pub struct FrameDecoder { /* ... */ }

impl FrameDecoder {
    pub fn new(data: &[u8]) -> Result<Self, ImageError>;
    pub fn canvas_info(&self) -> (u32, u32, PixelFormat);
    pub fn loop_count(&self) -> u32;
    pub fn frame_count(&self) -> Option<u32>; // None if unknown
    pub fn next_frame(&mut self) -> Result<Option<AnimationFrame>, ImageError>;
}
```

**Two API levels:**
1. **Bulk:** `decode_animated()` — loads all frames into memory. Simple API, works for
   small animations (<50 frames, <256×256). Analogous to `decode()`.
2. **Streaming:** `FrameDecoder` — yields frames one at a time. Required for WASM
   memory constraints and large GIFs. Caller processes/encodes each frame before
   requesting the next.

### 2.3 Proposed Encoder API

```rust
/// Encode an animated image to the specified format.
pub fn encode_animated(
    image: &AnimatedImage,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError>;

/// Streaming encoder for memory-constrained environments.
pub struct FrameEncoder { /* ... */ }

impl FrameEncoder {
    pub fn new(format: &str, canvas_width: u32, canvas_height: u32,
               loop_count: u32, quality: Option<u8>) -> Result<Self, ImageError>;
    pub fn add_frame(&mut self, frame: &AnimationFrame) -> Result<(), ImageError>;
    pub fn finish(self) -> Result<Vec<u8>, ImageError>;
}
```

### 2.4 Per-Frame Filter Application Strategy

**Approach: Coalesced frames, independent filtering.**

1. **Coalesce on decode:** Convert sub-region frames with disposal/blend into
   full-canvas RGBA frames. This is what ImageMagick `-coalesce` does. Each frame
   becomes a complete image that can be independently filtered.

2. **Apply filter to each frame:** `for frame in &mut animation.frames { filter(&mut frame) }`

3. **Optimize on encode:** After filtering, run disposal optimization to minimize
   output size (ImageMagick `-layers optimize`). This is optional — the naive
   approach (full frames with `DisposeOp::Background`) works but is larger.

**Why coalesce-first:**
- Filters don't need to understand disposal/blend semantics
- Each frame is a standalone image — all existing filter functions work unchanged
- No risk of filter artifacts at sub-region boundaries
- Cost: more memory (full canvas per frame instead of sub-regions)

**Alternative (rejected): Sub-region filtering.**
- Would preserve compression efficiency but requires every filter to handle
  offset/clip boundaries. This is a massive API change for minimal benefit.

### 2.5 Memory Budget Analysis

| Scenario | Frames | Canvas | Per-frame | Total | WASM fit? |
|----------|--------|--------|-----------|-------|-----------|
| Small sticker | 24 | 128×128 | 64KB | 1.5MB | easily |
| Web emoji | 30 | 256×256 | 256KB | 7.5MB | yes |
| Typical GIF | 100 | 500×500 | 1MB | 95MB | tight |
| Large GIF | 300 | 1024×1024 | 4MB | 1.2GB | **no** |
| WebP sticker | 24 | 512×512 | 1MB | 24MB | yes |

**WASM memory ceiling:** Default ~256MB, configurable up to 4GB.

**Conclusion:** Buffered approach works for stickers and small animations (<100
frames, <512 canvas). Large GIFs require the streaming `FrameDecoder`/`FrameEncoder`
API that processes one frame at a time.

**Recommended hybrid:** Decode to `AnimatedImage` if estimated total < 128MB
(check frame_count × canvas_size × 4). Otherwise, require streaming API.

---

## 3. Integration Assessment

### 3.1 Impact on Existing Types

**ImageInfo:** No changes needed. Stays single-frame focused. Animation metadata
lives in the new `AnimatedImage` / `FrameMetadata` types.

**DecodedImage:** No changes needed. `decode()` continues to return the first frame.
`decode_animated()` is a separate entry point.

**Pipeline:** The existing `decode -> filter -> encode` pipeline stays unchanged for
single images. A new `decode_animated -> for_each_frame(filter) -> encode_animated`
pipeline is added alongside it, not replacing it.

### 3.2 WASM Component Model Constraints

- **Memory:** Addressed by streaming API for large animations
- **WIT interface:** New types needed:
  ```wit
  record animation-frame-meta {
      delay-ms: u32,
      offset-x: u32,
      offset-y: u32,
      dispose: dispose-op,
      blend: blend-op,
  }

  enum dispose-op { none, background, previous }
  enum blend-op { source, over }

  record animation-info {
      canvas-width: u32,
      canvas-height: u32,
      loop-count: u32,
      frame-count: option<u32>,
  }
  ```
- **Resource types:** `FrameDecoder` and `FrameEncoder` map well to WASI resource types
  (stateful objects with methods). This is the standard pattern for streaming in
  Component Model.
- **No blocking:** All frame iteration is synchronous pull-based (caller calls
  `next_frame()`), which works well with WASM's single-threaded model.

### 3.3 Recommended Implementation Order

**Phase 1: GIF animation decode (lowest effort, highest impact)**
- gif crate 0.14 already has full frame iterator API
- Only need to: loop `read_next_frame()`, build `AnimatedImage`
- Enables: animated GIF → static frame extraction, frame count, metadata
- Estimated effort: 1 track

**Phase 2: GIF animation encode**
- gif crate 0.14 has `write_frame()` API
- Add `encode_animated("gif", ...)` path
- Enables: filtered GIF output, GIF optimization
- Estimated effort: 1 track

**Phase 3: Per-frame filter pipeline**
- Coalesce frames, apply filter to each, re-encode
- Add `pipeline::process_animated()` function
- Enables: batch filter all frames of an animation
- Estimated effort: 1 track

**Phase 4: WebP animation decode**
- image-webp 0.2 has `next_frame()` API
- Map WebP-specific metadata to unified types
- Estimated effort: 1 track

**Phase 5: APNG decode**
- png crate 0.17+ supports APNG via `next_frame()`
- Map APNG fcTL metadata to unified types
- Estimated effort: 1 track

**Phase 6: Streaming API + WASM integration**
- `FrameDecoder` / `FrameEncoder` for memory-constrained environments
- WIT interface definitions
- Estimated effort: 1-2 tracks

**Phase 7: WebP/APNG animation encode**
- WebP: requires RIFF muxing (not in rasmcore-webp or image-webp)
- APNG: use `apng` crate or custom chunk writing
- Estimated effort: 1-2 tracks

**Total: 7-9 tracks, GIF decode as the entry point.**

---

## 4. Open Questions for Implementation

1. **Should coalesced frames be mandatory?** Or should the API expose both
   raw sub-region frames and coalesced full-canvas frames?

2. **Frame-level ICC profiles:** GIF doesn't have them, but WebP/APNG could
   theoretically have per-frame ICC. Is this worth supporting?

3. **Temporal filters:** Should the animation pipeline support filters that
   look at adjacent frames (e.g., temporal denoise, frame interpolation)?
   This would require buffering 2-3 frames, which the streaming API can handle.

4. **Optimization passes:** Should the encoder automatically run disposal
   optimization, or should this be explicit? ImageMagick has both `-layers optimize`
   (automatic) and `-layers OptimizeFrame` / `-layers OptimizeTransparency`.

---

## Sources

- [image-webp crate](https://crates.io/crates/image-webp) — WebP decoder/encoder
- [webp-animation crate](https://github.com/blaind/webp-animation) — WebP animation wrapper
- [png crate](https://crates.io/crates/png) — PNG/APNG decoder
- [apng crate](https://crates.io/crates/apng) — APNG encoder
- [image-rs/image-webp](https://github.com/image-rs/image-webp) — image-webp source
- [image-rs/image-png](https://github.com/image-rs/image-png) — png crate source
