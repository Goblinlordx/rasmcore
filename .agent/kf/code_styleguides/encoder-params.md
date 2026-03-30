# Encoder Parameter Reference

Parameter mapping between rasmcore encoders and their reference implementations.

## JPEG

| Parameter | rasmcore | ImageMagick | Notes |
|-----------|----------|-------------|-------|
| Quality | `quality: u8` (1-100, default 85) | `-quality N` (1-100) | Same scale. Both use 1-100 linear mapping to quantization tables. Our encoder uses mozjpeg-derived tables; IM uses libjpeg tables. PSNR within 3 dB at matching quality levels. |
| Progressive | `progressive: bool` | `-interlace Plane` | Same behavior |
| Turbo mode | `turbo: bool` | N/A | rasmcore-specific: skips trellis/Huffman optimization for 3-10x speed |

**Quality scale comparison (256x256 gradient, same source):**

| Quality | Our PSNR (dB) | Our Size | IM PSNR (dB) | IM Size | PSNR Diff |
|---------|---------------|----------|--------------|---------|-----------|
| 25 | ~30 | ~1.1KB | ~30 | ~1.1KB | < 3 dB |
| 50 | ~33 | ~1.8KB | ~34 | ~1.9KB | < 3 dB |
| 75 | ~37 | ~2.5KB | ~38 | ~2.4KB | < 3 dB |
| 95 | ~46 | ~7KB | ~47 | ~7.2KB | < 3 dB |

**Known difference:** Our encoder uses mozjpeg-derived quantization which produces slightly different file sizes than IM's libjpeg. Quality perception is comparable (< 3 dB PSNR difference).

## WebP

| Parameter | rasmcore | cwebp (Google) | Notes |
|-----------|----------|----------------|-------|
| Quality | `quality: u8` (1-100, default 75) | `-q N` (0-100) | Same scale. Our native VP8 encoder has a known quality gap: 0.4-0.8x cwebp quality at low settings, closer at high. |
| Lossless | `lossless: bool` | `-lossless` | Our lossless currently falls back to quality=100 lossy |

**Known gap:** Our rasmcore-webp encoder produces lower quality than Google's cwebp at the same quality number. Root causes documented in project memory: missing RD B_PRED decision and segmentation. File sizes may be 2-3x larger at low quality.

## PNG

| Parameter | rasmcore | ImageMagick | Notes |
|-----------|----------|-------------|-------|
| Compression | `compression_level: u8` (0-9, default 6) | `-quality N0` (where N is zlib level) | Same 0-9 scale. We use fdeflate (fast) internally. |
| Filter | `filter_type: PngFilterType` | `-define png:filter-type=N` | Enum: NoFilter, Sub, Up, Avg, Paeth, Adaptive |

**Parity:** Pixel-exact roundtrip at all compression levels and filter types. Lossless format — no quality differences.

## TIFF

| Parameter | rasmcore | ImageMagick | Notes |
|-----------|----------|-------------|-------|
| Compression | `compression: TiffCompression` | `-compress Type` | Mapping below |

| rasmcore | IM equivalent | Status |
|----------|---------------|--------|
| `None` | `-compress None` | Pixel-exact roundtrip ✓ |
| `Lzw` (default) | `-compress LZW` | Pixel-exact roundtrip ✓ |
| `Deflate` | `-compress Zip` | Pixel-exact roundtrip ✓ |
| `PackBits` | `-compress RLE` | Pixel-exact roundtrip ✓ |

**Parity:** All compression types produce pixel-exact output on roundtrip.

## GIF

| Parameter | rasmcore | ImageMagick | ffmpeg | Notes |
|-----------|----------|-------------|--------|-------|
| Repeat | `repeat: u16` (0=infinite) | `-loop N` | `-loop N` | Same semantics |
| Frame delay | `FrameInfo.delay_ms: u32` | `-delay Ncs` | N/A | We use ms, IM uses centiseconds (cs = ms/10). Stored as cs in GIF format. |
| Disposal | `FrameInfo.disposal: DisposalMethod` | `-dispose Method` | N/A | None, Background, Previous |

**Animation timing:** Verified via ffprobe — encoded frames detected correctly.

## APNG

| Parameter | rasmcore | ffmpeg | Notes |
|-----------|----------|--------|-------|
| Compression | `PngEncodeConfig` (same as PNG) | N/A | Same PNG compression |
| Frame delay | `FrameInfo.delay_ms: u32` | Metadata | Stored as delay_num/delay_den in APNG |
| Disposal | `FrameInfo.disposal: DisposalMethod` | N/A | None, Background, Previous |
| Blend | Always `Source` | `Over` or `Source` | We always use Source blend |
| Per-frame size | `FrameInfo.width/height` | N/A | Sub-frame dimensions |
| Per-frame offset | `FrameInfo.x_offset/y_offset` | N/A | Sub-frame positioning |

**Disposal modes:** Validated via ffmpeg frame extraction — None and Background modes produce correct multi-frame output.

## AVIF

| Parameter | rasmcore | Notes |
|-----------|----------|-------|
| Quality | `quality: u8` (1-100, default 75) | **Encoding disabled** — rav1e dependency removed for binary size |
| Speed | `speed: u8` (1-10, default 6) | Not available |

## HEIC

| Parameter | rasmcore | Notes |
|-----------|----------|-------|
| Quality | `quality: u8` (1-100, default 75) | Requires `nonfree-hevc` feature flag. Quality maps linearly to HEVC QP (100→QP4, 50→QP27, 1→QP51). |

## BMP / ICO / QOI

No configurable parameters. Lossless, pixel-exact roundtrip.
