# TGA Codec Coverage

Compared: rasmcore-tga vs image crate (0.25) vs TGA 2.0 specification.

## Decode Coverage

| Feature | rasmcore-tga | image crate | TGA 2.0 Spec |
|---------|:---:|:---:|:---:|
| **Image Types** | | | |
| Type 1 — Color-mapped (raw) | Y | Y | Y |
| Type 2 — True-color (raw) | Y | Y | Y |
| Type 3 — Grayscale (raw) | Y | Y | Y |
| Type 9 — Color-mapped (RLE) | Y | Y | Y |
| Type 10 — True-color (RLE) | Y | Y | Y |
| Type 11 — Grayscale (RLE) | Y | Y | Y |
| **Bit Depths** | | | |
| 8-bit (grayscale/indexed) | Y | Y | Y |
| 16-bit (A1R5G5B5 true-color) | Y | **N** | Y |
| 16-bit (grayscale + alpha) | Y | Y | Y |
| 24-bit (RGB) | Y | Y | Y |
| 32-bit (RGBA) | Y | Y | Y |
| **Orientation** | | | |
| Bottom-up (default) | Y | Y | Y |
| Top-down (negative height) | Y | Y | Y |
| **Other** | | | |
| RLE run + raw packets | Y | Y | Y |
| 128-pixel max run length | Y | Y | Y |
| Color map entry 16/24/32-bit | Y | Y | Y |

## Encode Coverage

| Feature | rasmcore-tga | image crate | TGA 2.0 Spec |
|---------|:---:|:---:|:---:|
| 8-bit grayscale | Y | Y | Y |
| 16-bit grayscale+alpha | Y | N | Y |
| 16-bit RGB (A1R5G5B5) | Y | **N** | Y |
| 24-bit RGB | Y | Y | Y |
| 24-bit RGB RLE | Y | Y | Y |
| 32-bit RGBA | Y | Y | Y |
| 32-bit RGBA RLE | Y | Y | Y |
| 8-bit grayscale RLE | Y | N | Y |
| 8-bit color-mapped | Y | N | Y |
| 8-bit color-mapped RLE | Y | N | Y |

## Assessment

**rasmcore-tga exceeds image crate coverage** — supports 16-bit true-color
(A1R5G5B5), grayscale+alpha encode, color-mapped encode/RLE, and grayscale RLE.

## Gaps vs TGA 2.0 Spec (future work)

| Gap | Priority | Notes |
|-----|----------|-------|
| Extension Area | Low | Optional metadata (author, comments, timestamps) |
| Developer Area | Low | Custom application data blocks |
| Footer/Signature | Low | "TRUEVISION-XFILE" signature for TGA 2.0 detection |
| Thumbnails | Low | Embedded thumbnail images |
| 1-bit alpha in 16-bit | Low | A1R5G5B5 alpha bit treated as opaque (typical behavior) |

## Conclusion

**Safe to replace image crate for TGA.** rasmcore-tga handles all variants
the image crate handles, plus 16-bit true-color and additional encode modes.
