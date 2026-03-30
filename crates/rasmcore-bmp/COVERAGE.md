# BMP Codec Coverage

Compared: rasmcore-bmp vs image crate (0.25) vs full BMP specification.

## Decode Coverage

| Feature | rasmcore-bmp | image crate | Full Spec |
|---------|:---:|:---:|:---:|
| **Bit Depths** | | | |
| 1-bit (monochrome indexed) | Y | Y | Y |
| 2-bit (CGA, legacy) | **N** | N | Y |
| 4-bit (VGA indexed) | Y | Y | Y |
| 8-bit (256-color indexed) | Y | Y | Y |
| 16-bit (RGB555/bitfields) | Y | Y | Y |
| 24-bit (RGB) | Y | Y | Y |
| 32-bit (RGBA/bitfields) | Y | Y | Y |
| **Compression** | | | |
| BI_RGB (uncompressed) | Y | Y | Y |
| BI_RLE8 | Y | Y | Y |
| BI_RLE4 | Y | Y | Y |
| BI_BITFIELDS (custom masks) | Y | Y | Y |
| BI_ALPHABITFIELDS | partial | partial | Y |
| BI_JPEG (embedded JPEG) | N | N | Y |
| BI_PNG (embedded PNG) | N | N | Y |
| Huffman 1D (OS/2) | N | N | Y |
| **Header Variants** | | | |
| BITMAPCOREHEADER (12B, OS/2 1.x) | Y | Y | Y |
| BITMAPINFOHEADER (40B, Win NT) | Y | Y | Y |
| BITMAPV2INFOHEADER (52B) | Y | N | Y |
| BITMAPV3INFOHEADER (56B) | Y | N | Y |
| BITMAPV4HEADER (108B) | Y | Y | Y |
| BITMAPV5HEADER (124B) | Y | partial | Y |
| OS22XBITMAPHEADER (64B+) | N | N | Y |
| **Orientation** | | | |
| Bottom-up (positive height) | Y | Y | Y |
| Top-down (negative height) | Y | Y | Y |
| **Other** | | | |
| Row padding (4-byte align) | Y | Y | Y |
| Palette BGR→RGBA conversion | Y | Y | Y |
| RLE escape sequences (EOL/EOF/DELTA) | Y | partial | Y |

## Encode Coverage

| Feature | rasmcore-bmp | image crate | Full Spec |
|---------|:---:|:---:|:---:|
| 1-bit indexed | Y | N | Y |
| 4-bit indexed | Y | N | Y |
| 4-bit RLE | Y | N | Y |
| 8-bit indexed | Y | Y | Y |
| 8-bit RLE | Y | N | Y |
| 8-bit grayscale (auto palette) | Y | N | Y |
| 16-bit RGB (555) | Y | N | Y |
| 24-bit RGB | Y | Y | Y |
| 32-bit RGBA | Y | Y | Y |

## Assessment

**rasmcore-bmp exceeds image crate coverage** in most areas:
- More header variants (V2, V3 undocumented headers)
- More encode modes (RLE4, RLE8, indexed, grayscale, 16-bit)
- Full RLE escape sequence handling

## Gaps vs Full BMP Spec (future work)

| Gap | Priority | Notes |
|-----|----------|-------|
| 2-bit color depth | Low | Legacy CGA format, extremely rare |
| BI_JPEG (embedded JPEG) | Low | Print-only format, not used in practice |
| BI_PNG (embedded PNG) | Low | Print-only format, not used in practice |
| OS/2 2.x halftoning | Low | Legacy OS/2 feature |
| ICC color profile application | Medium | V4/V5 headers contain ICC data, parsed but not applied |
| CIEXYZ color space transform | Medium | V4/V5 color endpoints, parsed but not transformed |
| Gamma correction | Medium | V4/V5 gamma values, parsed but not applied |

## Conclusion

**Safe to replace image crate for BMP.** rasmcore-bmp handles all variants the
image crate handles, plus additional ones (V2/V3 headers, RLE encode, indexed
encode). No coverage regression from this swap.
