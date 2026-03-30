# JPEG Decoder Coverage

Compared: rasmcore-jpeg decode vs image crate (zune-jpeg 0.5) vs ITU-T T.81/JFIF spec.

## Decode Coverage

| Feature | rasmcore-jpeg | image/zune-jpeg | Full Spec |
|---------|:---:|:---:|:---:|
| **Frame Types** | | | |
| SOF0 — Baseline sequential (8-bit) | Y | Y | Y |
| SOF1 — Extended sequential (12-bit) | N | partial | Y |
| SOF2 — Progressive | Y | Y | Y |
| SOF9 — Arithmetic sequential | Y | N | Y |
| SOF10 — Arithmetic progressive | Y | N | Y |
| SOF3/7/11/15 — Lossless | N | N | Y |
| **Subsampling** | | | |
| 4:4:4 (H1V1) | Y | Y | Y |
| 4:2:2 (H2V1) | Y | Y | Y |
| 4:2:0 (H2V2) | Y | Y | Y |
| 4:1:1 (H4V1) | Y (generic) | Y | Y |
| Arbitrary h_samp/v_samp ratios | Y | Y | Y |
| **Color** | | | |
| YCbCr → RGB | Y | Y | Y |
| Grayscale (1 component) | Y | Y | Y |
| CMYK (4 components) | **N** | Y | Y |
| YCCK (4 components) | **N** | Y | Y |
| **Features** | | | |
| Restart markers (DRI) | Y | Y | Y |
| Multiple scan (progressive) | Y | Y | Y |
| Huffman tables (DHT) | Y | Y | Y |
| Quantization tables (DQT) | Y | Y | Y |
| Fancy upsampling (3:1 filter) | Y | Y | Y |
| APP markers (skip) | Y | Y | Y |
| EXIF extraction | N (skipped) | N (skipped) | Y |
| **Encoding (included in crate)** | | | |
| Baseline sequential encode | Y | N/A | Y |
| Turbo mode (no trellis) | Y | N/A | Y |
| Quality mode (trellis + opt Huffman) | Y | N/A | Y |
| Progressive encode | Y | N/A | Y |
| Arithmetic encode | Y | N/A | Y |

## Assessment

**rasmcore-jpeg exceeds zune-jpeg in some areas** (arithmetic coding support)
but has a **coverage gap for CMYK/YCCK images** (4-component JPEGs).

CMYK JPEGs are rare in consumer/web contexts but exist in print workflows.

## Gaps vs Full JPEG Spec (future work)

| Gap | Priority | Notes |
|-----|----------|-------|
| CMYK/YCCK decode (4 components) | Medium | Used in print workflows, rare in web |
| 12-bit sample precision (SOF1) | Low | Medical/scientific imaging |
| Lossless JPEG (SOF3/7/11/15) | Low | Largely superseded by JPEG-LS |
| EXIF metadata pass-through | Medium | Currently skipped, metadata handled at rasmcore-image layer |

## Conclusion

**Safe to replace zune-jpeg for the vast majority of JPEG files.** The only
regression is CMYK/YCCK support (4-component images). These are rare enough
that we can accept the gap and document it. If a CMYK JPEG is encountered,
the decoder returns an error rather than silently producing wrong output.
