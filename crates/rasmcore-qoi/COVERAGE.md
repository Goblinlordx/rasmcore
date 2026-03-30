# QOI Codec Coverage

Compared: rasmcore-qoi vs image crate (qoi 0.4) vs QOI specification (qoiformat.org).

## Decode/Encode Coverage

| Feature | rasmcore-qoi | image crate | QOI Spec |
|---------|:---:|:---:|:---:|
| **Channels** | | | |
| RGB (3 channels) | Y | Y | Y |
| RGBA (4 channels) | Y | Y | Y |
| **Colorspace** | | | |
| sRGB (0) | Y | Y | Y |
| Linear (1) | Y | Y | Y |
| **Operation Types** | | | |
| QOI_OP_RGB | Y | Y | Y |
| QOI_OP_RGBA | Y | Y | Y |
| QOI_OP_INDEX (64-entry hash) | Y | Y | Y |
| QOI_OP_DIFF (2-bit delta) | Y | Y | Y |
| QOI_OP_LUMA (green delta) | Y | Y | Y |
| QOI_OP_RUN (1-62 pixels) | Y | Y | Y |
| **Other** | | | |
| Magic bytes ("qoif") | Y | Y | Y |
| 8-byte end marker | Y | Y | Y |
| Hash: (r*3+g*5+b*7+a*11)%64 | Y | Y | Y |

## Assessment

**Full feature parity with image crate.** QOI is a simple format — both
implementations are spec-complete. rasmcore-qoi has zero external dependencies
(image crate uses the `qoi` crate internally).

## Gaps vs QOI Spec (future work)

None — the format is fully covered. QOI has no optional features or extensions.

## Conclusion

**Safe to replace image crate for QOI.** 6/6 operation types, both channel
modes, both colorspaces. Spec-complete implementation.
