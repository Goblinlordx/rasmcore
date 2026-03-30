# PNM Codec Coverage

Compared: rasmcore-pnm vs image crate (0.25) vs Netpbm specification.

## Decode Coverage

| Feature | rasmcore-pnm | image crate | Netpbm Spec |
|---------|:---:|:---:|:---:|
| **Format Types** | | | |
| P1 — PBM ASCII | Y | Y | Y |
| P2 — PGM ASCII | Y | Y | Y |
| P3 — PPM ASCII | Y | Y | Y |
| P4 — PBM Binary | Y | Y | Y |
| P5 — PGM Binary | Y | Y | Y |
| P6 — PPM Binary | Y | Y | Y |
| P7 — PAM | Y | Y | Y |
| PF — PFM RGB (float) | Y | **N** | Y (extension) |
| Pf — PFM Gray (float) | Y | **N** | Y (extension) |
| **Bit Depths** | | | |
| 1-bit (PBM) | Y | Y | Y |
| 8-bit (maxval <= 255) | Y | Y | Y |
| 16-bit (maxval > 255, big-endian) | Y | Y | Y |
| 32-bit float (PFM) | Y | **N** | Y |
| Maxval range 1-65535 | Y | Y | Y |
| **PAM Features** | | | |
| WIDTH/HEIGHT/DEPTH/MAXVAL | Y | Y | Y |
| TUPLTYPE header | Y | partial | Y |
| ENDHDR terminator | Y | Y | Y |
| Alpha channels (RGBA, GA) | Y | **N** (rejected) | Y |
| Arbitrary depth (1-4) | Y | partial | Y |
| **Other** | | | |
| Comment handling (#) | Y | Y | Y |
| PFM endianness detection | Y | N | Y |
| PFM bottom-to-top flip | Y | N | Y |
| Auto-maxval detection (encode) | Y | N | convenience |

## Encode Coverage

| Feature | rasmcore-pnm | image crate | Netpbm Spec |
|---------|:---:|:---:|:---:|
| PBM binary (P4) | Y | Y | Y |
| PBM ASCII (P1) | Y | **N** | Y |
| PGM binary (P5) | Y | Y | Y |
| PGM ASCII (P2) | Y | **N** | Y |
| PGM 16-bit (P5) | Y | Y | Y |
| PPM binary (P6) | Y | Y | Y |
| PPM ASCII (P3) | Y | **N** | Y |
| PPM 16-bit (P6) | Y | Y | Y |
| PAM (P7) | Y | Y | Y |
| PFM RGB (PF) | Y | **N** | Y |
| PFM Gray (Pf) | Y | **N** | Y |

## Assessment

**rasmcore-pnm significantly exceeds image crate coverage:**
- PFM (float) support — exclusive to rasmcore-pnm
- ASCII encode modes (P1, P2, P3) — image crate only does binary
- PAM alpha channels — image crate rejects RGBA/GA tuple types
- Auto-maxval detection in encoders

## Gaps vs Netpbm Spec (future work)

| Gap | Priority | Notes |
|-----|----------|-------|
| Multiple TUPLTYPE concatenation | Low | PAM spec allows multiple TUPLTYPE lines, should concatenate with spaces |
| PAM 16-bit encoder validation | Medium | Auto-compute maxval in encode_pam() like PGM/PPM do |
| Boundary maxval tests | Low | maxval=1, 256, 65535 untested (parser is generic) |

## Conclusion

**Safe to replace image crate for PNM.** rasmcore-pnm handles everything
the image crate handles plus PFM, ASCII encode modes, and PAM alpha channels.
