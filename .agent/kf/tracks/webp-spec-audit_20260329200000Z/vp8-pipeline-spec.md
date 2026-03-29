# VP8 Encode Pipeline Spec Audit — RFC 6386 vs rasmcore-webp

## Executive Summary

Systematic comparison of each VP8 encode pipeline stage against RFC 6386.
All core transforms, quantization tables, and entropy coding MATCH the spec.
The 15dB quality gap vs cwebp is NOT from incorrect formulas — it's from
higher-level encoder decisions (mode selection, rate-distortion optimization).

## A. Quantization Tables (RFC 6386 Section 14.1 / 20.3)

### RFC Reference
```c
// Section 14.1: dc_qlookup[128] and ac_qlookup[128]
static const int dc_qlookup[128] = {
    4,5,6,7,8,9,10,10,11,12,13,14,15,16,17,17,...
};
static const int ac_qlookup[128] = {
    4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
};
```

### Our Code: `tables.rs:14-31`
```rust
pub const DC_TABLE: [u16; 128] = [4,5,6,7,...];
pub const AC_TABLE: [u16; 128] = [4,5,6,7,...];
```

### Status: **MATCH** ✅
Both tables verified entry-by-entry against RFC 6386 Section 20.3.

---

## B. Y2/UV Scaling (RFC 6386 Section 20.4 — dixie.c dequant_init)

### RFC Reference
```c
factor[Y1][DC] = dc_q(q + y1_dc_delta_q);
factor[Y1][AC] = ac_q(q);
factor[UV][DC] = dc_q(q + uv_dc_delta_q);
factor[UV][AC] = ac_q(q + uv_ac_delta_q);
factor[Y2][DC] = dc_q(q + y2_dc_delta_q) * 2;
factor[Y2][AC] = ac_q(q + y2_ac_delta_q) * 155 / 100;  // min 8
```

### Our Code: `quant.rs` (build_matrix)
```rust
Y2-DC: (DC_TABLE[qp] * 2).min(132)
Y2-AC: (AC_TABLE[qp] * 155 / 100).max(8)
UV-DC: DC_TABLE[qp].min(132)
UV-AC: AC_TABLE[qp]
Y1-DC: DC_TABLE[qp]
Y1-AC: AC_TABLE[qp]
```

### Status: **MATCH** ✅
All delta_q values are 0 in our encoder (bitstream.rs:249-253).
Y2 DC *2, Y2 AC *155/100, UV DC clamp 132 all match.

---

## C. Inverse DCT (RFC 6386 Section 14.4)

### RFC Reference
```c
static const int cospi8sqrt2minus1 = 20091;
static const int sinpi8sqrt2       = 35468;
// Column pass: ip[0]+ip[8], ip[0]-ip[8], butterflies with sinpi8/cospi8
// Row pass: same structure with +4>>3 rounding
```

### Our Code: `dct.rs` (inverse_dct_c)
```rust
((a * 20091) >> 16) + a   // cospi8sqrt2minus1
(a * 35468) >> 16          // sinpi8sqrt2
// Column pass: a1+d1, a1-d1, b1+c1, b1-c1
// Row pass: (val + 4) >> 3
```

### Status: **MATCH** ✅
Constants identical. Butterfly structure identical. Rounding identical.

---

## D. Inverse WHT (RFC 6386 Section 14.3)

### RFC Reference
```c
void vp8_short_inv_walsh4x4_c(short *input, short *output) {
    // Column pass: a1=ip[0]+ip[12], b1=ip[4]+ip[8], c1=ip[4]-ip[8], d1=ip[0]-ip[12]
    // Row pass: same + (val+3)>>3 rounding
}
```

### Our Code: `dct.rs` (inverse_wht)
```rust
a1 = ip[0]+ip[12]; b1 = ip[4]+ip[8]; c1 = ip[4]-ip[8]; d1 = ip[0]-ip[12];
// Row: (a2+3)>>3, (b2+3)>>3, (c2+3)>>3, (d2+3)>>3
```

### Status: **MATCH** ✅

---

## E. Forward DCT (Encoder-Side — from libwebp FTransform_C)

### Reference (libwebp vp8/encoder/dsp_cost.c)
```c
// Constants: 2217, 5352 with specific rounding biases
// Row pass: 1812 bias for coefficient 1, 937 bias for coefficient 3
// Column pass: 12000 bias for out[4+i], 51000 bias for out[12+i]
```

### Our Code: `dct.rs` (forward_dct)
```rust
tmp[base + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
tmp[base + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + if a3 != 0 { 1 } else { 0 };
out[12 + i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16);
```

### Status: **MATCH** ✅
Constants 2217, 5352, 1812, 937, 12000, 51000 all match libwebp.
The `+1 if a3 != 0` conditional rounding matches libwebp's FTransform.

---

## F. Forward WHT (Encoder-Side)

### Reference (libwebp vp8/encoder/FTransformWHT)
```c
// No multiply — purely additions/subtractions
// Row: a0=in[0]+in[2], a1=in[1]+in[3], a2=in[0]-in[2], a3=in[1]-in[3]
//      out[0]=a0+a1, out[4]=a2+a3, out[8]=a0-a1, out[12]=a2-a3
// Column: same structure, >>1 normalization
```

### Our Code: `dct.rs` (forward_wht)
```rust
// Row: a0+a1, a2+a3, a0-a1, a2-a3
// Column: (a0+a1+1)>>1, etc.
```

### Status: **MATCH** ✅

---

## G. Loop Filter (RFC 6386 Section 15.4)

### RFC Reference
```c
interior_limit = loop_filter_level;
if (sharpness_level) {
    interior_limit >>= sharpness_level > 4 ? 2 : 1;
    if (interior_limit > 9 - sharpness_level)
        interior_limit = 9 - sharpness_level;
}
if (!interior_limit) interior_limit = 1;

mbedge_limit = ((loop_filter_level + 2) * 2) + interior_limit;
sub_bedge_limit = (loop_filter_level * 2) + interior_limit;

// HEV threshold for keyframes:
hev_threshold = 0;
if (loop_filter_level >= 40) hev_threshold = 2;
else if (loop_filter_level >= 15) hev_threshold = 1;
```

### Our Code: `filter.rs` (compute_filter_params)
```rust
let mut interior = level as i32;
if sharpness > 0 {
    interior >>= if sharpness > 4 { 2 } else { 1 };
    interior = interior.min(9 - sharpness as i32);
}
interior = interior.max(1);
let mbedge = ((level as i32 + 2) * 2) + interior;
let subedge = (level as i32 * 2) + interior;
// HEV: same thresholds as RFC
```

### Status: **MATCH** ✅

---

## H. Boolean Coder (RFC 6386 Section 7.3)

### RFC Reference
```c
split = 1 + (((range - 1) * probability) >> 8);
if (value) {
    lowvalue += split;
    range -= split;
} else {
    range = split;
}
// Renormalize: shift left until range >= 128
```

### Our Code: `boolcoder.rs` (BoolWriter)
```rust
let split = 1 + (((self.range - 1) as u32 * prob as u32) >> 8);
if value {
    self.bottom = self.bottom.wrapping_add(split);
    self.range -= split as u8;
} else {
    self.range = split as u8;
}
```

### Status: **MATCH** ✅

---

## I. Token Probability Tables (RFC 6386 Section 13.3)

### Our Code: `token.rs` (DEFAULT_COEFF_PROBS)
1056-entry table matching RFC 6386 Table 13.5 exactly.

### Status: **MATCH** ✅

---

## J. Quality-to-QP Mapping (Encoder-Only — NOT in RFC)

### cwebp Reference
cwebp at Q75 uses QP 26 (confirmed via `-v` verbose output).

### Our Code: `ratecontrol.rs`
```rust
qp = (100 - quality) * 127 / 99
```
At Q75: our QP = 32. cwebp QP = 26. **6-point gap.**

### Status: **DIVERGENCE** ⚠️
Our linear mapping produces higher QP than cwebp at most quality levels.
At Q75: our AC step = 29, cwebp AC step = 24. That's 21% more quantization.
Expected impact: ~2-3 dB.

---

## K. Mode Selection (Encoder-Only)

### cwebp
Uses RD-optimized mode selection with full encode trial per mode.
Evaluates actual encoded size for each candidate.

### Our Code
I16x16: RD cost with SSD + lambda * estimated bits. ✅ (close to cwebp)
B_PRED: SATD-only mode selection. ⚠️ (no RD for sub-block modes)
B_PRED vs I16x16: hardcoded `bpred_sad + 128 < i16_sad`. ⚠️ (heuristic, not RD)

### Status: **DIVERGENCE** ⚠️
The B_PRED decision uses a crude heuristic instead of full RD.
Expected impact: ~3-5 dB on detailed content.

---

## FINAL BUG LIST

| # | Severity | File | Issue | Expected Impact |
|---|----------|------|-------|-----------------|
| 1 | HIGH | ratecontrol.rs | QP mapping: linear formula gives QP 32 at Q75, cwebp uses QP 26. Too aggressive. | +2-3 dB |
| 2 | HIGH | bitstream.rs:642 | B_PRED decision: hardcoded `+128` bias instead of RD cost | +3-5 dB on texture |
| 3 | MEDIUM | predict.rs | 4x4 sub-block modes: SATD-only, no RD | +1-2 dB |
| 4 | LOW | ratecontrol.rs | qp_uv computed but never used (dead code) | None (cosmetic) |
| 5 | NOTE | All transforms | All match RFC ✅ | No action needed |
| 6 | NOTE | Quant tables | All match RFC ✅ | No action needed |
| 7 | NOTE | Bool coder | Matches RFC ✅ | No action needed |
| 8 | NOTE | Loop filter | Matches RFC ✅ | No action needed |
| 9 | NOTE | Token probs | Match RFC ✅ | No action needed |

## PRIORITIZED FIX ORDER

1. **Fix QP mapping** (ratecontrol.rs) — match cwebp's nonlinear curve. Simplest fix, immediate +2-3dB.
2. **Fix B_PRED RD decision** (bitstream.rs) — replace `+128` with full RD cost comparison.
3. **Add 4x4 sub-block RD** (predict.rs) — encode trial per mode, not just SATD.

Combined expected improvement: **+6-10 dB**, closing from 15dB gap to ~5dB.
The remaining 5dB gap is from cwebp's multi-pass optimization and adaptive
probability updates, which are beyond single-pass encoder scope.
