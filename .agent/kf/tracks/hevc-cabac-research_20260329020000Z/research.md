# HEVC CABAC Context Derivation — Research Findings

## Source: libde265 v1.0.18 (slice.cc) + ITU-T H.265

---

## 1. coded_sub_block_flag Context (libde265 line 1887)

```
csbfCtx = (coded_sub_block_neighbors & 1) | (coded_sub_block_neighbors >> 1)
ctxIdxInc = csbfCtx + (cIdx != 0 ? 2 : 0)
```

Uses `CONTEXT_MODEL_CODED_SUB_BLOCK_FLAG + ctxIdxInc` (4 contexts total).

**Our code**: Uses `SIG_COEFF_CTX_OFFSET` (wrong model entirely).
**Fix**: Add `CODED_SUB_BLOCK_FLAG_CTX_OFFSET` with 4 dedicated contexts.

---

## 2. sig_coeff_flag Context (libde265 lines 2059-2126)

### For log2TrafoSize == 2 (4x4 TU):
```
sigCtx = ctxIdxMap[(yC << 2) + xC]
ctxIdxMap = [0,1,4,5, 2,3,4,5, 6,6,8,8, 7,7,8,99]
```

### For log2TrafoSize > 2, position (0,0):
```
sigCtx = 0
```

### For log2TrafoSize > 2, non-DC positions:
```
xP = xC & 3  // position within 4x4 sub-block
yP = yC & 3
xS = xC >> 2  // sub-block index
yS = yC >> 2

switch (prevCsbf) {
  case 0: sigCtx = (xP+yP >= 3) ? 0 : (xP+yP > 0) ? 1 : 2
  case 1: sigCtx = (yP == 0) ? 2 : (yP == 1) ? 1 : 0
  case 2: sigCtx = (xP == 0) ? 2 : (xP == 1) ? 1 : 0
  default: sigCtx = 2
}

if (cIdx == 0) {  // luma
  if (xS + yS > 0) sigCtx += 3  // not first sub-block
  if (sbWidth == 2) sigCtx += (scanIdx==0) ? 9 : 15  // 8x8 TU
  else sigCtx += 21  // 16x16 or 32x32 TU
}
```

Final: `ctxIdxInc = (cIdx == 0) ? sigCtx : 27 + sigCtx`

**Our code**: Uses simplified diagonal-based offset, ignoring `prevCsbf` and sub-block position.
**Fix**: Implement the exact switch/case with prevCsbf and sub-block index offset.

---

## 3. coeff_abs_level_greater1_flag Context (libde265 lines 2385-2444)

### ctxSet derivation (called from sub-block loop, lines 3249-3253):
```
if (i == 0 || cIdx > 0) ctxSet = 0
else ctxSet = 2
if (c1 == 0) ctxSet++
```
Where `c1` starts at 1, resets to 0 on gt1=1, increments (up to 3) on gt1=0.

### greater1Ctx derivation:
```
if (firstCoeffInSubblock):
  if (firstSubblock): lastGreater1Ctx = 1
  else: lastGreater1Ctx = lastSubblock_greater1Ctx
  if (lastGreater1Ctx == 0): ctxSet++
  greater1Ctx = 1
else:
  greater1Ctx = lastInvocation_greater1Ctx
  if (greater1Ctx > 0):
    if (lastGreater1Flag == 1): greater1Ctx = 0
    else: greater1Ctx++

ctxSet = c1  // LINE 2437: override with the c1 parameter!
```

### Final context index:
```
ctxIdxInc = ctxSet * 4 + min(greater1Ctx, 3)
if (cIdx > 0) ctxIdxInc += 16
```

**Our code**: Uses `c1.min(3) * 4` for ctxSet which gives wrong results because
c1 should be the ctxSet computed at the sub-block level, not a simple counter.
**Fix**: Track `firstSubblock`, `lastSubblock_greater1Ctx`, and use the full
derivation from lines 2407-2437.

---

## 4. coeff_abs_level_greater2_flag Context (libde265 line 3297)

```
ctxIdxInc = lastInvocation_ctxSet
if (cIdx > 0) ctxIdxInc += 4
```

**Our code**: Uses fixed offset.
**Fix**: Use the ctxSet from the gt1 derivation.

---

## 5. Sign coding and signHidden (libde265 lines 3306-3327)

```
signHidden = (coeff_scan_pos[0] - coeff_scan_pos[nCoefficients-1] > 3)
```
If signHidden, the LAST coefficient's sign is NOT coded (inferred from parity).

**Our code**: Always decodes all signs.
**Fix**: Implement signHidden — skip last sign when first-last scan distance > 3.

---

## 6. coeff_abs_level_remaining (libde265 lines 3342-3390)

Rice parameter adaptation:
```
if (baseLevel + remaining > 3 * (1 << ricePar)): ricePar++
```
Where `baseLevel = 1 + greater1 + greater2` and max ricePar = 4.

**Our code**: Fixed ricePar=0, no adaptation.
**Fix**: Track baseLevel per coefficient and adapt ricePar.

---

## 7. Sub-block neighbor tracking (libde265 lines 3123-3126)

After decoding a coded sub-block:
```
if (S.x > 0) coded_sub_block_neighbors[S.x-1 + S.y*sbWidth] |= 1  // left
if (S.y > 0) coded_sub_block_neighbors[S.x + (S.y-1)*sbWidth] |= 2  // above
```

The `prevCsbf` for each sub-block is `coded_sub_block_neighbors[S.x + S.y*sbWidth]`.
Values 0-3: 0=neither, 1=right, 2=bottom, 3=both.

**Our code**: No neighbor tracking at all.
**Fix**: Add `coded_sub_block_neighbors` array and update after each sub-block.

---

## Summary: What needs to change in our code

| Element | Our current approach | Correct approach (from libde265) |
|---------|---------------------|----------------------------------|
| coded_sub_block_flag ctx | Uses SIG_COEFF context (wrong) | Separate CSBF context, 4 models, derived from neighbors |
| sig_coeff_flag ctx | Diagonal-only, no prevCsbf | prevCsbf-dependent switch/case + sub-block offset |
| gt1 ctxSet | c1 counter only | ctxSet from sub-block position + c1 + lastSubblock_greater1Ctx |
| gt1 greater1Ctx | Same as ctxSet | Separate counter, starts at 1, adapts per coefficient |
| gt2 ctx | Fixed offset | Uses ctxSet from gt1 derivation |
| signHidden | Not implemented | Skip last sign when scan distance > 3 |
| Rice param | Fixed 0 | Adaptive based on decoded level |
| Sub-block neighbors | Not tracked | coded_sub_block_neighbors array with |= 1/2 updates |
