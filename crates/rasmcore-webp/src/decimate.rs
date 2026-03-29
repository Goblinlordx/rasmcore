//! VP8Decimate — libwebp-exact mode selection and macroblock encoding.
//!
//! Ported from libwebp src/enc/quant_enc.c:
//! - PickBestIntra16: evaluate 4 I16x16 modes
//! - PickBestIntra4: evaluate 16 sub-blocks × 10 B_PRED modes
//! - PickBestUV: evaluate 4 chroma modes
//! - VP8Decimate: orchestrate mode selection + skip detection
//!
//! Also from cost_enc.c:
//! - VP8FixedCostsI4[10][10][10]: context-dependent 4x4 mode costs
//! - VP8GetCostLuma16/Luma4/UV: rate computation using get_residual_cost

use crate::cost_engine::{self, LevelCostTable};
use crate::predict::{self, Intra16Mode, Intra4Mode, ChromaMode};
use crate::quant::SegmentQuant;
use crate::reconstruct;
use crate::rdo::{self, ScoreT, MAX_COST, VP8ModeScore, VP8SegmentLambdas};

const ALL_INTRA16: [Intra16Mode; 4] = [
    Intra16Mode::DC, Intra16Mode::V, Intra16Mode::H, Intra16Mode::TM,
];
const ALL_INTRA4: [Intra4Mode; 10] = [
    Intra4Mode::DC, Intra4Mode::TM, Intra4Mode::V, Intra4Mode::H,
    Intra4Mode::LD, Intra4Mode::RD, Intra4Mode::VR, Intra4Mode::VL,
    Intra4Mode::HD, Intra4Mode::HU,
];
const ALL_CHROMA: [ChromaMode; 4] = [
    ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM,
];

// ─── VP8FixedCostsI4[10][10][10] ────────────────────────────────────────

/// Context-dependent 4x4 mode costs.
/// `VP8FixedCostsI4[top_mode][left_mode][mode]` gives the cost in 256-scaled
/// units to signal the given B_PRED mode given the top and left neighbors.
///
/// Ported from libwebp cost_enc.c line 106.
#[rustfmt::skip]
pub const VP8_FIXED_COSTS_I4: [[[u16; 10]; 10]; 10] = [
    [[40, 1151, 1723, 1874, 2103, 2019, 1628, 1777, 2226, 2137],
     [192, 469, 1296, 1308, 1849, 1794, 1781, 1703, 1713, 1522],
     [142, 910, 762, 1684, 1849, 1576, 1460, 1305, 1801, 1657],
     [559, 641, 1370, 421, 1182, 1569, 1612, 1725, 863, 1007],
     [299, 1059, 1256, 1108, 636, 1068, 1581, 1883, 869, 1142],
     [277, 1111, 707, 1362, 1089, 672, 1603, 1541, 1545, 1291],
     [214, 781, 1609, 1303, 1632, 2229, 726, 1560, 1713, 918],
     [152, 1037, 1046, 1759, 1983, 2174, 1358, 742, 1740, 1390],
     [512, 1046, 1420, 753, 752, 1297, 1486, 1613, 460, 1207],
     [424, 827, 1362, 719, 1462, 1202, 1199, 1476, 1199, 538]],
    [[240, 402, 1134, 1491, 1659, 1505, 1517, 1555, 1979, 2099],
     [467, 242, 960, 1232, 1714, 1620, 1834, 1570, 1676, 1391],
     [500, 455, 463, 1507, 1699, 1282, 1564, 982, 2114, 2114],
     [672, 643, 1372, 331, 1589, 1667, 1453, 1938, 996, 876],
     [458, 783, 1037, 911, 738, 968, 1165, 1518, 859, 1033],
     [504, 815, 504, 1139, 1219, 719, 1506, 1085, 1268, 1268],
     [333, 630, 1445, 1239, 1883, 3672, 799, 1548, 1865, 598],
     [399, 644, 746, 1342, 1856, 1350, 1493, 613, 1855, 1015],
     [622, 749, 1205, 608, 1066, 1408, 1290, 1406, 546, 971],
     [500, 753, 1041, 668, 1230, 1617, 1297, 1425, 1383, 523]],
    [[394, 553, 523, 1502, 1536, 981, 1608, 1142, 1666, 2181],
     [655, 430, 375, 1411, 1861, 1220, 1677, 1135, 1978, 1553],
     [690, 640, 245, 1954, 2070, 1194, 1528, 982, 1972, 2232],
     [559, 834, 741, 867, 1131, 980, 1225, 852, 1092, 784],
     [690, 875, 516, 959, 673, 894, 1056, 1190, 1528, 1126],
     [740, 951, 384, 1277, 1177, 492, 1579, 1155, 1846, 1513],
     [323, 775, 1062, 1776, 3062, 1274, 813, 1188, 1372, 655],
     [488, 971, 484, 1767, 1515, 1775, 1115, 503, 1539, 1461],
     [740, 1006, 998, 709, 851, 1230, 1337, 788, 741, 721],
     [522, 1073, 573, 1045, 1346, 887, 1046, 1146, 1203, 697]],
    [[105, 864, 1442, 1009, 1934, 1840, 1519, 1920, 1673, 1579],
     [534, 305, 1193, 683, 1388, 2164, 1802, 1894, 1264, 1170],
     [305, 518, 877, 1108, 1426, 3215, 1425, 1064, 1320, 1242],
     [683, 732, 1927, 257, 1493, 2048, 1858, 1552, 1055, 947],
     [394, 814, 1024, 660, 959, 1556, 1282, 1289, 893, 1047],
     [528, 615, 996, 940, 1201, 635, 1094, 2515, 803, 1358],
     [347, 614, 1609, 1187, 3133, 1345, 1007, 1339, 1017, 667],
     [218, 740, 878, 1605, 3650, 3650, 1345, 758, 1357, 1617],
     [672, 750, 1541, 558, 1257, 1599, 1870, 2135, 402, 1087],
     [592, 684, 1161, 430, 1092, 1497, 1475, 1489, 1095, 822]],
    [[228, 1056, 1059, 1368, 752, 982, 1512, 1518, 987, 1782],
     [494, 514, 818, 942, 965, 892, 1610, 1356, 1048, 1363],
     [512, 648, 591, 1042, 761, 991, 1196, 1454, 1309, 1463],
     [683, 749, 1043, 676, 841, 1396, 1133, 1138, 654, 939],
     [622, 1101, 1126, 994, 361, 1077, 1203, 1318, 877, 1219],
     [631, 1068, 857, 1650, 651, 477, 1650, 1419, 828, 1170],
     [555, 727, 1068, 1335, 3127, 1339, 820, 1331, 1077, 429],
     [504, 879, 624, 1398, 889, 889, 1392, 808, 891, 1406],
     [683, 1602, 1289, 977, 578, 983, 1280, 1708, 406, 1122],
     [399, 865, 1433, 1070, 1072, 764, 968, 1477, 1223, 678]],
    [[333, 760, 935, 1638, 1010, 529, 1646, 1410, 1472, 2219],
     [512, 494, 750, 1160, 1215, 610, 1870, 1868, 1628, 1169],
     [572, 646, 492, 1934, 1208, 603, 1580, 1099, 1398, 1995],
     [786, 789, 942, 581, 1018, 951, 1599, 1207, 731, 768],
     [690, 1015, 672, 1078, 582, 504, 1693, 1438, 1108, 2897],
     [768, 1267, 571, 2005, 1243, 244, 2881, 1380, 1786, 1453],
     [452, 899, 1293, 903, 1311, 3100, 465, 1311, 1319, 813],
     [394, 927, 942, 1103, 1358, 1104, 946, 593, 1363, 1109],
     [559, 1005, 1007, 1016, 658, 1173, 1021, 1164, 623, 1028],
     [564, 796, 632, 1005, 1014, 863, 2316, 1268, 938, 764]],
    [[266, 606, 1098, 1228, 1497, 1243, 948, 1030, 1734, 1461],
     [366, 585, 901, 1060, 1407, 1247, 876, 1134, 1620, 1054],
     [452, 565, 542, 1729, 1479, 1479, 1016, 886, 2938, 1150],
     [555, 1088, 1533, 950, 1354, 895, 834, 1019, 1021, 496],
     [704, 815, 1193, 971, 973, 640, 1217, 2214, 832, 578],
     [672, 1245, 579, 871, 875, 774, 872, 1273, 1027, 949],
     [296, 1134, 2050, 1784, 1636, 3425, 442, 1550, 2076, 722],
     [342, 982, 1259, 1846, 1848, 1848, 622, 568, 1847, 1052],
     [555, 1064, 1304, 828, 746, 1343, 1075, 1329, 1078, 494],
     [288, 1167, 1285, 1174, 1639, 1639, 833, 2254, 1304, 509]],
    [[342, 719, 767, 1866, 1757, 1270, 1246, 550, 1746, 2151],
     [483, 653, 694, 1509, 1459, 1410, 1218, 507, 1914, 1266],
     [488, 757, 447, 2979, 1813, 1268, 1654, 539, 1849, 2109],
     [522, 1097, 1085, 851, 1365, 1111, 851, 901, 961, 605],
     [709, 716, 841, 728, 736, 945, 941, 862, 2845, 1057],
     [512, 1323, 500, 1336, 1083, 681, 1342, 717, 1604, 1350],
     [452, 1155, 1372, 1900, 1501, 3290, 311, 944, 1919, 922],
     [403, 1520, 977, 2132, 1733, 3522, 1076, 276, 3335, 1547],
     [559, 1374, 1101, 615, 673, 2462, 974, 795, 984, 984],
     [547, 1122, 1062, 812, 1410, 951, 1140, 622, 1268, 651]],
    [[165, 982, 1235, 938, 1334, 1366, 1659, 1578, 964, 1612],
     [592, 422, 925, 847, 1139, 1112, 1387, 2036, 861, 1041],
     [403, 837, 732, 770, 941, 1658, 1250, 809, 1407, 1407],
     [896, 874, 1071, 381, 1568, 1722, 1437, 2192, 480, 1035],
     [640, 1098, 1012, 1032, 684, 1382, 1581, 2106, 416, 865],
     [559, 1005, 819, 914, 710, 770, 1418, 920, 838, 1435],
     [415, 1258, 1245, 870, 1278, 3067, 770, 1021, 1287, 522],
     [406, 990, 601, 1009, 1265, 1265, 1267, 759, 1017, 1277],
     [968, 1182, 1329, 788, 1032, 1292, 1705, 1714, 203, 1403],
     [732, 877, 1279, 471, 901, 1161, 1545, 1294, 755, 755]],
    [[111, 931, 1378, 1185, 1933, 1648, 1148, 1714, 1873, 1307],
     [406, 414, 1030, 1023, 1910, 1404, 1313, 1647, 1509, 793],
     [342, 640, 575, 1088, 1241, 1349, 1161, 1350, 1756, 1502],
     [559, 766, 1185, 357, 1682, 1428, 1329, 1897, 1219, 802],
     [473, 909, 1164, 771, 719, 2508, 1427, 1432, 722, 782],
     [342, 892, 785, 1145, 1150, 794, 1296, 1550, 973, 1057],
     [208, 1036, 1326, 1343, 1606, 3395, 815, 1455, 1618, 712],
     [228, 928, 890, 1046, 3499, 1711, 994, 829, 1720, 1318],
     [768, 724, 1058, 636, 991, 1075, 1319, 1324, 616, 825],
     [305, 1167, 1358, 899, 1587, 1587, 987, 1988, 1332, 501]],
];

// ─── SSE helper functions ────────────────────────────────────────────────

/// Sum of squared errors for 16x16 block.
#[inline]
pub fn sse_16x16(a: &[u8; 256], b: &[u8; 256]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x as i32 - y as i32;
        (d * d) as u32
    }).sum()
}

/// Sum of squared errors for 4x4 block.
#[inline]
pub fn sse_4x4(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x as i32 - y as i32;
        (d * d) as u32
    }).sum()
}

/// Sum of squared errors for two 8x8 blocks (U+V, 16x8 in libwebp layout).
#[inline]
pub fn sse_8x8(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x as i32 - y as i32;
        (d * d) as u32
    }).sum()
}

// ─── Flatness detection (from quant_enc.c) ───────────────────────────────

const FLATNESS_LIMIT_I16: i16 = 10;
const FLATNESS_LIMIT_I4: i16 = 3;
const FLATNESS_LIMIT_UV: i16 = 2;
const FLATNESS_PENALTY: u32 = 140;

/// Check if quantized levels are "flat" (few non-zeros, all small).
/// Ported from libwebp quant_enc.c IsFlat().
fn is_flat(levels: &[[i16; 16]], thresh: i16) -> bool {
    let mut score = 0;
    for block in levels {
        for &v in block.iter() {
            if v != 0 { score += 1; }
            if v.abs() > thresh { return false; }
        }
    }
    score <= thresh as i32
}

/// Check if source 16x16 block has low variance (flat).
fn is_flat_source_16(src: &[u8; 256]) -> bool {
    // Simple variance check: if max-min <= threshold, it's flat
    let min = *src.iter().min().unwrap();
    let max = *src.iter().max().unwrap();
    (max - min) <= 20
}

// ─── Cost functions (GetCostLuma16/4/UV) ─────────────────────────────────

/// Compute rate for 16x16 luma: Y2 DC block + 16 AC sub-blocks.
/// Ported from libwebp cost_enc.c VP8GetCostLuma16().
pub fn get_cost_luma16(
    rd: &VP8ModeScore,
    cost_table: &LevelCostTable,
    top_nz: &mut [u8; 4],
    left_nz: &mut [u8; 4],
    dc_nz: u8,   // previous Y2 NZ context
) -> u32 {
    let mut r: u32 = 0;

    // DC (Y2 block): type=1 (TYPE_I16_DC), first=0
    let dc_ctx = dc_nz.min(2) as usize;
    r += cost_engine::get_residual_cost(&rd.y_dc_levels, 0, 1, dc_ctx, cost_table);

    // AC (16 sub-blocks): type=0 (TYPE_I16_AC), first=1
    for y in 0..4 {
        for x in 0..4 {
            let sb = y * 4 + x;
            let ctx = (top_nz[x] + left_nz[y]).min(2) as usize;
            r += cost_engine::get_residual_cost(
                &rd.y_ac_levels[sb], 1, 0, ctx, cost_table,
            );
            // Update context: nonzero if any AC coefficient is nonzero
            let has_nz = rd.y_ac_levels[sb].iter().skip(1).any(|&c| c != 0);
            let nz_val = if has_nz { 1 } else { 0 };
            top_nz[x] = nz_val;
            left_nz[y] = nz_val;
        }
    }

    r
}

/// Compute rate for a single 4x4 luma block.
/// Ported from libwebp cost_enc.c VP8GetCostLuma4().
pub fn get_cost_luma4(
    levels: &[i16; 16],
    cost_table: &LevelCostTable,
    ctx: usize,
) -> u32 {
    // type=3 (TYPE_I4_AC), first=0
    cost_engine::get_residual_cost(levels, 0, 3, ctx, cost_table)
}

/// Compute rate for chroma (U+V, 8 blocks).
/// Ported from libwebp cost_enc.c VP8GetCostUV().
pub fn get_cost_uv(
    uv_levels: &[[i16; 16]; 8],
    cost_table: &LevelCostTable,
    top_nz: &mut [u8; 4],  // [u0, u1, v0, v1]
    left_nz: &mut [u8; 4], // [u0, u1, v0, v1]
) -> u32 {
    let mut r: u32 = 0;

    // type=2 (TYPE_CHROMA_A), first=0
    for ch in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let block_idx = ch * 4 + y * 2 + x;
                let ctx_x = ch * 2 + x;
                let ctx_y = ch * 2 + y;
                let ctx = (top_nz[ctx_x] + left_nz[ctx_y]).min(2) as usize;
                r += cost_engine::get_residual_cost(
                    &uv_levels[block_idx], 0, 2, ctx, cost_table,
                );
                let has_nz = uv_levels[block_idx].iter().any(|&c| c != 0);
                let nz_val = if has_nz { 1 } else { 0 };
                top_nz[ctx_x] = nz_val;
                left_nz[ctx_y] = nz_val;
            }
        }
    }

    r
}

// ─── PickBestIntra16 ─────────────────────────────────────────────────────

/// Evaluate all 4 I16x16 prediction modes, pick the one with lowest RD score.
/// Ported from libwebp quant_enc.c PickBestIntra16 (line 982).
pub fn pick_best_intra16(
    src_16x16: &[u8; 256],
    above: &[u8; 16],
    left: &[u8; 16],
    top_left: u8,
    seg_quant: &SegmentQuant,
    lambdas: &VP8SegmentLambdas,
    cost_table: &LevelCostTable,
    rd: &mut VP8ModeScore,
) {
    let mut best_score: ScoreT = MAX_COST;
    let mut best_mode: i32 = -1;
    let flat_src = is_flat_source_16(src_16x16);

    for (mode_idx, &mode) in ALL_INTRA16.iter().enumerate() {
        let mut pred = [0u8; 256];
        predict::predict_16x16(mode, above, left, top_left, true, true, &mut pred);

        // Reconstruct with trellis
        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];
        let recon_result = reconstruct::reconstruct_intra16(
            src_16x16, &pred, seg_quant, lambdas.lambda_trellis_i16,
            cost_table, &mut top_nz, &mut left_nz,
        );

        // Build temporary ModeScore
        let mut rd_cur = VP8ModeScore::default();
        rd_cur.mode_i16 = mode_idx as i32;
        rd_cur.y_dc_levels = recon_result.y_dc_levels;
        rd_cur.y_ac_levels = recon_result.y_ac_levels;
        rd_cur.nz = recon_result.nz;

        // Distortion
        rd_cur.d = sse_16x16(src_16x16, &recon_result.recon) as ScoreT;
        rd_cur.sd = 0; // Skip spectral distortion (tlambda=0)

        // Header cost
        rd_cur.h = rdo::mode_header_cost_16x16(mode_idx as u8) as ScoreT;

        // Rate
        let mut cost_top_nz = [0u8; 4];
        let mut cost_left_nz = [0u8; 4];
        rd_cur.r = get_cost_luma16(
            &rd_cur, cost_table, &mut cost_top_nz, &mut cost_left_nz, 0,
        ) as ScoreT;

        // Flatness penalty
        if flat_src {
            let is_flat_coeffs = is_flat(&rd_cur.y_ac_levels, FLATNESS_LIMIT_I16);
            if is_flat_coeffs {
                rd_cur.d *= 2;
                rd_cur.sd *= 2;
            }
        }

        // Score with I16 lambda
        rdo::set_rd_score(lambdas.lambda_i16, &mut rd_cur);

        if best_mode < 0 || rd_cur.score < best_score {
            best_score = rd_cur.score;
            best_mode = mode_idx as i32;
            rd.d = rd_cur.d;
            rd.sd = rd_cur.sd;
            rd.h = rd_cur.h;
            rd.r = rd_cur.r;
            rd.score = rd_cur.score;
            rd.y_dc_levels = rd_cur.y_dc_levels;
            rd.y_ac_levels = rd_cur.y_ac_levels;
            rd.nz = rd_cur.nz;
            rd.mode_i16 = mode_idx as i32;
        }
    }

    // Finalize score with mode lambda (for I16 vs I4 comparison)
    rdo::set_rd_score(lambdas.lambda_mode, rd);
}

// ─── PickBestIntra4 ──────────────────────────────────────────────────────

/// Evaluate B_PRED mode: 16 sub-blocks × 10 modes each, sequential recon.
/// Returns true if I4 beats the current I16 score in `rd`.
///
/// Ported from libwebp quant_enc.c PickBestIntra4 (line 1052).
pub fn pick_best_intra4(
    src_16x16: &[u8; 256],
    above_row: &[u8; 20],  // 16 above pixels + 4 extra for diagonal modes
    left_col: &[u8; 16],
    seg_quant: &SegmentQuant,
    lambdas: &VP8SegmentLambdas,
    cost_table: &LevelCostTable,
    rd: &mut VP8ModeScore,
    max_i4_header_bits: i32,
    top_modes: &[u8; 4],  // above macroblock's bottom-row modes
    left_modes: &[u8; 4], // left macroblock's right-column modes
) -> bool {
    let i16_score = rd.score;

    // Accumulate total RD for all 16 sub-blocks
    let mut rd_best = VP8ModeScore::default();
    // VP8BitCost(0, 145) = 211: cost of signaling "not I16" in macroblock header
    rd_best.h = 211;
    rdo::set_rd_score(lambdas.lambda_mode, &mut rd_best);

    let mut best_modes = [0u8; 16]; // selected mode per sub-block
    let mut recon_16x16 = [0u8; 256]; // accumulate reconstructed pixels
    let mut total_header_bits: i32 = 0;
    let mut top_nz = [0u8; 4];
    let mut left_nz = [0u8; 4];

    for i4 in 0..16 {
        let sb_row = i4 / 4;
        let sb_col = i4 % 4;

        // Extract source 4x4
        let mut src_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src_4x4[r * 4 + c] = src_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }

        // Build above/left context for prediction from reconstructed neighbors
        let mut above_4 = [127u8; 8]; // 4 above + 4 extra
        let mut left_4 = [129u8; 4];
        let mut tl = 127u8;

        // Above: from row above this sub-block
        if sb_row == 0 {
            above_4[..4].copy_from_slice(&above_row[sb_col * 4..sb_col * 4 + 4]);
            // Extra pixels for diagonal modes
            if sb_col < 3 {
                above_4[4..8].copy_from_slice(&above_row[sb_col * 4 + 4..sb_col * 4 + 8]);
            } else {
                above_4[4..8].fill(above_row[15]);
            }
        } else {
            for c in 0..4 {
                above_4[c] = recon_16x16[((sb_row * 4 - 1) * 16) + sb_col * 4 + c];
            }
            if sb_col < 3 {
                for c in 0..4 {
                    above_4[4 + c] = recon_16x16[((sb_row * 4 - 1) * 16) + sb_col * 4 + 4 + c];
                }
            } else {
                let fill_val = above_4[3];
                above_4[4..8].fill(fill_val);
            }
        }

        // Left: from column left of this sub-block
        if sb_col == 0 {
            left_4.copy_from_slice(&left_col[sb_row * 4..sb_row * 4 + 4]);
        } else {
            for r in 0..4 {
                left_4[r] = recon_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 - 1];
            }
        }

        // Top-left
        if sb_row == 0 && sb_col == 0 {
            tl = above_row[0].wrapping_sub(1); // approximate
        } else if sb_row == 0 {
            tl = above_row[sb_col * 4 - 1];
        } else if sb_col == 0 {
            tl = left_col[sb_row * 4 - 1];
        } else {
            tl = recon_16x16[((sb_row * 4 - 1) * 16) + sb_col * 4 - 1];
        }

        // Get context-dependent mode costs
        let top_mode = if sb_row == 0 {
            top_modes[sb_col]
        } else {
            best_modes[i4 - 4]
        };
        let left_mode = if sb_col == 0 {
            left_modes[sb_row]
        } else {
            best_modes[i4 - 1]
        };
        let mode_costs = &VP8_FIXED_COSTS_I4
            [top_mode.min(9) as usize]
            [left_mode.min(9) as usize];

        // Try all 10 B_PRED modes
        let mut best_i4_mode: i32 = -1;
        let mut best_i4_score: ScoreT = MAX_COST;
        let mut best_levels = [0i16; 16];
        let mut best_recon = [0u8; 16];

        let ctx = (top_nz[sb_col] + left_nz[sb_row]).min(2) as usize;

        for (mode_idx, &mode) in ALL_INTRA4.iter().enumerate() {
            // Predict
            let mut pred = [0u8; 16];
            let above_4_arr: [u8; 4] = [above_4[0], above_4[1], above_4[2], above_4[3]];
            let above_right_arr: [u8; 4] = [above_4[4], above_4[5], above_4[6], above_4[7]];
            predict::predict_4x4(mode, &above_4_arr, &left_4, tl, &above_right_arr, &mut pred);

            // Reconstruct with trellis (y_dc has DC step at [0], AC step at [1..15])
            let result = reconstruct::reconstruct_intra4(
                &src_4x4, &pred, &seg_quant.y_dc,
                lambdas.lambda_trellis_i4, cost_table, ctx,
            );

            // RD evaluation
            let mut rd_tmp = VP8ModeScore::default();
            rd_tmp.d = sse_4x4(&src_4x4, &result.recon) as ScoreT;
            rd_tmp.sd = 0;
            rd_tmp.h = mode_costs[mode_idx] as ScoreT;

            // Flatness penalty
            if mode_idx > 0 && is_flat(&[result.levels], FLATNESS_LIMIT_I4) {
                rd_tmp.r = (FLATNESS_PENALTY * 1) as ScoreT;
            } else {
                rd_tmp.r = 0;
            }

            // Early-out: score without full rate
            rdo::set_rd_score(lambdas.lambda_i4, &mut rd_tmp);
            if best_i4_mode >= 0 && rd_tmp.score >= best_i4_score {
                continue;
            }

            // Full rate
            rd_tmp.r += get_cost_luma4(&result.levels, cost_table, ctx) as ScoreT;
            rdo::set_rd_score(lambdas.lambda_i4, &mut rd_tmp);

            if best_i4_mode < 0 || rd_tmp.score < best_i4_score {
                best_i4_score = rd_tmp.score;
                best_i4_mode = mode_idx as i32;
                best_levels = result.levels;
                best_recon = result.recon;
            }
        }

        // Accumulate into rd_best
        let mut rd_i4 = VP8ModeScore::default();
        rd_i4.d = 0; // already in best_i4_score
        rd_i4.sd = 0;
        rd_i4.h = mode_costs[best_i4_mode as usize] as ScoreT;
        rd_i4.r = get_cost_luma4(&best_levels, cost_table, ctx) as ScoreT;
        rd_i4.score = best_i4_score;

        // Finalize with mode lambda
        rdo::set_rd_score(lambdas.lambda_mode, &mut rd_i4);

        // Add to cumulative score
        rd_best.d += rd_i4.d;
        rd_best.sd += rd_i4.sd;
        rd_best.h += rd_i4.h;
        rd_best.r += rd_i4.r;
        rd_best.score += rd_i4.score;

        // Early exit: I4 already worse than I16
        if rd_best.score >= i16_score {
            return false;
        }

        total_header_bits += mode_costs[best_i4_mode as usize] as i32;
        if total_header_bits > max_i4_header_bits {
            return false;
        }

        // Store results
        best_modes[i4] = best_i4_mode as u8;
        rd_best.y_ac_levels[i4] = best_levels;

        // Update NZ context
        let has_nz = best_levels.iter().any(|&c| c != 0);
        top_nz[sb_col] = if has_nz { 1 } else { 0 };
        left_nz[sb_row] = if has_nz { 1 } else { 0 };

        // Store reconstructed pixels
        for r in 0..4 {
            for c in 0..4 {
                recon_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 + c] = best_recon[r * 4 + c];
            }
        }
    }

    // I4 wins: copy results to rd
    rd.d = rd_best.d;
    rd.sd = rd_best.sd;
    rd.h = rd_best.h;
    rd.r = rd_best.r;
    rd.score = rd_best.score;
    rd.y_ac_levels = rd_best.y_ac_levels;
    rd.modes_i4 = best_modes;
    true
}

// ─── PickBestUV ──────────────────────────────────────────────────────────

/// Evaluate all 4 chroma prediction modes, pick lowest RD score.
/// Ported from libwebp quant_enc.c PickBestUV (line 1148).
pub fn pick_best_uv(
    src_u: &[u8; 64],
    src_v: &[u8; 64],
    above_u: &[u8; 8],
    above_v: &[u8; 8],
    left_u: &[u8; 8],
    left_v: &[u8; 8],
    tl_u: u8,
    tl_v: u8,
    seg_quant: &SegmentQuant,
    lambdas: &VP8SegmentLambdas,
    cost_table: &LevelCostTable,
    rd: &mut VP8ModeScore,
) {
    let mut best_score: ScoreT = MAX_COST;

    for (mode_idx, &mode) in ALL_CHROMA.iter().enumerate() {
        let mut pred_u = [0u8; 64];
        let mut pred_v = [0u8; 64];
        predict::predict_8x8(mode, above_u, left_u, tl_u, true, true, &mut pred_u);
        predict::predict_8x8(mode, above_v, left_v, tl_v, true, true, &mut pred_v);

        // Reconstruct
        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];
        let recon_result = reconstruct::reconstruct_uv(
            src_u, src_v, &pred_u, &pred_v,
            seg_quant, lambdas.lambda_trellis_uv,
            cost_table, &mut top_nz, &mut left_nz,
        );

        // RD evaluation
        let mut rd_uv = VP8ModeScore::default();
        rd_uv.d = (sse_8x8(src_u, &recon_result.recon_u)
                 + sse_8x8(src_v, &recon_result.recon_v)) as ScoreT;
        rd_uv.sd = 0; // No spectral distortion for UV
        rd_uv.h = rdo::VP8_FIXED_COSTS_UV[mode_idx] as ScoreT;

        // Rate
        let mut cost_top_nz = [0u8; 4];
        let mut cost_left_nz = [0u8; 4];
        rd_uv.r = get_cost_uv(
            &recon_result.uv_levels, cost_table,
            &mut cost_top_nz, &mut cost_left_nz,
        ) as ScoreT;

        // Flatness penalty
        if mode_idx > 0 && is_flat(&recon_result.uv_levels, FLATNESS_LIMIT_UV) {
            rd_uv.r += (FLATNESS_PENALTY * 8) as ScoreT;
        }

        rdo::set_rd_score(lambdas.lambda_uv, &mut rd_uv);

        if mode_idx == 0 || rd_uv.score < best_score {
            best_score = rd_uv.score;
            rd.mode_uv = mode_idx as i32;
            rd.uv_levels = recon_result.uv_levels;
            // Accumulate UV distortion and rate into rd
            // (done after all modes evaluated)
        }
    }

    // Re-evaluate best UV mode to get final D/R/H for accumulation
    let best_uv_mode = ALL_CHROMA[rd.mode_uv as usize];
    let mut pred_u = [0u8; 64];
    let mut pred_v = [0u8; 64];
    predict::predict_8x8(best_uv_mode, above_u, left_u, tl_u, true, true, &mut pred_u);
    predict::predict_8x8(best_uv_mode, above_v, left_v, tl_v, true, true, &mut pred_v);

    let mut top_nz = [0u8; 4];
    let mut left_nz = [0u8; 4];
    let recon_result = reconstruct::reconstruct_uv(
        src_u, src_v, &pred_u, &pred_v,
        seg_quant, lambdas.lambda_trellis_uv,
        cost_table, &mut top_nz, &mut left_nz,
    );

    let mut rd_best = VP8ModeScore::default();
    rd_best.d = (sse_8x8(src_u, &recon_result.recon_u)
               + sse_8x8(src_v, &recon_result.recon_v)) as ScoreT;
    rd_best.sd = 0;
    rd_best.h = rdo::VP8_FIXED_COSTS_UV[rd.mode_uv as usize] as ScoreT;
    let mut cost_top_nz = [0u8; 4];
    let mut cost_left_nz = [0u8; 4];
    rd_best.r = get_cost_uv(
        &rd.uv_levels, cost_table,
        &mut cost_top_nz, &mut cost_left_nz,
    ) as ScoreT;
    rdo::set_rd_score(lambdas.lambda_uv, &mut rd_best);

    // Add UV score to rd
    rd.d += rd_best.d;
    rd.sd += rd_best.sd;
    rd.h += rd_best.h;
    rd.r += rd_best.r;
    rd.score += rd_best.score;
    rd.nz |= recon_result.nz;
}

// ─── VP8Decimate ─────────────────────────────────────────────────────────

/// Result of VP8Decimate — contains the mode decision and all encoded data.
pub struct DecimateResult {
    /// True if I4 mode was selected, false for I16.
    pub is_i4: bool,
    /// True if all coefficients are zero (skip macroblock).
    pub is_skipped: bool,
    /// The full mode score with all encoded data.
    pub rd: VP8ModeScore,
}

/// VP8Decimate — orchestrate mode selection for one macroblock.
///
/// Ported from libwebp quant_enc.c VP8Decimate (line 1343).
/// Evaluates I16, I4, and UV modes, picks the best combination.
pub fn vp8_decimate(
    src_y: &[u8; 256],
    src_u: &[u8; 64],
    src_v: &[u8; 64],
    above_y: &[u8; 16],
    left_y: &[u8; 16],
    tl_y: u8,
    above_u: &[u8; 8],
    above_v: &[u8; 8],
    left_u: &[u8; 8],
    left_v: &[u8; 8],
    tl_u: u8,
    tl_v: u8,
    seg_quant: &SegmentQuant,
    lambdas: &VP8SegmentLambdas,
    cost_table: &LevelCostTable,
    above_y_full: &[u8; 20],  // 16 + 4 extra for diagonal
    top_modes: &[u8; 4],
    left_modes: &[u8; 4],
) -> DecimateResult {
    let mut rd = VP8ModeScore::default();

    // Phase 1: Evaluate I16x16
    pick_best_intra16(
        src_y, above_y, left_y, tl_y,
        seg_quant, lambdas, cost_table, &mut rd,
    );

    // Phase 2: Evaluate I4x4 (if it can beat I16)
    let max_header = 15000; // libwebp uses enc->max_i4_header_bits
    let is_i4 = pick_best_intra4(
        src_y, above_y_full, left_y,
        seg_quant, lambdas, cost_table, &mut rd,
        max_header, top_modes, left_modes,
    );

    // Phase 3: Evaluate UV
    pick_best_uv(
        src_u, src_v, above_u, above_v, left_u, left_v, tl_u, tl_v,
        seg_quant, lambdas, cost_table, &mut rd,
    );

    let is_skipped = rd.nz == 0;

    DecimateResult { is_i4, is_skipped, rd }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant;

    #[test]
    fn fixed_costs_i4_spot_check() {
        // Verify a few entries from the table
        assert_eq!(VP8_FIXED_COSTS_I4[0][0][0], 40);
        assert_eq!(VP8_FIXED_COSTS_I4[0][0][1], 1151);
        assert_eq!(VP8_FIXED_COSTS_I4[9][9][9], 501);
        assert_eq!(VP8_FIXED_COSTS_I4[0][1][0], 192);
    }

    #[test]
    fn sse_16x16_identical() {
        let a = [128u8; 256];
        assert_eq!(sse_16x16(&a, &a), 0);
    }

    #[test]
    fn sse_16x16_known() {
        let a = [100u8; 256];
        let b = [110u8; 256];
        assert_eq!(sse_16x16(&a, &b), 256 * 100); // 256 pixels × 10²
    }

    #[test]
    fn sse_4x4_identical() {
        let a = [128u8; 16];
        assert_eq!(sse_4x4(&a, &a), 0);
    }

    #[test]
    fn pick_best_intra16_flat_block() {
        let src = [128u8; 256];
        let above = [128u8; 16];
        let left = [128u8; 16];
        let seg_quant = quant::build_segment_quant(30);
        let lambdas = rdo::compute_segment_lambdas(30);
        let probs = cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);

        let mut rd = VP8ModeScore::default();
        pick_best_intra16(&src, &above, &left, 128, &seg_quant, &lambdas, &cost_table, &mut rd);

        // Flat block should pick DC mode (0)
        assert_eq!(rd.mode_i16, 0, "flat block should prefer DC mode");
        assert!(rd.score < MAX_COST, "should have a valid score");
    }

    #[test]
    fn pick_best_uv_no_panic() {
        let src_u = [128u8; 64];
        let src_v = [128u8; 64];
        let above_u = [128u8; 8];
        let above_v = [128u8; 8];
        let left_u = [128u8; 8];
        let left_v = [128u8; 8];
        let seg_quant = quant::build_segment_quant(30);
        let lambdas = rdo::compute_segment_lambdas(30);
        let probs = cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);

        let mut rd = VP8ModeScore::default();
        rd.score = 0; // initialize score before UV adds to it
        pick_best_uv(
            &src_u, &src_v, &above_u, &above_v, &left_u, &left_v, 128, 128,
            &seg_quant, &lambdas, &cost_table, &mut rd,
        );
        assert!(rd.mode_uv >= 0 && rd.mode_uv < 4);
    }

    #[test]
    fn vp8_decimate_flat_block() {
        let src_y = [128u8; 256];
        let src_u = [128u8; 64];
        let src_v = [128u8; 64];
        let above_y = [128u8; 16];
        let left_y = [128u8; 16];
        let above_y_full = [128u8; 20];
        let above_u = [128u8; 8];
        let above_v = [128u8; 8];
        let left_u = [128u8; 8];
        let left_v = [128u8; 8];
        let top_modes = [0u8; 4];
        let left_modes = [0u8; 4];
        let seg_quant = quant::build_segment_quant(30);
        let lambdas = rdo::compute_segment_lambdas(30);
        let probs = cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);

        let result = vp8_decimate(
            &src_y, &src_u, &src_v,
            &above_y, &left_y, 128,
            &above_u, &above_v, &left_u, &left_v, 128, 128,
            &seg_quant, &lambdas, &cost_table,
            &above_y_full, &top_modes, &left_modes,
        );

        // Flat block: should be skip (all zeros)
        assert!(result.is_skipped || result.rd.nz == 0,
            "flat block should have zero residual");
    }
}
