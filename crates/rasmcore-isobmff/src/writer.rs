//! ISOBMFF container writer for HEIC still images.
//!
//! Produces a valid HEIC file from HEVC bitstream bytes and metadata.
//! Validated by roundtrip: `assemble_heic()` → `parse()` → verify all fields.

use crate::boxwriter::{write_box, write_full_box};
use crate::types::{ColorInfo, HevcConfig};

/// Input for assembling a HEIC container.
#[derive(Debug, Clone)]
pub struct HeicInput {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw HEVC bitstream (IDR slice data, without parameter sets).
    pub bitstream: Vec<u8>,
    /// HEVC decoder configuration (VPS/SPS/PPS extracted from bitstream).
    pub hevc_config: HevcConfig,
    /// Optional color information.
    pub color: Option<ColorInfo>,
}

/// Input for a single tile in a grid HEIC.
#[derive(Debug, Clone)]
pub struct TileInput {
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
    /// Raw HEVC bitstream for this tile.
    pub bitstream: Vec<u8>,
}

/// Input for assembling a grid HEIC container.
#[derive(Debug, Clone)]
pub struct HeicGridInput {
    /// Output image width.
    pub output_width: u32,
    /// Output image height.
    pub output_height: u32,
    /// Number of tile rows.
    pub rows: u8,
    /// Number of tile columns.
    pub cols: u8,
    /// Tile data (must have rows × cols entries).
    pub tiles: Vec<TileInput>,
    /// Shared HEVC config for all tiles.
    pub hevc_config: HevcConfig,
    /// Optional color information.
    pub color: Option<ColorInfo>,
}

/// Assemble a minimal HEIC container (single image, no grid).
///
/// Layout: `ftyp | meta | mdat`
///
/// The meta box contains hdlr, pitm, iinf, iloc, and iprp (with ispe, hvcC, and
/// optional colr). The iloc references the mdat payload.
pub fn assemble_heic(input: &HeicInput) -> Vec<u8> {
    let ftyp = write_ftyp();

    // Item ID 1 = primary image
    let item_id: u16 = 1;

    // Build property boxes
    let ispe = write_ispe(input.width, input.height);
    let hvcc = write_hvcc(&input.hevc_config);
    let mut ipco_content = Vec::new();
    ipco_content.extend(&ispe);
    ipco_content.extend(&hvcc);

    let mut prop_index: u8 = 2; // ispe=1, hvcC=2
    if let Some(ref color) = input.color {
        ipco_content.extend(&write_colr(color));
        prop_index = 3;
    }

    let ipco = write_box(b"ipco", &ipco_content);
    let ipma = write_ipma_single(item_id, prop_index, input.color.is_some());
    let mut iprp_content = Vec::new();
    iprp_content.extend(&ipco);
    iprp_content.extend(&ipma);
    let iprp = write_box(b"iprp", &iprp_content);

    // Build meta children (without iloc — size depends on iloc which depends on meta size)
    let hdlr = write_hdlr();
    let pitm = write_pitm(item_id);
    let iinf = write_iinf_single(item_id, b"hvc1");

    // Build iloc with placeholder base_offset
    let iloc = write_iloc_single(item_id, 0, input.bitstream.len() as u32);

    // Assemble meta
    let mut meta_content = Vec::new();
    meta_content.extend(&hdlr);
    meta_content.extend(&pitm);
    meta_content.extend(&iinf);
    meta_content.extend(&iloc);
    meta_content.extend(&iprp);
    let meta = write_full_box(b"meta", 0, 0, &meta_content);

    // mdat = 8-byte header + bitstream
    let mdat_header_size = 8u32;
    let mdat_content_offset = ftyp.len() + meta.len() + mdat_header_size as usize;

    // Patch iloc base_offset
    let mut meta_patched = meta.clone();
    patch_iloc_base_offset(&mut meta_patched, mdat_content_offset as u32);

    // Build mdat with explicit size (not extending to EOF, so the file is well-formed)
    let mdat = write_box(b"mdat", &input.bitstream);

    let mut output = Vec::with_capacity(ftyp.len() + meta_patched.len() + mdat.len());
    output.extend(&ftyp);
    output.extend(&meta_patched);
    output.extend(&mdat);
    output
}

/// Assemble a grid HEIC container (multi-tile).
///
/// Layout: `ftyp | meta | mdat`
///
/// Item 1 = grid descriptor (type "grid"), items 2..N+1 = tiles (type "hvc1").
/// iref dimg references link grid → tiles.
pub fn assemble_heic_grid(input: &HeicGridInput) -> Vec<u8> {
    let ftyp = write_ftyp();
    let n_tiles = input.tiles.len() as u16;
    let grid_item_id: u16 = 1;
    // Tile item IDs: 2, 3, ..., n_tiles+1
    let tile_ids: Vec<u16> = (2..2 + n_tiles).collect();

    // Grid descriptor (stored in mdat for the grid item)
    let grid_desc = write_grid_descriptor(
        input.rows,
        input.cols,
        input.output_width,
        input.output_height,
    );

    // Build properties — shared hvcC + ispe per tile + ispe for grid output
    let ispe_grid = write_ispe(input.output_width, input.output_height);
    let ispe_tile = if !input.tiles.is_empty() {
        write_ispe(input.tiles[0].width, input.tiles[0].height)
    } else {
        write_ispe(0, 0)
    };
    let hvcc = write_hvcc(&input.hevc_config);

    let mut ipco_content = Vec::new();
    ipco_content.extend(&ispe_grid); // property 1
    ipco_content.extend(&ispe_tile); // property 2
    ipco_content.extend(&hvcc); // property 3

    let mut colr_idx: Option<u8> = None;
    if let Some(ref color) = input.color {
        let colr = write_colr(color);
        ipco_content.extend(&colr); // property 4
        colr_idx = Some(4);
    }

    let ipco = write_box(b"ipco", &ipco_content);

    // ipma: grid gets ispe_grid(1) + optional colr(4)
    //        tiles get ispe_tile(2) + hvcC(3)
    let ipma = write_ipma_grid(grid_item_id, &tile_ids, colr_idx);
    let mut iprp_content = Vec::new();
    iprp_content.extend(&ipco);
    iprp_content.extend(&ipma);
    let iprp = write_box(b"iprp", &iprp_content);

    let hdlr = write_hdlr();
    let pitm = write_pitm(grid_item_id);

    // iinf: grid item + tile items
    let iinf = write_iinf_grid(grid_item_id, &tile_ids);

    // iref: grid → tiles (dimg)
    let iref = write_iref_dimg(grid_item_id, &tile_ids);

    // iloc: grid descriptor extent + tile extents (all placeholder offsets)
    let mut all_extents: Vec<(u16, u32)> = Vec::new(); // (item_id, length)
    all_extents.push((grid_item_id, grid_desc.len() as u32));
    for (i, tile) in input.tiles.iter().enumerate() {
        all_extents.push((tile_ids[i], tile.bitstream.len() as u32));
    }
    let iloc = write_iloc_multi(&all_extents);

    // Assemble meta
    let mut meta_content = Vec::new();
    meta_content.extend(&hdlr);
    meta_content.extend(&pitm);
    meta_content.extend(&iinf);
    meta_content.extend(&iloc);
    meta_content.extend(&iref);
    meta_content.extend(&iprp);
    let meta = write_full_box(b"meta", 0, 0, &meta_content);

    // Build mdat: grid_desc + tile bitstreams concatenated
    let mut mdat_payload = Vec::new();
    mdat_payload.extend(&grid_desc);
    for tile in &input.tiles {
        mdat_payload.extend(&tile.bitstream);
    }
    let mdat = write_box(b"mdat", &mdat_payload);

    let mdat_content_offset = (ftyp.len() + meta.len() + 8) as u32; // 8 = mdat header

    // Patch iloc offsets — items are contiguous in mdat
    let mut meta_patched = meta.clone();
    patch_iloc_multi_offsets(&mut meta_patched, mdat_content_offset, &all_extents);

    let mut output = Vec::with_capacity(ftyp.len() + meta_patched.len() + mdat.len());
    output.extend(&ftyp);
    output.extend(&meta_patched);
    output.extend(&mdat);
    output
}

// ─── Box Writers ─────────────────────────────────────────────────────────────

/// Write ftyp box for HEIC: major brand = heic, compatible = [mif1, heic].
fn write_ftyp() -> Vec<u8> {
    let mut content = Vec::with_capacity(16);
    content.extend_from_slice(b"heic"); // major brand
    content.extend_from_slice(&0u32.to_be_bytes()); // minor version
    content.extend_from_slice(b"mif1"); // compatible brand 1
    content.extend_from_slice(b"heic"); // compatible brand 2
    write_box(b"ftyp", &content)
}

/// Write hdlr box: handler type = pict (picture).
fn write_hdlr() -> Vec<u8> {
    let mut content = Vec::with_capacity(25);
    content.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    content.extend_from_slice(b"pict"); // handler_type
    content.extend_from_slice(&[0u8; 12]); // reserved
    content.push(0); // name (null-terminated empty string)
    write_full_box(b"hdlr", 0, 0, &content)
}

/// Write pitm box: primary item ID.
fn write_pitm(item_id: u16) -> Vec<u8> {
    write_full_box(b"pitm", 0, 0, &item_id.to_be_bytes())
}

/// Write iinf box with a single item entry.
fn write_iinf_single(item_id: u16, item_type: &[u8; 4]) -> Vec<u8> {
    let infe = write_infe(item_id, item_type);
    let mut content = Vec::new();
    content.extend_from_slice(&1u16.to_be_bytes()); // entry_count
    content.extend(&infe);
    write_full_box(b"iinf", 0, 0, &content)
}

/// Write iinf box with grid + tile items.
fn write_iinf_grid(grid_id: u16, tile_ids: &[u16]) -> Vec<u8> {
    let count = 1 + tile_ids.len() as u16;
    let mut content = Vec::new();
    content.extend_from_slice(&count.to_be_bytes());
    content.extend(&write_infe(grid_id, b"grid"));
    for &tid in tile_ids {
        content.extend(&write_infe(tid, b"hvc1"));
    }
    write_full_box(b"iinf", 0, 0, &content)
}

/// Write infe (item info entry) box — version 2 format.
fn write_infe(item_id: u16, item_type: &[u8; 4]) -> Vec<u8> {
    let mut content = Vec::with_capacity(9);
    content.extend_from_slice(&item_id.to_be_bytes()); // item_id
    content.extend_from_slice(&0u16.to_be_bytes()); // item_protection_index
    content.extend_from_slice(item_type); // item_type
    content.push(0); // item_name (null-terminated empty)
    write_full_box(b"infe", 2, 0, &content)
}

/// Write iloc box for a single item with placeholder base_offset.
///
/// Format: offset_size=4, length_size=4, base_offset_size=4, version=0.
fn write_iloc_single(item_id: u16, base_offset: u32, extent_length: u32) -> Vec<u8> {
    let mut content = Vec::with_capacity(22);
    content.push(0x44); // offset_size=4, length_size=4
    content.push(0x40); // base_offset_size=4, index_size=0 (unused in v0)
    content.extend_from_slice(&1u16.to_be_bytes()); // item_count
    content.extend_from_slice(&item_id.to_be_bytes()); // item_id
    content.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index
    content.extend_from_slice(&base_offset.to_be_bytes()); // base_offset
    content.extend_from_slice(&1u16.to_be_bytes()); // extent_count
    content.extend_from_slice(&0u32.to_be_bytes()); // extent_offset
    content.extend_from_slice(&extent_length.to_be_bytes()); // extent_length
    write_full_box(b"iloc", 0, 0, &content)
}

/// Write iloc box for multiple items (grid + tiles), all with placeholder offsets.
fn write_iloc_multi(items: &[(u16, u32)]) -> Vec<u8> {
    let mut content = Vec::new();
    content.push(0x44); // offset_size=4, length_size=4
    content.push(0x40); // base_offset_size=4
    content.extend_from_slice(&(items.len() as u16).to_be_bytes());
    for &(item_id, length) in items {
        content.extend_from_slice(&item_id.to_be_bytes());
        content.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index
        content.extend_from_slice(&0u32.to_be_bytes()); // base_offset (placeholder)
        content.extend_from_slice(&1u16.to_be_bytes()); // extent_count
        content.extend_from_slice(&0u32.to_be_bytes()); // extent_offset
        content.extend_from_slice(&length.to_be_bytes()); // extent_length
    }
    write_full_box(b"iloc", 0, 0, &content)
}

/// Write ispe (image spatial extents) property box.
fn write_ispe(width: u32, height: u32) -> Vec<u8> {
    let mut content = Vec::with_capacity(8);
    content.extend_from_slice(&width.to_be_bytes());
    content.extend_from_slice(&height.to_be_bytes());
    write_full_box(b"ispe", 0, 0, &content)
}

/// Write hvcC (HEVC decoder configuration record) property box.
fn write_hvcc(config: &HevcConfig) -> Vec<u8> {
    let mut content = Vec::new();

    // Fixed header (23 bytes)
    content.push(config.configuration_version);
    let byte1 = (config.general_profile_space << 6)
        | (if config.general_tier_flag { 0x20 } else { 0 })
        | (config.general_profile_idc & 0x1F);
    content.push(byte1);
    // general_profile_compatibility_flags (4 bytes) — write zeros (simplified)
    content.extend_from_slice(&[0u8; 4]);
    // general_constraint_indicator_flags (6 bytes) — write zeros
    content.extend_from_slice(&[0u8; 6]);
    content.push(config.general_level_idc);
    // min_spatial_segmentation_idc (2 bytes, with 0xF000 mask)
    content.extend_from_slice(&[0xF0, 0x00]);
    // parallelism_type (1 byte, with 0xFC mask)
    content.push(0xFC);
    // chroma_format (1 byte, with 0xFC mask)
    content.push(0xFC | (config.chroma_format_idc & 0x03));
    // bit_depth_luma (1 byte, with 0xF8 mask)
    content.push(0xF8 | ((config.bit_depth_luma - 8) & 0x07));
    // bit_depth_chroma (1 byte, with 0xF8 mask)
    content.push(0xF8 | ((config.bit_depth_chroma - 8) & 0x07));
    // avg_frame_rate (2 bytes)
    content.extend_from_slice(&0u16.to_be_bytes());
    // constant_frame_rate(2) + num_temporal_layers(3) + temporal_id_nested(1) + length_size_minus1(2)
    // = 0x0F for length_size=4 (0x03 in low 2 bits), rest 0
    content.push(0x0F); // lengthSizeMinusOne = 3 → 4-byte NAL length prefix
    // num_of_arrays
    content.push(config.nal_arrays.len() as u8);

    // NAL unit arrays
    for array in &config.nal_arrays {
        let array_byte = if array.completeness { 0x80 } else { 0x00 } | (array.nal_type & 0x3F);
        content.push(array_byte);
        content.extend_from_slice(&(array.nal_units.len() as u16).to_be_bytes());
        for nal in &array.nal_units {
            content.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            content.extend(nal);
        }
    }

    write_box(b"hvcC", &content)
}

/// Write colr (color information) property box.
fn write_colr(color: &ColorInfo) -> Vec<u8> {
    match color {
        ColorInfo::Nclx {
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        } => {
            let mut content = Vec::with_capacity(11);
            content.extend_from_slice(b"nclx");
            content.extend_from_slice(&colour_primaries.to_be_bytes());
            content.extend_from_slice(&transfer_characteristics.to_be_bytes());
            content.extend_from_slice(&matrix_coefficients.to_be_bytes());
            content.push(if *full_range { 0x80 } else { 0x00 });
            write_box(b"colr", &content)
        }
        ColorInfo::Icc(profile) => {
            let mut content = Vec::with_capacity(4 + profile.len());
            content.extend_from_slice(b"prof");
            content.extend(profile);
            write_box(b"colr", &content)
        }
    }
}

/// Write ipma for a single image item.
///
/// Associates properties 1..prop_count to item_id. All marked essential.
fn write_ipma_single(item_id: u16, prop_count: u8, has_color: bool) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    content.extend_from_slice(&item_id.to_be_bytes()); // item_id
    content.push(prop_count); // association_count
    // ispe (index 1, essential)
    content.push(0x81); // essential=1, property_index=1
    // hvcC (index 2, essential)
    content.push(0x82);
    if has_color {
        // colr (index 3, essential)
        content.push(0x83);
    }
    write_full_box(b"ipma", 0, 0, &content)
}

/// Write ipma for grid (item 1) + tiles.
fn write_ipma_grid(grid_id: u16, tile_ids: &[u16], colr_idx: Option<u8>) -> Vec<u8> {
    let entry_count = 1 + tile_ids.len() as u32;
    let mut content = Vec::new();
    content.extend_from_slice(&entry_count.to_be_bytes());

    // Grid item: ispe_grid(1) + optional colr
    content.extend_from_slice(&grid_id.to_be_bytes());
    let n_assoc = if colr_idx.is_some() { 2u8 } else { 1u8 };
    content.push(n_assoc);
    content.push(0x81); // ispe_grid, essential
    if let Some(idx) = colr_idx {
        content.push(0x80 | idx); // colr, essential
    }

    // Tile items: ispe_tile(2) + hvcC(3)
    for &tid in tile_ids {
        content.extend_from_slice(&tid.to_be_bytes());
        content.push(2); // 2 associations
        content.push(0x82); // ispe_tile, essential
        content.push(0x83); // hvcC, essential
    }

    write_full_box(b"ipma", 0, 0, &content)
}

/// Write iref box with dimg references (grid → tiles).
fn write_iref_dimg(from_id: u16, to_ids: &[u16]) -> Vec<u8> {
    // SingleItemTypeReferenceBox: box_type=dimg, from_item_id, ref_count, to_item_ids
    let ref_box_size = (8 + 2 + 2 + to_ids.len() * 2) as u32;
    let mut ref_content = Vec::new();
    ref_content.extend_from_slice(&ref_box_size.to_be_bytes());
    ref_content.extend_from_slice(b"dimg");
    ref_content.extend_from_slice(&from_id.to_be_bytes());
    ref_content.extend_from_slice(&(to_ids.len() as u16).to_be_bytes());
    for &tid in to_ids {
        ref_content.extend_from_slice(&tid.to_be_bytes());
    }

    write_full_box(b"iref", 0, 0, &ref_content)
}

/// Write grid descriptor for a grid image item.
fn write_grid_descriptor(rows: u8, cols: u8, width: u32, height: u32) -> Vec<u8> {
    let mut desc = Vec::with_capacity(12);
    desc.push(0); // version
    let use_32bit = width > 0xFFFF || height > 0xFFFF;
    desc.push(if use_32bit { 1 } else { 0 }); // flags
    desc.push(rows - 1); // rows_minus_one
    desc.push(cols - 1); // columns_minus_one
    if use_32bit {
        desc.extend_from_slice(&width.to_be_bytes());
        desc.extend_from_slice(&height.to_be_bytes());
    } else {
        desc.extend_from_slice(&(width as u16).to_be_bytes());
        desc.extend_from_slice(&(height as u16).to_be_bytes());
    }
    desc
}

// ─── Offset Patching ─────────────────────────────────────────────────────────

/// Patch the iloc base_offset for a single-item HEIC.
///
/// Searches for the iloc box within the meta box and writes the base_offset.
fn patch_iloc_base_offset(meta: &mut [u8], mdat_content_offset: u32) {
    // Find iloc within meta: meta full-box header = 12 bytes, then children
    if let Some(iloc_offset) = find_box_in_meta(meta, b"iloc") {
        // iloc full-box header: 12 bytes
        // content: [0x44, 0x40, count(2), item_id(2), data_ref(2), BASE_OFFSET(4), ...]
        let base_offset_pos = iloc_offset + 12 + 8; // 12=fullbox header, 8=to base_offset field
        let bytes = mdat_content_offset.to_be_bytes();
        meta[base_offset_pos..base_offset_pos + 4].copy_from_slice(&bytes);
    }
}

/// Patch iloc offsets for multi-item HEIC (grid + tiles).
fn patch_iloc_multi_offsets(meta: &mut [u8], mdat_content_start: u32, items: &[(u16, u32)]) {
    if let Some(iloc_offset) = find_box_in_meta(meta, b"iloc") {
        // iloc: 12 (full-box header) + 2 (sizes byte) + 2 (item_count)
        let mut pos = iloc_offset + 12 + 4; // after full-box header + size bytes + count
        let mut mdat_offset = mdat_content_start;

        for &(_item_id, length) in items {
            // Per item: item_id(2) + data_ref(2) + base_offset(4) + extent_count(2) + extent_offset(4) + extent_length(4)
            // base_offset is at pos + 4
            let bo_pos = pos + 4; // skip item_id(2) + data_ref(2)
            let bytes = mdat_offset.to_be_bytes();
            meta[bo_pos..bo_pos + 4].copy_from_slice(&bytes);
            mdat_offset += length;
            pos += 18; // 2+2+4+2+4+4 = 18 bytes per item
        }
    }
}

/// Find a child box within the meta box by FourCC, return its offset within the buffer.
fn find_box_in_meta(meta: &[u8], target: &[u8; 4]) -> Option<usize> {
    // meta full-box: 12 bytes header (8 box + 4 version/flags)
    let mut pos = 12;
    while pos + 8 <= meta.len() {
        let size =
            u32::from_be_bytes([meta[pos], meta[pos + 1], meta[pos + 2], meta[pos + 3]]) as usize;
        let fourcc = &meta[pos + 4..pos + 8];
        if fourcc == target {
            return Some(pos);
        }
        if size < 8 {
            break;
        }
        pos += size;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::parse;
    use crate::types::{Brand, CodecType, NalArray};

    fn make_test_hevc_config() -> HevcConfig {
        HevcConfig {
            configuration_version: 1,
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 1, // Main profile
            general_level_idc: 93,  // Level 3.1
            chroma_format_idc: 1,   // 4:2:0
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            nal_arrays: vec![
                NalArray {
                    completeness: true,
                    nal_type: 32, // VPS
                    nal_units: vec![vec![0x40, 0x01, 0x0C, 0x01]],
                },
                NalArray {
                    completeness: true,
                    nal_type: 33, // SPS
                    nal_units: vec![vec![0x42, 0x01, 0x01, 0x01]],
                },
                NalArray {
                    completeness: true,
                    nal_type: 34, // PPS
                    nal_units: vec![vec![0x44, 0x01]],
                },
            ],
        }
    }

    #[test]
    fn roundtrip_single_image() {
        let bitstream = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        let input = HeicInput {
            width: 1920,
            height: 1080,
            bitstream: bitstream.clone(),
            hevc_config: make_test_hevc_config(),
            color: None,
        };

        let container = assemble_heic(&input);
        let parsed = parse(&container).unwrap();

        assert_eq!(parsed.ftyp.major_brand, Brand::Heic);
        assert!(parsed.ftyp.compatible_brands.contains(&Brand::Mif1));
        assert_eq!(parsed.primary_image.item_id, 1);
        assert_eq!(parsed.primary_image.codec, CodecType::Hevc);
        assert_eq!(parsed.primary_image.width, 1920);
        assert_eq!(parsed.primary_image.height, 1080);
        assert_eq!(parsed.primary_image.bitstream, bitstream);
    }

    #[test]
    fn roundtrip_with_nclx_color() {
        let input = HeicInput {
            width: 640,
            height: 480,
            bitstream: vec![0x01, 0x02, 0x03],
            hevc_config: make_test_hevc_config(),
            color: Some(ColorInfo::Nclx {
                colour_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 6,
                full_range: true,
            }),
        };

        let container = assemble_heic(&input);
        let parsed = parse(&container).unwrap();

        assert_eq!(parsed.primary_image.width, 640);
        assert_eq!(parsed.primary_image.height, 480);
        match &parsed.primary_image.color {
            Some(ColorInfo::Nclx {
                colour_primaries,
                transfer_characteristics,
                matrix_coefficients,
                full_range,
            }) => {
                assert_eq!(*colour_primaries, 1);
                assert_eq!(*transfer_characteristics, 13);
                assert_eq!(*matrix_coefficients, 6);
                assert!(*full_range);
            }
            other => panic!("expected Nclx color, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_with_icc_profile() {
        let fake_icc = vec![0x49u8; 256]; // fake ICC data
        let input = HeicInput {
            width: 256,
            height: 256,
            bitstream: vec![0xFF; 100],
            hevc_config: make_test_hevc_config(),
            color: Some(ColorInfo::Icc(fake_icc.clone())),
        };

        let container = assemble_heic(&input);
        let parsed = parse(&container).unwrap();

        assert_eq!(parsed.primary_image.icc_profile, Some(fake_icc));
    }

    #[test]
    fn roundtrip_grid_image() {
        let config = make_test_hevc_config();
        let input = HeicGridInput {
            output_width: 128,
            output_height: 128,
            rows: 2,
            cols: 2,
            tiles: vec![
                TileInput {
                    width: 64,
                    height: 64,
                    bitstream: vec![0xAA; 50],
                },
                TileInput {
                    width: 64,
                    height: 64,
                    bitstream: vec![0xBB; 60],
                },
                TileInput {
                    width: 64,
                    height: 64,
                    bitstream: vec![0xCC; 70],
                },
                TileInput {
                    width: 64,
                    height: 64,
                    bitstream: vec![0xDD; 80],
                },
            ],
            hevc_config: config,
            color: None,
        };

        let container = assemble_heic_grid(&input);
        let parsed = parse(&container).unwrap();

        assert_eq!(parsed.ftyp.major_brand, Brand::Heic);
        assert_eq!(parsed.primary_image.width, 128);
        assert_eq!(parsed.primary_image.height, 128);

        let grid = parsed.primary_image.grid.as_ref().expect("expected grid");
        assert_eq!(grid.rows, 2);
        assert_eq!(grid.cols, 2);
        assert_eq!(grid.tiles.len(), 4);
        assert_eq!(grid.tiles[0].bitstream, vec![0xAA; 50]);
        assert_eq!(grid.tiles[1].bitstream, vec![0xBB; 60]);
        assert_eq!(grid.tiles[2].bitstream, vec![0xCC; 70]);
        assert_eq!(grid.tiles[3].bitstream, vec![0xDD; 80]);
    }

    #[test]
    fn ftyp_brands_correct() {
        let data = write_ftyp();
        let ftyp = crate::ftyp::parse_ftyp(&data).unwrap();
        assert_eq!(ftyp.major_brand, Brand::Heic);
        assert!(ftyp.compatible_brands.contains(&Brand::Mif1));
        assert!(ftyp.compatible_brands.contains(&Brand::Heic));
    }

    #[test]
    fn hvcc_roundtrip() {
        let config = make_test_hevc_config();
        let hvcc_box = write_hvcc(&config);
        // Verify it's a valid box
        let hdr = crate::boxreader::read_box_header(&hvcc_box, 0).unwrap();
        assert_eq!(hdr.box_type, *b"hvcC");
        assert!(hdr.content_size.unwrap() > 20); // at least the fixed header
    }
}
