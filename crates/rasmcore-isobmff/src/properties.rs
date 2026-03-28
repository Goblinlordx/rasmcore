//! Item properties parser — iprp, ipco, ipma, and property box parsers.

use crate::boxreader::{BoxIterator, read_box_header, read_full_box_header};
use crate::error::IsobmffError;
use crate::types::{
    Av1Config, ColorInfo, HevcConfig, ImageSpatialExtents, ItemProperties, NalArray, PixelInfo,
    Property, PropertyAssociation,
};

/// Parse the iprp (item properties) box.
pub fn parse_iprp(data: &[u8], offset: usize) -> Result<ItemProperties, IsobmffError> {
    let header = read_box_header(data, offset)?;
    let content_start = header.content_offset;
    let content_end = match header.content_size {
        Some(s) => content_start + s as usize,
        None => data.len(),
    };

    let mut properties = Vec::new();
    let mut associations = Vec::new();

    for box_result in BoxIterator::new(data, content_start, content_end) {
        let child = box_result?;
        let child_offset = child.content_offset - child.header_size as usize;
        match &child.box_type {
            b"ipco" => {
                properties = parse_ipco(data, child_offset)?;
            }
            b"ipma" => {
                associations = parse_ipma(data, child_offset)?;
            }
            _ => {}
        }
    }

    Ok(ItemProperties {
        properties,
        associations,
    })
}

/// Parse ipco (item property container) — a sequence of property boxes.
fn parse_ipco(data: &[u8], offset: usize) -> Result<Vec<Property>, IsobmffError> {
    let header = read_box_header(data, offset)?;
    let content_start = header.content_offset;
    let content_end = match header.content_size {
        Some(s) => content_start + s as usize,
        None => data.len(),
    };

    let mut properties = Vec::new();

    for box_result in BoxIterator::new(data, content_start, content_end) {
        let prop_header = box_result?;
        let prop_start = prop_header.content_offset;
        let prop_end = match prop_header.content_size {
            Some(s) => prop_start + s as usize,
            None => content_end,
        };
        let prop_data = &data[prop_start..prop_end.min(data.len())];
        let prop_offset = prop_start - prop_header.header_size as usize;

        let property = match &prop_header.box_type {
            b"ispe" => parse_ispe(data, prop_offset)?,
            b"colr" => parse_colr(prop_data)?,
            b"pixi" => parse_pixi(data, prop_offset)?,
            b"hvcC" => parse_hvcc(prop_data)?,
            b"av1C" => parse_av1c(prop_data)?,
            _ => Property::Unknown {
                box_type: prop_header.box_type,
                data: prop_data.to_vec(),
            },
        };

        properties.push(property);
    }

    Ok(properties)
}

/// Parse ipma (item property association) box.
fn parse_ipma(data: &[u8], offset: usize) -> Result<Vec<PropertyAssociation>, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let content_end = match full.box_header.content_size {
        Some(s) => pos + s as usize,
        None => data.len(),
    };
    let avail = content_end.saturating_sub(pos);

    if avail < 4 {
        return Err(IsobmffError::Truncated {
            expected: 4,
            available: avail,
        });
    }

    let entry_count = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let mut cursor = pos + 4;
    let mut associations = Vec::new();

    for _ in 0..entry_count {
        if cursor >= content_end {
            break;
        }

        // item_id: u16 for version 0, u32 for version >= 1
        let item_id = if full.version < 1 {
            if cursor + 2 > content_end {
                break;
            }
            let id = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as u32;
            cursor += 2;
            id
        } else {
            if cursor + 4 > content_end {
                break;
            }
            let id = u32::from_be_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;
            id
        };

        // association_count
        if cursor >= content_end {
            break;
        }
        let assoc_count = data[cursor];
        cursor += 1;

        for _ in 0..assoc_count {
            if cursor >= content_end {
                break;
            }

            // flags & 1 selects 15-bit or 7-bit property index
            if full.flags & 1 != 0 {
                // 2 bytes: essential(1) + property_index(15)
                if cursor + 2 > content_end {
                    break;
                }
                let val = u16::from_be_bytes([data[cursor], data[cursor + 1]]);
                let essential = (val >> 15) != 0;
                let property_index = val & 0x7FFF;
                cursor += 2;
                associations.push(PropertyAssociation {
                    item_id,
                    property_index,
                    essential,
                });
            } else {
                // 1 byte: essential(1) + property_index(7)
                let byte = data[cursor];
                let essential = (byte >> 7) != 0;
                let property_index = (byte & 0x7F) as u16;
                cursor += 1;
                associations.push(PropertyAssociation {
                    item_id,
                    property_index,
                    essential,
                });
            }
        }
    }

    Ok(associations)
}

/// Parse ispe (image spatial extents) — full-box with width + height.
fn parse_ispe(data: &[u8], offset: usize) -> Result<Property, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let avail = data.len().saturating_sub(pos);

    if avail < 8 {
        return Err(IsobmffError::Truncated {
            expected: 8,
            available: avail,
        });
    }

    let width = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let height = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);

    Ok(Property::ImageSpatialExtents(ImageSpatialExtents {
        width,
        height,
    }))
}

/// Parse colr (colour information) property.
fn parse_colr(content: &[u8]) -> Result<Property, IsobmffError> {
    if content.len() < 4 {
        return Err(IsobmffError::Truncated {
            expected: 4,
            available: content.len(),
        });
    }

    let colour_type = &content[0..4];
    match colour_type {
        b"nclx" => {
            if content.len() < 11 {
                return Err(IsobmffError::Truncated {
                    expected: 11,
                    available: content.len(),
                });
            }
            let colour_primaries = u16::from_be_bytes([content[4], content[5]]);
            let transfer_characteristics = u16::from_be_bytes([content[6], content[7]]);
            let matrix_coefficients = u16::from_be_bytes([content[8], content[9]]);
            let full_range = (content[10] >> 7) != 0;

            Ok(Property::Color(ColorInfo::Nclx {
                colour_primaries,
                transfer_characteristics,
                matrix_coefficients,
                full_range,
            }))
        }
        b"prof" | b"rICC" => {
            let icc_data = content[4..].to_vec();
            Ok(Property::Color(ColorInfo::Icc(icc_data)))
        }
        _ => Ok(Property::Unknown {
            box_type: [
                colour_type[0],
                colour_type[1],
                colour_type[2],
                colour_type[3],
            ],
            data: content.to_vec(),
        }),
    }
}

/// Parse pixi (pixel information) — full-box with bits_per_channel.
fn parse_pixi(data: &[u8], offset: usize) -> Result<Property, IsobmffError> {
    let full = read_full_box_header(data, offset)?;
    let pos = full.box_header.content_offset;
    let avail = data.len().saturating_sub(pos);

    if avail < 1 {
        return Err(IsobmffError::Truncated {
            expected: 1,
            available: avail,
        });
    }

    let num_channels = data[pos] as usize;
    if avail < 1 + num_channels {
        return Err(IsobmffError::Truncated {
            expected: 1 + num_channels,
            available: avail,
        });
    }

    let bits_per_channel = data[pos + 1..pos + 1 + num_channels].to_vec();

    Ok(Property::Pixel(PixelInfo { bits_per_channel }))
}

/// Parse hvcC (HEVC decoder configuration record).
fn parse_hvcc(content: &[u8]) -> Result<Property, IsobmffError> {
    if content.len() < 23 {
        return Err(IsobmffError::Truncated {
            expected: 23,
            available: content.len(),
        });
    }

    let configuration_version = content[0];
    let general_profile_space = (content[1] >> 6) & 0x03;
    let general_tier_flag = ((content[1] >> 5) & 0x01) != 0;
    let general_profile_idc = content[1] & 0x1F;
    // bytes 2-5: general_profile_compatibility_flags (skip)
    // bytes 6-11: general_constraint_indicator_flags (skip)
    let general_level_idc = content[12];
    // bytes 13-14: min_spatial_segmentation_idc (skip)
    // byte 15: parallelism_type (skip)
    let chroma_format_idc = content[16] & 0x03;
    let bit_depth_luma = (content[17] & 0x07) + 8;
    let bit_depth_chroma = (content[18] & 0x07) + 8;
    // bytes 19-20: avg_frame_rate (skip)
    // byte 21: constant_frame_rate, num_temporal_layers, temporal_id_nesting, length_size_minus_one
    let num_arrays = content[22];

    let mut pos = 23;
    let mut nal_arrays = Vec::with_capacity(num_arrays as usize);

    for _ in 0..num_arrays {
        if pos + 3 > content.len() {
            break;
        }
        let completeness = (content[pos] >> 7) != 0;
        let nal_type = content[pos] & 0x3F;
        let num_nalus = u16::from_be_bytes([content[pos + 1], content[pos + 2]]);
        pos += 3;

        let mut nal_units = Vec::with_capacity(num_nalus as usize);
        for _ in 0..num_nalus {
            if pos + 2 > content.len() {
                break;
            }
            let nal_len = u16::from_be_bytes([content[pos], content[pos + 1]]) as usize;
            pos += 2;
            if pos + nal_len > content.len() {
                break;
            }
            nal_units.push(content[pos..pos + nal_len].to_vec());
            pos += nal_len;
        }

        nal_arrays.push(NalArray {
            completeness,
            nal_type,
            nal_units,
        });
    }

    Ok(Property::HevcConfig(HevcConfig {
        configuration_version,
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_level_idc,
        chroma_format_idc,
        bit_depth_luma,
        bit_depth_chroma,
        nal_arrays,
    }))
}

/// Parse av1C (AV1 codec configuration record).
fn parse_av1c(content: &[u8]) -> Result<Property, IsobmffError> {
    if content.len() < 4 {
        return Err(IsobmffError::Truncated {
            expected: 4,
            available: content.len(),
        });
    }

    // Byte 0: marker(1) + version(7)
    // Byte 1: seq_profile(3) + seq_level_idx_0(5)
    let seq_profile = (content[1] >> 5) & 0x07;
    let seq_level_idx_0 = content[1] & 0x1F;

    // Byte 2: seq_tier_0(1) + high_bitdepth(1) + twelve_bit(1) + monochrome(1)
    //         + chroma_subsampling_x(1) + chroma_subsampling_y(1) + chroma_sample_position(2)
    let seq_tier_0 = (content[2] >> 7) != 0;
    let high_bitdepth = ((content[2] >> 6) & 1) != 0;
    let twelve_bit = ((content[2] >> 5) & 1) != 0;
    let monochrome = ((content[2] >> 4) & 1) != 0;
    let chroma_subsampling_x = ((content[2] >> 3) & 1) != 0;
    let chroma_subsampling_y = ((content[2] >> 2) & 1) != 0;
    let chroma_sample_position = content[2] & 0x03;

    // Remaining bytes are configOBUs
    let config_obus = if content.len() > 4 {
        content[4..].to_vec()
    } else {
        Vec::new()
    };

    Ok(Property::Av1Config(Av1Config {
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
        config_obus,
    }))
}

/// Resolve properties for a given item_id.
pub fn resolve_properties(props: &ItemProperties, item_id: u32) -> Vec<(bool, &Property)> {
    props
        .associations
        .iter()
        .filter(|a| a.item_id == item_id && a.property_index > 0)
        .filter_map(|a| {
            let idx = (a.property_index - 1) as usize; // ipma is 1-based
            props.properties.get(idx).map(|p| (a.essential, p))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(fourcc: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size = (8 + content.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(fourcc);
        buf.extend_from_slice(content);
        buf
    }

    fn make_full_box(fourcc: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
        let size = (12 + content.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(fourcc);
        buf.push(version);
        let fb = flags.to_be_bytes();
        buf.extend_from_slice(&fb[1..4]);
        buf.extend_from_slice(content);
        buf
    }

    #[test]
    fn ispe_parses_dimensions() {
        let content = [
            0x00, 0x00, 0x10, 0x00, // width = 4096
            0x00, 0x00, 0x0C, 0x00, // height = 3072
        ];
        let data = make_full_box(b"ispe", 0, 0, &content);
        let prop = parse_ispe(&data, 0).unwrap();
        match prop {
            Property::ImageSpatialExtents(ispe) => {
                assert_eq!(ispe.width, 4096);
                assert_eq!(ispe.height, 3072);
            }
            _ => panic!("expected ImageSpatialExtents"),
        }
    }

    #[test]
    fn colr_nclx() {
        let mut content = Vec::new();
        content.extend_from_slice(b"nclx");
        content.extend_from_slice(&1u16.to_be_bytes()); // BT.709
        content.extend_from_slice(&13u16.to_be_bytes()); // sRGB
        content.extend_from_slice(&1u16.to_be_bytes()); // BT.709
        content.push(0x80); // full_range = true

        let prop = parse_colr(&content).unwrap();
        match prop {
            Property::Color(ColorInfo::Nclx {
                colour_primaries,
                transfer_characteristics,
                full_range,
                ..
            }) => {
                assert_eq!(colour_primaries, 1);
                assert_eq!(transfer_characteristics, 13);
                assert!(full_range);
            }
            _ => panic!("expected Nclx"),
        }
    }

    #[test]
    fn colr_icc_profile() {
        let mut content = Vec::new();
        content.extend_from_slice(b"prof");
        content.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // fake ICC data

        let prop = parse_colr(&content).unwrap();
        match prop {
            Property::Color(ColorInfo::Icc(icc)) => {
                assert_eq!(icc, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            _ => panic!("expected ICC"),
        }
    }

    #[test]
    fn pixi_three_channels() {
        let content = [3, 8, 8, 8]; // 3 channels, 8 bits each
        let data = make_full_box(b"pixi", 0, 0, &content);
        let prop = parse_pixi(&data, 0).unwrap();
        match prop {
            Property::Pixel(pi) => {
                assert_eq!(pi.bits_per_channel, vec![8, 8, 8]);
            }
            _ => panic!("expected PixelInfo"),
        }
    }

    #[test]
    fn hvcc_parses_config() {
        let mut content = vec![0u8; 23];
        content[0] = 1; // configurationVersion
        content[1] = 0x60; // profile_space=1, tier_flag=1, profile_idc=0
        content[12] = 120; // general_level_idc
        content[16] = 0x01; // chroma_format_idc = 1 (4:2:0)
        content[17] = 0x00; // bit_depth_luma = 8
        content[18] = 0x00; // bit_depth_chroma = 8
        content[22] = 1; // num_arrays = 1

        // One NAL array: SPS (type=33), 1 NAL unit of 4 bytes
        content.push(0x80 | 33); // completeness=1, nal_type=33 (SPS)
        content.extend_from_slice(&1u16.to_be_bytes()); // num_nalus = 1
        content.extend_from_slice(&4u16.to_be_bytes()); // nal_length = 4
        content.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]); // NAL data

        let prop = parse_hvcc(&content).unwrap();
        match prop {
            Property::HevcConfig(cfg) => {
                assert_eq!(cfg.configuration_version, 1);
                assert_eq!(cfg.general_profile_space, 1);
                assert!(cfg.general_tier_flag);
                assert_eq!(cfg.general_level_idc, 120);
                assert_eq!(cfg.chroma_format_idc, 1);
                assert_eq!(cfg.bit_depth_luma, 8);
                assert_eq!(cfg.bit_depth_chroma, 8);
                assert_eq!(cfg.nal_arrays.len(), 1);
                assert!(cfg.nal_arrays[0].completeness);
                assert_eq!(cfg.nal_arrays[0].nal_type, 33);
                assert_eq!(cfg.nal_arrays[0].nal_units[0], vec![0xAA, 0xBB, 0xCC, 0xDD]);
            }
            _ => panic!("expected HevcConfig"),
        }
    }

    #[test]
    fn av1c_parses_config() {
        let content = [
            0x81,        // marker=1, version=1
            0x44,        // seq_profile=2, seq_level_idx_0=4
            0b1101_0100, // tier=1, high_bitdepth=1, twelve_bit=0, mono=1, subx=0, suby=1, pos=0
            0x00,        // reserved
        ];
        let prop = parse_av1c(&content).unwrap();
        match prop {
            Property::Av1Config(cfg) => {
                assert_eq!(cfg.seq_profile, 2);
                assert_eq!(cfg.seq_level_idx_0, 4);
                assert!(cfg.seq_tier_0);
                assert!(cfg.high_bitdepth);
                assert!(!cfg.twelve_bit);
                assert!(cfg.monochrome);
                assert!(!cfg.chroma_subsampling_x);
                assert!(cfg.chroma_subsampling_y);
            }
            _ => panic!("expected Av1Config"),
        }
    }

    #[test]
    fn ipma_v0_associations() {
        // version=0, flags=0 (7-bit property indices)
        let mut content = Vec::new();
        content.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
        content.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
        content.push(2); // association_count = 2
        content.push(0x81); // essential=1, property_index=1
        content.push(0x02); // essential=0, property_index=2

        let data = make_full_box(b"ipma", 0, 0, &content);
        let assocs = parse_ipma(&data, 0).unwrap();
        assert_eq!(assocs.len(), 2);
        assert_eq!(assocs[0].item_id, 1);
        assert_eq!(assocs[0].property_index, 1);
        assert!(assocs[0].essential);
        assert_eq!(assocs[1].property_index, 2);
        assert!(!assocs[1].essential);
    }

    #[test]
    fn ipma_v0_flags1_15bit_indices() {
        // version=0, flags=1 (15-bit property indices)
        let mut content = Vec::new();
        content.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
        content.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
        content.push(1); // association_count = 1
        // essential=1, property_index=300 (0x012C)
        let val: u16 = 0x8000 | 300;
        content.extend_from_slice(&val.to_be_bytes());

        let data = make_full_box(b"ipma", 0, 1, &content); // flags=1
        let assocs = parse_ipma(&data, 0).unwrap();
        assert_eq!(assocs.len(), 1);
        assert!(assocs[0].essential);
        assert_eq!(assocs[0].property_index, 300);
    }

    #[test]
    fn iprp_with_ispe_and_colr() {
        // Build ipco with ispe + colr
        let ispe = make_full_box(
            b"ispe",
            0,
            0,
            &[
                0x00, 0x00, 0x08, 0x00, // width=2048
                0x00, 0x00, 0x06, 0x00, // height=1536
            ],
        );
        let mut colr_content = Vec::new();
        colr_content.extend_from_slice(b"nclx");
        colr_content.extend_from_slice(&1u16.to_be_bytes());
        colr_content.extend_from_slice(&13u16.to_be_bytes());
        colr_content.extend_from_slice(&1u16.to_be_bytes());
        colr_content.push(0x80);
        let colr = make_box(b"colr", &colr_content);

        let mut ipco_content = Vec::new();
        ipco_content.extend(&ispe);
        ipco_content.extend(&colr);
        let ipco = make_box(b"ipco", &ipco_content);

        // Build ipma: item 1 -> props 1,2
        let mut ipma_content = Vec::new();
        ipma_content.extend_from_slice(&1u32.to_be_bytes()); // 1 entry
        ipma_content.extend_from_slice(&1u16.to_be_bytes()); // item_id=1
        ipma_content.push(2); // 2 associations
        ipma_content.push(0x81); // essential, prop 1
        ipma_content.push(0x82); // essential, prop 2
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_content);

        let mut iprp_content = Vec::new();
        iprp_content.extend(&ipco);
        iprp_content.extend(&ipma);
        let iprp = make_box(b"iprp", &iprp_content);

        let result = parse_iprp(&iprp, 0).unwrap();
        assert_eq!(result.properties.len(), 2);
        assert_eq!(result.associations.len(), 2);

        // Check resolver
        let resolved = resolve_properties(&result, 1);
        assert_eq!(resolved.len(), 2);
        match &resolved[0].1 {
            Property::ImageSpatialExtents(ispe) => {
                assert_eq!(ispe.width, 2048);
                assert_eq!(ispe.height, 1536);
            }
            _ => panic!("expected ispe"),
        }
        match &resolved[1].1 {
            Property::Color(ColorInfo::Nclx { .. }) => {}
            _ => panic!("expected colr nclx"),
        }
    }

    #[test]
    fn iprp_multiple_items() {
        // ipco with 3 properties, ipma maps to 2 items
        let ispe1 = make_full_box(b"ispe", 0, 0, &[0, 0, 4, 0, 0, 0, 3, 0]); // 1024x768
        let ispe2 = make_full_box(b"ispe", 0, 0, &[0, 0, 0, 160, 0, 0, 0, 120]); // 160x120
        let pixi = make_full_box(b"pixi", 0, 0, &[3, 8, 8, 8]);

        let mut ipco_c = Vec::new();
        ipco_c.extend(&ispe1);
        ipco_c.extend(&ispe2);
        ipco_c.extend(&pixi);
        let ipco = make_box(b"ipco", &ipco_c);

        let mut ipma_c = Vec::new();
        ipma_c.extend_from_slice(&2u32.to_be_bytes()); // 2 entries
        // Item 1 -> props 1,3
        ipma_c.extend_from_slice(&1u16.to_be_bytes());
        ipma_c.push(2);
        ipma_c.push(0x81); // prop 1
        ipma_c.push(0x83); // prop 3
        // Item 2 -> prop 2
        ipma_c.extend_from_slice(&2u16.to_be_bytes());
        ipma_c.push(1);
        ipma_c.push(0x82); // prop 2
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_c);

        let mut iprp_c = Vec::new();
        iprp_c.extend(&ipco);
        iprp_c.extend(&ipma);
        let iprp = make_box(b"iprp", &iprp_c);

        let result = parse_iprp(&iprp, 0).unwrap();

        let item1_props = resolve_properties(&result, 1);
        assert_eq!(item1_props.len(), 2);

        let item2_props = resolve_properties(&result, 2);
        assert_eq!(item2_props.len(), 1);
        match &item2_props[0].1 {
            Property::ImageSpatialExtents(ispe) => {
                assert_eq!(ispe.width, 160);
                assert_eq!(ispe.height, 120);
            }
            _ => panic!("expected thumbnail ispe"),
        }
    }

    #[test]
    fn hvcc_multiple_nal_arrays() {
        let mut content = vec![0u8; 23];
        content[0] = 1;
        content[22] = 3; // 3 arrays: VPS, SPS, PPS

        // VPS (type=32)
        content.push(0x80 | 32);
        content.extend_from_slice(&1u16.to_be_bytes());
        content.extend_from_slice(&2u16.to_be_bytes());
        content.extend_from_slice(&[0x01, 0x02]);

        // SPS (type=33)
        content.push(0x80 | 33);
        content.extend_from_slice(&1u16.to_be_bytes());
        content.extend_from_slice(&3u16.to_be_bytes());
        content.extend_from_slice(&[0x0A, 0x0B, 0x0C]);

        // PPS (type=34)
        content.push(0x80 | 34);
        content.extend_from_slice(&1u16.to_be_bytes());
        content.extend_from_slice(&1u16.to_be_bytes());
        content.extend_from_slice(&[0xFF]);

        let prop = parse_hvcc(&content).unwrap();
        match prop {
            Property::HevcConfig(cfg) => {
                assert_eq!(cfg.nal_arrays.len(), 3);
                assert_eq!(cfg.nal_arrays[0].nal_type, 32); // VPS
                assert_eq!(cfg.nal_arrays[1].nal_type, 33); // SPS
                assert_eq!(cfg.nal_arrays[2].nal_type, 34); // PPS
                assert_eq!(cfg.nal_arrays[2].nal_units[0], vec![0xFF]);
            }
            _ => panic!("expected HevcConfig"),
        }
    }
}
