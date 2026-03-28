//! Top-level ISOBMFF image assembly — parse(), primary image, grid, thumbnails.

use crate::boxreader::BoxIterator;
use crate::error::IsobmffError;
use crate::ftyp::parse_ftyp;
use crate::meta::parse_meta;
use crate::properties::{parse_iprp, resolve_properties};
use crate::types::{
    CodecType, ColorInfo, Ftyp, HevcConfig, ImageSpatialExtents, ItemProperties, MetaBox, Property,
    ReferenceType,
};

/// A fully parsed ISOBMFF still-image file.
#[derive(Debug, Clone)]
pub struct IsobmffFile {
    /// Parsed ftyp box.
    pub ftyp: Ftyp,
    /// Primary image item.
    pub primary_image: ImageItem,
    /// Thumbnail items (if any).
    pub thumbnails: Vec<ImageItem>,
}

/// A resolved image item with properties and bitstream location.
#[derive(Debug, Clone)]
pub struct ImageItem {
    /// Item ID.
    pub item_id: u32,
    /// Codec type (Hevc, Av1, Jpeg).
    pub codec: CodecType,
    /// Image dimensions from ispe property.
    pub width: u32,
    /// Image dimensions from ispe property.
    pub height: u32,
    /// Raw codec bitstream bytes extracted from mdat.
    pub bitstream: Vec<u8>,
    /// Codec configuration record (hvcC or av1C raw bytes).
    pub codec_config: Option<CodecConfig>,
    /// Color information (ICC or nclx).
    pub color: Option<ColorInfo>,
    /// ICC profile bytes (convenience — extracted from ColorInfo::Icc).
    pub icc_profile: Option<Vec<u8>>,
    /// Grid descriptor if this is a grid image.
    pub grid: Option<GridDescriptor>,
}

/// Codec-specific configuration extracted from properties.
#[derive(Debug, Clone)]
pub enum CodecConfig {
    Hevc(HevcConfig),
    Av1(crate::types::Av1Config),
}

/// Grid image descriptor.
#[derive(Debug, Clone)]
pub struct GridDescriptor {
    /// Number of tile rows.
    pub rows: u8,
    /// Number of tile columns.
    pub cols: u8,
    /// Output width of the composed image.
    pub output_width: u32,
    /// Output height of the composed image.
    pub output_height: u32,
    /// Tile image items (in order).
    pub tiles: Vec<ImageItem>,
}

/// Parse an ISOBMFF still-image file (HEIF/HEIC/AVIF).
///
/// Returns a fully resolved `IsobmffFile` with primary image, thumbnails,
/// and extracted bitstreams.
pub fn parse(data: &[u8]) -> Result<IsobmffFile, IsobmffError> {
    let ftyp = parse_ftyp(data)?;

    // Find meta and mdat boxes at top level
    let mut meta_offset: Option<usize> = None;
    let mut properties: Option<ItemProperties> = None;

    for box_result in BoxIterator::top_level(data) {
        let header = box_result?;
        let box_offset = header.content_offset - header.header_size as usize;
        if &header.box_type == b"meta" {
            meta_offset = Some(box_offset);
            // Also parse iprp within meta
            let full = crate::boxreader::read_full_box_header(data, box_offset)?;
            let cs = full.box_header.content_offset;
            let ce = match full.box_header.content_size {
                Some(s) => cs + s as usize,
                None => data.len(),
            };
            for child in BoxIterator::new(data, cs, ce) {
                let ch = child?;
                if &ch.box_type == b"iprp" {
                    let iprp_offset = ch.content_offset - ch.header_size as usize;
                    properties = Some(parse_iprp(data, iprp_offset)?);
                }
            }
        }
    }

    let meta_off = meta_offset.ok_or(IsobmffError::MissingBox { box_type: *b"meta" })?;
    let meta = parse_meta(data, meta_off)?;
    let props = properties.unwrap_or(ItemProperties {
        properties: Vec::new(),
        associations: Vec::new(),
    });

    // Resolve primary image
    let primary = resolve_image_item(data, meta.primary_item_id, &meta, &props)?;

    // Resolve thumbnails
    let mut thumbnails = Vec::new();
    for iref in &meta.references {
        if iref.ref_type == ReferenceType::Thmb {
            for &to_id in &iref.to_item_ids {
                if to_id == meta.primary_item_id {
                    // This reference's from_item is a thumbnail OF the primary
                    if let Ok(thumb) = resolve_image_item(data, iref.from_item_id, &meta, &props) {
                        thumbnails.push(thumb);
                    }
                }
            }
        }
    }

    Ok(IsobmffFile {
        ftyp,
        primary_image: primary,
        thumbnails,
    })
}

/// Resolve a single image item by ID.
fn resolve_image_item(
    data: &[u8],
    item_id: u32,
    meta: &MetaBox,
    props: &ItemProperties,
) -> Result<ImageItem, IsobmffError> {
    // Find item info
    let info =
        meta.items
            .iter()
            .find(|i| i.item_id == item_id)
            .ok_or(IsobmffError::InvalidStructure(format!(
                "item {item_id} not found in iinf"
            )))?;

    let codec = CodecType::from_fourcc(info.item_type);

    // Check if this is a grid item
    if &info.item_type == b"grid" {
        return resolve_grid_item(data, item_id, meta, props);
    }

    // Extract bitstream from mdat via iloc
    let bitstream = extract_bitstream(data, item_id, meta)?;

    // Resolve properties
    let resolved = resolve_properties(props, item_id);
    let mut width = 0u32;
    let mut height = 0u32;
    let mut codec_config = None;
    let mut color = None;
    let mut icc_profile = None;

    for (_essential, prop) in &resolved {
        match prop {
            Property::ImageSpatialExtents(ImageSpatialExtents {
                width: w,
                height: h,
            }) => {
                width = *w;
                height = *h;
            }
            Property::HevcConfig(cfg) => {
                codec_config = Some(CodecConfig::Hevc(cfg.clone()));
            }
            Property::Av1Config(cfg) => {
                codec_config = Some(CodecConfig::Av1(cfg.clone()));
            }
            Property::Color(ci) => {
                if let ColorInfo::Icc(icc) = ci {
                    icc_profile = Some(icc.clone());
                }
                color = Some(ci.clone());
            }
            _ => {}
        }
    }

    Ok(ImageItem {
        item_id,
        codec,
        width,
        height,
        bitstream,
        codec_config,
        color,
        icc_profile,
        grid: None,
    })
}

/// Resolve a grid image item — parse grid descriptor and resolve tiles.
fn resolve_grid_item(
    data: &[u8],
    item_id: u32,
    meta: &MetaBox,
    props: &ItemProperties,
) -> Result<ImageItem, IsobmffError> {
    // Extract grid descriptor from bitstream
    let grid_data = extract_bitstream(data, item_id, meta)?;

    if grid_data.len() < 8 {
        return Err(IsobmffError::InvalidStructure(
            "grid descriptor too short".to_string(),
        ));
    }

    // Grid descriptor format: version(1), flags(1), rows_minus1(1), cols_minus1(1),
    // output_width(2 or 4), output_height(2 or 4)
    let _version = grid_data[0];
    let flags = grid_data[1];
    let rows = grid_data[2] + 1;
    let cols = grid_data[3] + 1;

    let (output_width, output_height) = if (flags & 1) != 0 {
        // 32-bit dimensions
        if grid_data.len() < 12 {
            return Err(IsobmffError::InvalidStructure(
                "grid descriptor too short for 32-bit dimensions".to_string(),
            ));
        }
        let w = u32::from_be_bytes([grid_data[4], grid_data[5], grid_data[6], grid_data[7]]);
        let h = u32::from_be_bytes([grid_data[8], grid_data[9], grid_data[10], grid_data[11]]);
        (w, h)
    } else {
        // 16-bit dimensions
        let w = u16::from_be_bytes([grid_data[4], grid_data[5]]) as u32;
        let h = u16::from_be_bytes([grid_data[6], grid_data[7]]) as u32;
        (w, h)
    };

    // Resolve properties for the grid item (ispe, colr, etc.)
    let resolved = resolve_properties(props, item_id);
    let mut color = None;
    let mut icc_profile = None;

    for (_essential, prop) in &resolved {
        if let Property::Color(ci) = prop {
            if let ColorInfo::Icc(icc) = ci {
                icc_profile = Some(icc.clone());
            }
            color = Some(ci.clone());
        }
    }

    // Find tile items via dimg references
    let mut tile_ids = Vec::new();
    for iref in &meta.references {
        if iref.ref_type == ReferenceType::Dimg && iref.from_item_id == item_id {
            tile_ids.extend(&iref.to_item_ids);
        }
    }

    // Resolve each tile
    let mut tiles = Vec::new();
    let mut tile_codec = CodecType::Unknown([0; 4]);
    for &tile_id in &tile_ids {
        let tile = resolve_image_item(data, tile_id, meta, props)?;
        tile_codec = tile.codec;
        tiles.push(tile);
    }

    Ok(ImageItem {
        item_id,
        codec: tile_codec,
        width: output_width,
        height: output_height,
        bitstream: Vec::new(), // Grid items don't have their own bitstream
        codec_config: tiles.first().and_then(|t| t.codec_config.clone()),
        color,
        icc_profile,
        grid: Some(GridDescriptor {
            rows,
            cols,
            output_width,
            output_height,
            tiles,
        }),
    })
}

/// Extract raw bitstream bytes for an item from mdat using iloc.
fn extract_bitstream(data: &[u8], item_id: u32, meta: &MetaBox) -> Result<Vec<u8>, IsobmffError> {
    let loc = meta.locations.iter().find(|l| l.item_id == item_id).ok_or(
        IsobmffError::InvalidStructure(format!("no iloc entry for item {item_id}")),
    )?;

    let mut bitstream = Vec::new();
    for extent in &loc.extents {
        let offset = (loc.base_offset + extent.offset) as usize;
        let length = extent.length as usize;
        if offset + length > data.len() {
            return Err(IsobmffError::Truncated {
                expected: offset + length,
                available: data.len(),
            });
        }
        bitstream.extend_from_slice(&data[offset..offset + length]);
    }

    Ok(bitstream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Brand;

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

    fn make_ftyp(brand: &[u8; 4]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(brand);
        content.extend_from_slice(&0u32.to_be_bytes());
        content.extend_from_slice(brand);
        make_box(b"ftyp", &content)
    }

    /// Build a minimal but complete HEIC file for testing.
    fn build_minimal_heic() -> Vec<u8> {
        // mdat payload at a known offset
        let codec_payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];

        // Build meta children
        let pitm = make_full_box(b"pitm", 0, 0, &[0x00, 0x01]);

        let mut infe_c = Vec::new();
        infe_c.extend_from_slice(&[0x00, 0x01]); // item_id=1
        infe_c.extend_from_slice(&[0x00, 0x00]); // protection=0
        infe_c.extend_from_slice(b"hvc1");
        infe_c.push(0x00);
        let infe = make_full_box(b"infe", 2, 0, &infe_c);
        let mut iinf_c = Vec::new();
        iinf_c.extend_from_slice(&[0x00, 0x01]);
        iinf_c.extend(&infe);
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_c);

        // We'll compute the mdat offset after building all boxes before it.
        // For simplicity, use a placeholder and fix later.
        // ftyp + meta + mdat header = ftyp_len + meta_len + 8
        // The mdat content starts at that offset.

        // Build ispe property
        let ispe = make_full_box(
            b"ispe",
            0,
            0,
            &[0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00],
        ); // 1024x768
        let ipco = make_box(b"ipco", &ispe);
        let mut ipma_c = Vec::new();
        ipma_c.extend_from_slice(&1u32.to_be_bytes());
        ipma_c.extend_from_slice(&1u16.to_be_bytes());
        ipma_c.push(1);
        ipma_c.push(0x81);
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_c);
        let mut iprp_c = Vec::new();
        iprp_c.extend(&ipco);
        iprp_c.extend(&ipma);
        let iprp = make_box(b"iprp", &iprp_c);

        // Build meta (without iloc yet — need to know mdat offset)
        let ftyp = make_ftyp(b"heic");

        // Calculate meta size without iloc to determine offsets
        let _meta_children_without_iloc_size = pitm.len() + iinf.len() + iprp.len();
        // iloc will be added last, so mdat starts after ftyp + meta + iloc

        // Build iloc pointing into mdat
        // iloc: offset_size=4, length_size=4, base_offset_size=4
        let mut iloc_c = Vec::new();
        iloc_c.push(0x44); // offset_size=4, length_size=4
        iloc_c.push(0x40); // base_offset_size=4
        iloc_c.extend_from_slice(&[0x00, 0x01]); // 1 item
        iloc_c.extend_from_slice(&[0x00, 0x01]); // item_id=1
        iloc_c.extend_from_slice(&[0x00, 0x00]); // data_ref=0

        // base_offset will be the start of mdat content
        // We need to compute: ftyp.len() + meta_box_size + 8 (mdat header)
        // meta_box_size = 12 (full-box header) + pitm + iinf + iprp + iloc_box_size
        // iloc_box_size = 12 (full-box header) + iloc_c.len() + 8 (extent) + 4 (base_offset)
        // This is circular, so let's compute step by step.

        // iloc content so far = 8 bytes. Add base_offset(4) + extent_count(2) + extent(8) = 14 more
        // Total iloc content = 8 + 4 + 2 + 4 + 4 = 22
        // iloc box = 12 + 22 = 34

        // Actually let me just build it with a placeholder and fix the offset.
        iloc_c.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // base_offset placeholder
        iloc_c.extend_from_slice(&[0x00, 0x01]); // extent_count=1
        iloc_c.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // extent_offset=0
        iloc_c.extend_from_slice(&(codec_payload.len() as u32).to_be_bytes()); // extent_length
        let iloc = make_full_box(b"iloc", 0, 0, &iloc_c);

        let mut meta_c = Vec::new();
        meta_c.extend(&pitm);
        meta_c.extend(&iinf);
        meta_c.extend(&iloc);
        meta_c.extend(&iprp);
        let meta = make_full_box(b"meta", 0, 0, &meta_c);

        let mdat = make_box(b"mdat", &codec_payload);

        // Now compute the actual mdat content offset
        let mdat_content_offset = ftyp.len() + meta.len() + 8; // 8 = mdat header

        // Patch the base_offset in iloc
        // iloc is inside meta. Find the base_offset field.
        // meta full-box header: 12 bytes
        // pitm: pitm.len()
        // iinf: iinf.len()
        // iloc full-box header: 12 bytes
        // iloc content: [0x44, 0x40, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, BASE_OFFSET(4), ...]
        let iloc_start_in_meta = 12 + pitm.len() + iinf.len();
        let base_offset_in_iloc = iloc_start_in_meta + 12 + 8; // 12=full-box-header, 8=iloc fields before base_offset
        let base_offset_in_file = ftyp.len() + base_offset_in_iloc;

        let mut file_data = Vec::new();
        file_data.extend(&ftyp);
        file_data.extend(&meta);
        file_data.extend(&mdat);

        // Patch base_offset
        let offset_bytes = (mdat_content_offset as u32).to_be_bytes();
        file_data[base_offset_in_file] = offset_bytes[0];
        file_data[base_offset_in_file + 1] = offset_bytes[1];
        file_data[base_offset_in_file + 2] = offset_bytes[2];
        file_data[base_offset_in_file + 3] = offset_bytes[3];

        file_data
    }

    #[test]
    fn parse_minimal_heic() {
        let data = build_minimal_heic();
        let file = parse(&data).unwrap();

        assert_eq!(file.ftyp.major_brand, Brand::Heic);
        assert_eq!(file.primary_image.item_id, 1);
        assert_eq!(file.primary_image.codec, CodecType::Hevc);
        assert_eq!(file.primary_image.width, 1024);
        assert_eq!(file.primary_image.height, 768);
        assert_eq!(
            file.primary_image.bitstream,
            vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]
        );
    }

    #[test]
    fn parse_detects_brand() {
        let data = build_minimal_heic();
        let file = parse(&data).unwrap();
        assert!(file.ftyp.major_brand.is_heic());
    }

    #[test]
    fn missing_meta_returns_error() {
        // Just ftyp + mdat, no meta
        let mut data = make_ftyp(b"heic");
        data.extend(make_box(b"mdat", &[0; 10]));
        let err = parse(&data).unwrap_err();
        assert!(matches!(err, IsobmffError::MissingBox { .. }));
    }
}
