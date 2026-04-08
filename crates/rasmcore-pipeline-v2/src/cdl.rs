//! ASC CDL (Color Decision List) parser and apply.
//!
//! Supports .cdl, .cc, and .ccc XML format variants.
//! Implements the ASC CDL v1.2 specification:
//!   out = clamp((in * slope + offset) ^ power, 0, 1)  [per channel]
//!   out = luma + saturation * (out - luma)             [combined]
//!
//! Note: The clamp-to-[0,1] in the ASC spec is display-referred.
//! In our scene-referred pipeline, we skip the clamp and allow
//! unbounded values (matching DaVinci Resolve's CDL behavior).

use crate::node::PipelineError;

/// Parsed CDL values.
#[derive(Debug, Clone)]
pub struct CdlValues {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
    pub saturation: f32,
    pub id: Option<String>,
}

impl Default for CdlValues {
    fn default() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
            id: None,
        }
    }
}

/// Parse a CDL XML file (.cdl, .cc, or .ccc).
///
/// Returns one or more CDL corrections. A .cdl file may contain multiple
/// ColorCorrection elements in a ColorCorrectionCollection.
pub fn parse_cdl(xml: &str) -> Result<Vec<CdlValues>, PipelineError> {
    let mut results = Vec::new();

    // Find all <ColorCorrection> blocks
    let mut remaining = xml;
    while let Some(cc_start) = remaining.find("<ColorCorrection") {
        let after = &remaining[cc_start..];
        let cc_end = after.find("</ColorCorrection>")
            .ok_or_else(|| PipelineError::InvalidParams("unclosed <ColorCorrection>".into()))?;
        let block = &after[..cc_end + "</ColorCorrection>".len()];

        let mut cdl = CdlValues::default();

        // Extract id attribute
        if let Some(id) = extract_attr(block, "ColorCorrection", "id") {
            cdl.id = Some(id);
        }

        // Parse SOPNode: <Slope>, <Offset>, <Power>
        if let Some(sop) = extract_element(block, "SOPNode") {
            if let Some(slope) = extract_element(&sop, "Slope") {
                cdl.slope = parse_three_floats(&slope)?;
            }
            if let Some(offset) = extract_element(&sop, "Offset") {
                cdl.offset = parse_three_floats(&offset)?;
            }
            if let Some(power) = extract_element(&sop, "Power") {
                cdl.power = parse_three_floats(&power)?;
            }
        }

        // Parse SatNode: <Saturation>
        if let Some(sat_node) = extract_element(block, "SatNode") {
            if let Some(sat_str) = extract_element(&sat_node, "Saturation") {
                cdl.saturation = sat_str.trim().parse::<f32>()
                    .map_err(|_| PipelineError::InvalidParams(format!("invalid saturation: {sat_str}")))?;
            }
        }

        results.push(cdl);
        remaining = &remaining[cc_start + cc_end + "</ColorCorrection>".len()..];
    }

    // If no <ColorCorrection> found, try bare SOP (some .cc files)
    if results.is_empty() {
        let mut cdl = CdlValues::default();
        if let Some(slope) = extract_element(xml, "Slope") {
            cdl.slope = parse_three_floats(&slope)?;
        }
        if let Some(offset) = extract_element(xml, "Offset") {
            cdl.offset = parse_three_floats(&offset)?;
        }
        if let Some(power) = extract_element(xml, "Power") {
            cdl.power = parse_three_floats(&power)?;
        }
        if let Some(sat) = extract_element(xml, "Saturation") {
            cdl.saturation = sat.trim().parse::<f32>()
                .map_err(|_| PipelineError::InvalidParams("invalid saturation".into()))?;
        }
        // Only add if we found at least one SOP element
        if cdl.slope != [1.0; 3] || cdl.offset != [0.0; 3] || cdl.power != [1.0; 3] {
            results.push(cdl);
        }
    }

    if results.is_empty() {
        return Err(PipelineError::InvalidParams("no CDL data found in XML".into()));
    }

    Ok(results)
}

/// Apply CDL SOP + saturation to pixel data in-place.
///
/// Scene-referred: no [0,1] clamp on SOP output (matches Resolve behavior).
/// The power exponent uses max(0, value) to avoid NaN from negative bases.
pub fn apply_cdl(pixels: &mut [f32], cdl: &CdlValues) {
    let sat = cdl.saturation;
    for px in pixels.chunks_exact_mut(4) {
        // SOP per channel: (in * slope + offset) ^ power
        for c in 0..3 {
            let v = px[c] * cdl.slope[c] + cdl.offset[c];
            px[c] = v.max(0.0).powf(cdl.power[c]);
        }
        // Saturation via Rec.709 luma
        if (sat - 1.0).abs() > 1e-6 {
            let luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            for c in 0..3 {
                px[c] = luma + sat * (px[c] - luma);
            }
        }
    }
}

// ─── Minimal XML helpers ────────────────────────────────────────────────────

fn extract_element(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let after_open = &xml[start..];
    let content_start = after_open.find('>')? + 1;
    let end = after_open.find(&close)?;
    Some(after_open[content_start..end].to_string())
}

fn extract_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let open = format!("<{tag}");
    let start = xml.find(&open)?;
    let tag_end = xml[start..].find('>')?;
    let tag_str = &xml[start..start + tag_end];
    let attr_pattern = format!("{attr}=\"");
    let attr_start = tag_str.find(&attr_pattern)? + attr_pattern.len();
    let attr_end = tag_str[attr_start..].find('"')?;
    Some(tag_str[attr_start..attr_start + attr_end].to_string())
}

fn parse_three_floats(s: &str) -> Result<[f32; 3], PipelineError> {
    let parts: Vec<f32> = s.split_whitespace()
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if parts.len() < 3 {
        return Err(PipelineError::InvalidParams(format!(
            "expected 3 floats, got {}: '{s}'", parts.len()
        )));
    }
    Ok([parts[0], parts[1], parts[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CDL: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">
  <ColorDecision>
    <ColorCorrection id="shot_001">
      <SOPNode>
        <Slope>1.1 1.0 0.9</Slope>
        <Offset>0.01 0.0 -0.01</Offset>
        <Power>1.0 1.0 1.0</Power>
      </SOPNode>
      <SatNode>
        <Saturation>1.2</Saturation>
      </SatNode>
    </ColorCorrection>
  </ColorDecision>
</ColorDecisionList>"#;

    #[test]
    fn parse_cdl_basic() {
        let results = parse_cdl(SAMPLE_CDL).unwrap();
        assert_eq!(results.len(), 1);
        let cdl = &results[0];
        assert_eq!(cdl.id.as_deref(), Some("shot_001"));
        assert!((cdl.slope[0] - 1.1).abs() < 1e-6);
        assert!((cdl.slope[2] - 0.9).abs() < 1e-6);
        assert!((cdl.offset[0] - 0.01).abs() < 1e-6);
        assert!((cdl.saturation - 1.2).abs() < 1e-6);
    }

    #[test]
    fn apply_cdl_identity() {
        let cdl = CdlValues::default();
        let mut pixels = vec![0.5, 0.3, 0.7, 1.0];
        apply_cdl(&mut pixels, &cdl);
        assert!((pixels[0] - 0.5).abs() < 1e-6);
        assert!((pixels[1] - 0.3).abs() < 1e-6);
        assert!((pixels[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn apply_cdl_slope_doubles() {
        let cdl = CdlValues { slope: [2.0, 2.0, 2.0], ..CdlValues::default() };
        let mut pixels = vec![0.3, 0.3, 0.3, 1.0];
        apply_cdl(&mut pixels, &cdl);
        assert!((pixels[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn apply_cdl_saturation() {
        let cdl = CdlValues { saturation: 0.0, ..CdlValues::default() };
        let mut pixels = vec![1.0, 0.0, 0.0, 1.0]; // pure red
        apply_cdl(&mut pixels, &cdl);
        // With saturation=0, should be grayscale (luma)
        let spread = (pixels[0] - pixels[1]).abs().max((pixels[1] - pixels[2]).abs());
        assert!(spread < 1e-6, "saturation 0 should produce grayscale");
    }

    #[test]
    fn cdl_no_scene_referred_clamp() {
        // Scene-referred: slope can produce values > 1.0
        let cdl = CdlValues { slope: [3.0, 3.0, 3.0], ..CdlValues::default() };
        let mut pixels = vec![0.5, 0.5, 0.5, 1.0];
        apply_cdl(&mut pixels, &cdl);
        assert!((pixels[0] - 1.5).abs() < 1e-6, "should NOT clamp to 1.0: {}", pixels[0]);
    }
}
