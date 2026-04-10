//! Draw text node — renders text using Font resource.

// Text rendering
// ═══════════════════════════════════════════════════════════════════════════

/// Draw text onto the image using an externally-provided Font resource.
///
/// The font must be set on the pipeline via set_font() before using draw_text.
/// Text is rendered using CPU glyph rasterization from the cached Font atlas.
/// GPU path: not yet implemented (future: atlas texture + instanced quads).
pub struct DrawTextNode {
    upstream: u32,
    info: crate::node::NodeInfo,
    font: std::rc::Rc<crate::font::Font>,
    text: String,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 4],
}

impl DrawTextNode {
    pub fn new(
        upstream: u32,
        info: crate::node::NodeInfo,
        font: std::rc::Rc<crate::font::Font>,
        text: String,
        x: f32,
        y: f32,
        size: f32,
        color: [f32; 4],
    ) -> Self {
        Self {
            upstream,
            info,
            font,
            text,
            x,
            y,
            size,
            color,
        }
    }
}

impl crate::node::Node for DrawTextNode {
    fn info(&self) -> crate::node::NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: crate::rect::Rect,
        upstream: &mut dyn crate::node::Upstream,
    ) -> Result<Vec<f32>, crate::node::PipelineError> {
        let mut pixels = upstream.request(self.upstream, request)?;
        self.font.render_text(
            &mut pixels,
            request.width,
            request.height,
            &self.text,
            self.x,
            self.y,
            self.size,
            self.color,
        );
        Ok(pixels)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }
}

// Register draw_text as an operation (handled specially by pipeline, not via FilterFactory)
inventory::submit! {
    &crate::registry::OperationRegistration {
        name: "draw_text",
        display_name: "Draw Text",
        category: "draw",
        kind: crate::registry::OperationKind::Filter,
        capabilities: crate::registry::OperationCapabilities {
            gpu: false, analytic: false, affine: false, clut: false,
        },
        doc_path: "",
        cost: "O(n)",
        params: &[
            crate::registry::ParamDescriptor {
                name: "text", value_type: crate::registry::ParamType::String,
                min: None, max: None, step: None, default: None,
                hint: Some("text to render"), description: "Text string to render",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "x", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(10000.0), step: Some(1.0), default: Some(10.0),
                hint: Some("rc.pixels"), description: "X position",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "y", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(10000.0), step: Some(1.0), default: Some(10.0),
                hint: Some("rc.pixels"), description: "Y position",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "size", value_type: crate::registry::ParamType::F32,
                min: Some(4.0), max: Some(500.0), step: Some(1.0), default: Some(24.0),
                hint: Some("rc.pixels"), description: "Font size in pixels",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_r", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color red",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_g", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color green",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_b", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color blue",
                constraints: &[],
            },
            crate::registry::ParamDescriptor {
                name: "color_a", value_type: crate::registry::ParamType::F32,
                min: Some(0.0), max: Some(1.0), step: Some(0.01), default: Some(1.0),
                hint: None, description: "Text color alpha",
                constraints: &[],
            },
        ],
    }
}
