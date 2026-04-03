//! Integration tests for the pipeline engine.

#[cfg(test)]
mod tests {
    use crate::domain::filters::{BlurParams, CharcoalParams};
    use crate::domain::pipeline::graph::NodeGraph;
    use crate::domain::pipeline::nodes::composite::CompositeNode;
    use crate::domain::pipeline::nodes::filters::{
        BlurNode, CharcoalMapperNode, GrayscaleMapperNode,
    };
    use crate::domain::pipeline::nodes::sink;
    use crate::domain::pipeline::nodes::source::SourceNode;
    use crate::domain::pipeline::nodes::transform::{CropNode, FlipNode, ResizeNode, RotateNode};
    use crate::domain::types::*;
    use rasmcore_pipeline::Rect;

    fn make_test_png() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64 * 64)
            .flat_map(|i| {
                let x = i % 64;
                let y = i / 64;
                [(x * 4) as u8, (y * 4) as u8, 128u8]
            })
            .collect();
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        crate::domain::encoder::png::encode(
            &pixels,
            &info,
            &crate::domain::encoder::png::PngEncodeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn pipeline_read_write_roundtrip() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let output = sink::write(&mut graph, src, "png", None, None).unwrap();

        // Output should be valid PNG
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn pipeline_read_resize_write() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let resized = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            32,
            16,
            ResizeFilter::Lanczos3,
        )));

        assert_eq!(graph.node_info(resized).unwrap().width, 32);
        assert_eq!(graph.node_info(resized).unwrap().height, 16);

        let output = sink::write(&mut graph, resized, "png", None, None).unwrap();
        // Decode the output and verify dimensions
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 32);
        assert_eq!(decoded.info.height, 16);
    }

    #[test]
    fn pipeline_read_crop_write() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let cropped = graph.add_node(Box::new(CropNode::new(src, src_info, 10, 10, 20, 20)));

        let output = sink::write(&mut graph, cropped, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 20);
        assert_eq!(decoded.info.height, 20);
    }

    #[test]
    fn pipeline_multi_step_chain() {
        // read → resize → blur → write
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        let resized = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            32,
            32,
            ResizeFilter::Lanczos3,
        )));
        let resized_info = graph.node_info(resized).unwrap();

        let blurred = graph.add_node(Box::new(BlurNode::new(
            resized,
            resized_info,
            BlurParams { radius: 2.0 },
        )));

        let output = sink::write(&mut graph, blurred, "jpeg", Some(85), None).unwrap();
        // Should produce valid JPEG
        assert_eq!(&output[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn pipeline_dag_same_source_two_outputs() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        // Two different resizes from the same source
        let thumb = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info.clone(),
            16,
            16,
            ResizeFilter::Nearest,
        )));
        let large = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            48,
            48,
            ResizeFilter::Lanczos3,
        )));

        let thumb_out = sink::write(&mut graph, thumb, "png", None, None).unwrap();
        let large_out = sink::write(&mut graph, large, "png", None, None).unwrap();

        let thumb_dec = crate::domain::decoder::decode(&thumb_out).unwrap();
        let large_dec = crate::domain::decoder::decode(&large_out).unwrap();
        assert_eq!(thumb_dec.info.width, 16);
        assert_eq!(large_dec.info.width, 48);
    }

    #[test]
    fn pipeline_grayscale_chain() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let gray = graph.add_node(Box::new(GrayscaleMapperNode::new(src, src_info)));

        // After first compute, info() should return Gray8
        // (Before compute it returns source_info as conservative estimate)
        let output = sink::write(&mut graph, gray, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Gray8);
    }

    #[test]
    fn pipeline_charcoal_blur_chain() {
        // Charcoal outputs Gray8 — verify downstream blur works correctly on Gray8
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        // Charcoal mapper: RGB8 → Gray8
        let charcoal_node = graph.add_node(Box::new(CharcoalMapperNode::new(
            src,
            src_info,
            CharcoalParams {
                radius: 1.0,
                sigma: 0.5,
            },
        )));

        // Query charcoal's output info — should be Gray8 thanks to output_format
        let charcoal_out_info = graph.node_info(charcoal_node).unwrap();
        assert_eq!(charcoal_out_info.format, PixelFormat::Gray8);

        // Blur on Gray8 output — this was broken before the mapper fix
        let blurred = graph.add_node(Box::new(BlurNode::new(
            charcoal_node,
            charcoal_out_info,
            BlurParams { radius: 1.0 },
        )));

        let output = sink::write(&mut graph, blurred, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        // Final output should be Gray8 (charcoal converts, blur preserves)
        assert_eq!(decoded.info.format, PixelFormat::Gray8);
        // Verify dimensions preserved
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_rotate_flip_chain() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let rotated = graph.add_node(Box::new(RotateNode::new(src, src_info, Rotation::R90)));
        let rotated_info = graph.node_info(rotated).unwrap();
        let flipped = graph.add_node(Box::new(FlipNode::new(
            rotated,
            rotated_info,
            FlipDirection::Horizontal,
        )));

        let output = sink::write(&mut graph, flipped, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        // 64x64 rotated 90 = 64x64 (square), flipped = 64x64
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_cache_reuse() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));

        // Request full region
        let full = Rect::new(0, 0, 64, 64);
        let p1 = graph.request_region(src, full).unwrap();

        // Request again — should be cached
        let p2 = graph.request_region(src, full).unwrap();
        assert_eq!(p1, p2);

        // Request sub-region — should extract from cache
        let sub = graph
            .request_region(src, Rect::new(10, 10, 20, 20))
            .unwrap();
        let src_info = graph.node_info(src).unwrap();
        let bpp = match src_info.format {
            PixelFormat::Rgb8 => 3usize,
            PixelFormat::Rgba8 => 4,
            _ => 4,
        };
        assert_eq!(sub.len(), 20 * 20 * bpp);
    }

    fn make_test_rgba_png(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let pixels: Vec<u8> = (0..(w * h)).flat_map(|_| [r, g, b, a]).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        crate::domain::encoder::png::encode(
            &pixels,
            &info,
            &crate::domain::encoder::png::PngEncodeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn pipeline_composite_opaque_watermark() {
        // Composite a small red overlay onto a blue background
        let bg_png = make_test_rgba_png(64, 64, 0, 0, 255, 255);
        let fg_png = make_test_rgba_png(16, 16, 255, 0, 0, 255);

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let bg_info = graph.node_info(bg).unwrap();
        let fg_info = graph.node_info(fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            fg, bg, fg_info, bg_info, 10, 10, None,
        )));

        // Output should have bg dimensions
        assert_eq!(graph.node_info(comp).unwrap().width, 64);
        assert_eq!(graph.node_info(comp).unwrap().height, 64);

        let output = sink::write(&mut graph, comp, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_composite_pixel_correctness() {
        // Verify exact pixel values after compositing
        let bg_png = make_test_rgba_png(4, 4, 0, 0, 255, 255); // blue
        let fg_png = make_test_rgba_png(2, 2, 255, 0, 0, 128); // 50% red

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let bg_info = graph.node_info(bg).unwrap();
        let fg_info = graph.node_info(fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            fg, bg, fg_info, bg_info, 1, 1, None,
        )));

        // Request full region
        let pixels = graph.request_region(comp, Rect::new(0, 0, 4, 4)).unwrap();

        // Pixel (0,0): should be pure blue (no fg overlay)
        assert_eq!(&pixels[0..4], [0, 0, 255, 255]);

        // Pixel (1,1): should be blended (50% red over blue)
        let px = &pixels[(1 * 4 + 1) * 4..(1 * 4 + 1) * 4 + 4];
        assert_eq!(px[0], 128); // red channel
        assert_eq!(px[1], 0); // green
        assert_eq!(px[2], 127); // blue channel
        assert_eq!(px[3], 255); // alpha

        // Pixel (3,3): should be pure blue (outside fg)
        let px = &pixels[(3 * 4 + 3) * 4..(3 * 4 + 3) * 4 + 4];
        assert_eq!(px, [0, 0, 255, 255]);
    }

    #[test]
    fn pipeline_composite_chain() {
        // read bg → read fg → resize fg → composite → write
        let bg_png = make_test_rgba_png(64, 64, 0, 0, 255, 255);
        let fg_png = make_test_rgba_png(32, 32, 255, 0, 0, 255);

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let fg_info = graph.node_info(fg).unwrap();
        let resized_fg = graph.add_node(Box::new(ResizeNode::new(
            fg,
            fg_info,
            16,
            16,
            ResizeFilter::Lanczos3,
        )));

        let bg_info = graph.node_info(bg).unwrap();
        let resized_fg_info = graph.node_info(resized_fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            resized_fg,
            bg,
            resized_fg_info,
            bg_info,
            24,
            24,
            None,
        )));

        let output = sink::write(&mut graph, comp, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pre_execution_validation_catches_empty_graph() {
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let result = sink::write(&mut graph, 0, "png", None, None);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("empty graph"),
            "expected empty graph error, got: {err}"
        );
    }

    #[test]
    fn pre_execution_validation_passes_for_valid_pipeline() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        // Should succeed — graph is valid
        let output = sink::write(&mut graph, src, "png", None, None).unwrap();
        assert_eq!(&output[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn graph_description_survives_write() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        let resized = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            32,
            16,
            ResizeFilter::Lanczos3,
        )));

        // Write (triggers auto_cleanup internally via finalize_layer_cache)
        let _ = sink::write(&mut graph, resized, "png", None, None).unwrap();

        // Description should still be available after write
        let desc = graph.description();
        assert_eq!(desc.len(), 2, "description should have 2 nodes after write");
        assert_eq!(desc.get(0).unwrap().output_info.width, 64); // original test PNG is 64x64
        assert_eq!(desc.get(1).unwrap().output_info.width, 32);
    }

    #[test]
    fn execute_from_description_basic() {
        use crate::domain::pipeline::graph::execute_from_description;

        let png_data = make_test_png();

        // Build pipeline to get a description
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data.clone()).unwrap()));

        // Serialize the description
        let desc = graph.description().clone();
        let desc_bytes = desc.serialize();

        // Execute from the serialized description
        let output =
            execute_from_description(&desc, &png_data, src, "png", None).unwrap();
        assert_eq!(&output[..4], &[0x89, b'P', b'N', b'G']);

        // Verify the description is reusable — execute again
        let output2 =
            execute_from_description(&desc, &png_data, src, "png", None).unwrap();
        assert_eq!(output, output2, "repeated execution must produce identical output");

        // Also verify deserialized description works
        let desc_from_bytes =
            crate::domain::pipeline::graph::GraphDescription::deserialize(&desc_bytes).unwrap();
        let output3 =
            execute_from_description(&desc_from_bytes, &png_data, 0, "png", None).unwrap();
        assert_eq!(output, output3, "deserialized description must produce identical output");
    }

    #[test]
    fn description_serialization_roundtrip_via_pipeline() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        let _blurred = graph.add_node(Box::new(BlurNode::new(
            src,
            src_info,
            crate::domain::filters::BlurParams { radius: 2.0 },
        )));

        let desc = graph.description();
        let bytes = desc.serialize();
        let desc2 =
            crate::domain::pipeline::graph::GraphDescription::deserialize(&bytes).unwrap();
        assert_eq!(desc2.len(), 2);

        // Verify JSON introspection works
        let json = desc.to_json();
        assert!(json.contains("\"id\":0"), "JSON should contain node 0");
        assert!(json.contains("\"id\":1"), "JSON should contain node 1");
    }

    #[test]
    fn fan_out_shared_upstream_with_description() {
        // source → resize(16x16) AND source → resize(48x48)
        // Both share the same source node (fan-out / multiple consumers)
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        let small = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info.clone(),
            16,
            16,
            ResizeFilter::Nearest,
        )));
        let large = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            48,
            48,
            ResizeFilter::Lanczos3,
        )));

        // Both should reference the same upstream (node 0)
        let desc = graph.description();
        assert_eq!(desc.len(), 3);
        assert_eq!(desc.get(1).unwrap().upstreams, vec![0]); // small refs source
        assert_eq!(desc.get(2).unwrap().upstreams, vec![0]); // large refs source

        // Both outputs should be correct
        let small_out = sink::write(&mut graph, small, "png", None, None).unwrap();
        let small_dec = crate::domain::decoder::decode(&small_out).unwrap();
        assert_eq!(small_dec.info.width, 16);
        assert_eq!(small_dec.info.height, 16);

        // Description still intact after first write
        assert_eq!(graph.description().len(), 3);
    }

    #[test]
    fn graph_reuse_execute_twice() {
        use crate::domain::pipeline::graph::execute_from_description;

        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data.clone()).unwrap()));

        let desc = graph.description().clone();

        // Execute from description twice
        let out1 = execute_from_description(&desc, &png_data, src, "png", None).unwrap();
        let out2 = execute_from_description(&desc, &png_data, src, "png", None).unwrap();

        assert_eq!(out1, out2, "two executions from same description must be identical");

        // Verify the description is unchanged
        let desc2 = graph.description();
        assert_eq!(desc.len(), desc2.len());
    }

    // ─── f32 Pipeline Tests ────────────────────────────────────────────────

    use crate::domain::pipeline::nodes::precision::PromoteNode;

    #[test]
    fn f32_pipeline_promote_invert_produces_different_output() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        // Source → Promote to Rgba32f → Invert
        let src = graph.add_node(Box::new(SourceNode::new(png_data.clone()).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let promoted = graph.add_node(Box::new(PromoteNode::new(src, src_info)));
        let promoted_info = graph.node_info(promoted).unwrap();
        assert_eq!(promoted_info.format, PixelFormat::Rgba32f);

        let inverted = graph.add_node(Box::new(
            crate::domain::pipeline::nodes::filters::InvertNode::new(promoted, promoted_info),
        ));

        // Write to PNG (quantizes back to u8)
        let inverted_png = sink::write(&mut graph, inverted, "png", None, None).unwrap();

        // Also write the original (no invert)
        let mut graph2 = NodeGraph::new(4 * 1024 * 1024);
        let src2 = graph2.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let original_png = sink::write(&mut graph2, src2, "png", None, None).unwrap();

        // Inverted should be different from original
        assert_ne!(inverted_png, original_png, "invert should change the image");
    }

    #[test]
    fn f32_pipeline_double_invert_is_identity() {
        let png_data = make_test_png();

        // Original: source → write
        let mut graph1 = NodeGraph::new(4 * 1024 * 1024);
        let src1 = graph1.add_node(Box::new(SourceNode::new(png_data.clone()).unwrap()));
        let original_png = sink::write(&mut graph1, src1, "png", None, None).unwrap();
        let original = crate::domain::decoder::decode(&original_png).unwrap();

        // Double invert: source → promote → invert → invert → write
        // Tests f32 pipeline with point ops (hits LUT fusion + auto-wrap).
        let mut graph2 = NodeGraph::new(4 * 1024 * 1024);
        let src2 = graph2.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src2_info = graph2.node_info(src2).unwrap();
        let promoted = graph2.add_node(Box::new(PromoteNode::new(src2, src2_info)));
        let p_info = graph2.node_info(promoted).unwrap();

        let inv1 = graph2.add_node(Box::new(
            crate::domain::pipeline::nodes::filters::InvertNode::new(promoted, p_info.clone()),
        ));
        let inv2 = graph2.add_node(Box::new(
            crate::domain::pipeline::nodes::filters::InvertNode::new(inv1, p_info),
        ));

        let roundtrip_png = sink::write(&mut graph2, inv2, "png", None, None).unwrap();
        let roundtrip = crate::domain::decoder::decode(&roundtrip_png).unwrap();

        assert_eq!(original.info.width, roundtrip.info.width);
        assert_eq!(original.info.height, roundtrip.info.height);

        // Compare RGB channels only — f32 pipeline promotes to Rgba32f so
        // output may be Rgba8 while original is Rgb8.
        let orig_rgb: Vec<u8> = match original.info.format {
            PixelFormat::Rgb8 => original.pixels.clone(),
            PixelFormat::Rgba8 => original.pixels.chunks_exact(4).flat_map(|c| &c[..3]).copied().collect(),
            _ => original.pixels.clone(),
        };
        let rt_rgb: Vec<u8> = match roundtrip.info.format {
            PixelFormat::Rgb8 => roundtrip.pixels.clone(),
            PixelFormat::Rgba8 => roundtrip.pixels.chunks_exact(4).flat_map(|c| &c[..3]).copied().collect(),
            _ => roundtrip.pixels.clone(),
        };

        let max_diff: u8 = orig_rgb
            .iter()
            .zip(rt_rgb.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert!(
            max_diff <= 1,
            "double invert should be identity (max pixel diff = {max_diff}, expected <= 1)"
        );
        eprintln!("f32_pipeline_double_invert_is_identity: PASS — max_diff = {max_diff}");
    }

    #[test]
    fn f32_pipeline_invert_changes_output() {
        let png_data = make_test_png();

        // Source → promote → invert → write (should differ from original)
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data.clone()).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let promoted = graph.add_node(Box::new(PromoteNode::new(src, src_info)));
        let p_info = graph.node_info(promoted).unwrap();
        let inverted = graph.add_node(Box::new(
            crate::domain::pipeline::nodes::filters::InvertNode::new(promoted, p_info),
        ));
        let inverted_png = sink::write(&mut graph, inverted, "png", None, None).unwrap();

        // Original: source → write
        let mut graph2 = NodeGraph::new(4 * 1024 * 1024);
        let src2 = graph2.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let original_png = sink::write(&mut graph2, src2, "png", None, None).unwrap();

        assert_ne!(inverted_png, original_png, "invert should change the image");
        eprintln!("f32_pipeline_invert_changes_output: PASS");
    }
}
