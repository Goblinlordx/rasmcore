#!/usr/bin/env node
// Test V2 WASM component: brightness +0.5 +0.5 -0.5 -0.5 = identity
//
// Usage: node scripts/test-v2-wasm-roundtrip.mjs
//
// Requires: ./scripts/build-sdk.sh to have been run first

import { pipelineV2 } from '../sdk/dist/wasm/rasmcore-v2-image.js';

const { ImagePipelineV2 } = pipelineV2;

// Read a real test image from disk (any JPEG/PNG)
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');

// Find a test image
const testImages = [
  join(projectRoot, 'tests', 'fixtures', 'test.jpg'),
  join(projectRoot, 'tests', 'fixtures', 'test.png'),
  join(projectRoot, 'tests', 'codec-parity', 'fixtures', 'reference.jpg'),
];
let testImagePath = testImages.find(p => existsSync(p));

// If no fixture, create a minimal valid BMP (2x2, 24-bit)
let testImageBytes;
if (testImagePath) {
  testImageBytes = readFileSync(testImagePath);
  console.log(`Using test image: ${testImagePath}`);
} else {
  // Create minimal valid 2x2 BMP
  const bmp = new Uint8Array([
    0x42, 0x4d, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // row 0 (bottom): blue, green + 2 pad bytes
    0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
    // row 1 (top): red, white + 2 pad bytes
    0x00, 0x00, 0x80, 0x80, 0x80, 0x80, 0x00, 0x00,
  ]);
  testImageBytes = bmp;
  console.log('Using generated 2x2 BMP test image');
}

function serializeF32Param(name, value) {
  const buf = new Uint8Array(1 + name.length + 1 + 4);
  let i = 0;
  buf[i++] = name.length;
  for (let j = 0; j < name.length; j++) buf[i++] = name.charCodeAt(j);
  buf[i++] = 0; // f32 type
  const view = new DataView(buf.buffer);
  view.setFloat32(i, value, true); // little-endian
  return buf;
}

try {
  console.log('Creating V2 pipeline...');
  const pipe = new ImagePipelineV2();

  // List available operations
  const ops = pipe.listOperations();
  console.log(`Registered operations: ${ops.length}`);
  const filterOps = ops.filter(o => o.kind === 'filter');
  console.log(`Registered filters: ${filterOps.length}`);
  console.log(`Filter names: ${filterOps.map(o => o.name).join(', ')}`);

  // Read image
  console.log('\nReading image...');
  const src = pipe.read(testImageBytes);
  const info = pipe.nodeInfo(src);
  console.log(`Source: ${info.width}x${info.height} ${info.colorSpace}`);

  // Get original pixel values
  const original = pipe.render(src);
  console.log(`Original pixels: R=${original[0].toFixed(4)} G=${original[1].toFixed(4)} B=${original[2].toFixed(4)} A=${original[3].toFixed(4)}`);

  // Apply brightness +0.5, +0.5, -0.5, -0.5
  console.log('\nApplying brightness: +0.5, +0.5, -0.5, -0.5...');
  let current = src;
  for (const amount of [0.5, 0.5, -0.5, -0.5]) {
    const params = serializeF32Param('amount', amount);
    current = pipe.applyFilter(current, 'brightness', params);
    console.log(`  +${amount}: node ${current}`);
  }

  // Render result
  const result = pipe.render(current);
  console.log(`\nResult pixels:   R=${result[0].toFixed(4)} G=${result[1].toFixed(4)} B=${result[2].toFixed(4)} A=${result[3].toFixed(4)}`);

  // Compare
  const maxError = Math.max(
    Math.abs(result[0] - original[0]),
    Math.abs(result[1] - original[1]),
    Math.abs(result[2] - original[2]),
  );

  console.log(`\nMax error: ${maxError.toExponential(4)}`);

  if (maxError < 1e-5) {
    console.log('✅ TEST 1 PASS: Round-trip preserves f32 precision (fusion working)');
  } else {
    console.log('❌ TEST 1 FAIL: Round-trip lost precision — fusion may not be working');
    process.exit(1);
  }

  // TEST 2: Non-identity — brightness +0.5 +0.5 should differ from original
  console.log('\n=== TEST 2: Non-identity (+0.5, +0.5) should differ ===');
  let nonId = src;
  for (const amount of [0.5, 0.5]) {
    const params = serializeF32Param('amount', amount);
    nonId = pipe.applyFilter(nonId, 'brightness', params);
  }
  const nonIdResult = pipe.render(nonId);
  console.log(`Non-identity pixels: R=${nonIdResult[0].toFixed(4)} G=${nonIdResult[1].toFixed(4)} B=${nonIdResult[2].toFixed(4)}`);
  const diff = Math.abs(nonIdResult[0] - original[0]);
  console.log(`Diff from original: ${diff.toFixed(4)}`);
  if (diff > 0.1) {
    console.log('✅ TEST 2 PASS: Non-identity result differs from original');
  } else {
    console.log('❌ TEST 2 FAIL: Non-identity result should differ but is too close');
    process.exit(1);
  }

  console.log('\n✅ ALL TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('Error:', e.payload ? JSON.stringify(e.payload) : e.message || e);
  process.exit(1);
}
