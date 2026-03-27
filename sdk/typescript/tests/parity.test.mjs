/**
 * WASM parity tests for rasmcore-image TypeScript SDK.
 *
 * Loads the jco-transpiled component and validates all interfaces
 * against ImageMagick reference outputs.
 *
 * Run with: node --test sdk/typescript/tests/parity.test.mjs
 *
 * Prerequisites:
 *   1. cargo component build -p rasmcore-image
 *   2. npm run sdk:generate
 *   3. tests/fixtures/generate.sh
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "../../..");
const FIXTURES_DIR = join(PROJECT_ROOT, "tests/fixtures/generated");

// ─── Import the transpiled component ───

const sdk = await import("../../typescript/generated/rasmcore-image.js");
const { decoder, encoder, transform, filters } = sdk;

// ─── Fixture utilities ───

function loadFixture(name) {
  return readFileSync(join(FIXTURES_DIR, "inputs", name));
}

function loadReference(name) {
  return readFileSync(join(FIXTURES_DIR, "reference", name));
}

// ─── Comparison utilities ───

function meanAbsoluteError(a, b) {
  assert.equal(a.length, b.length, "buffer length mismatch");
  if (a.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / a.length;
}

function psnr(a, b) {
  assert.equal(a.length, b.length);
  let mse = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    mse += diff * diff;
  }
  mse /= a.length;
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}

// =============================================================================
// Decoder tests
// =============================================================================

describe("decoder", () => {
  it("detects PNG format", () => {
    const data = loadFixture("gradient_64x64.png");
    assert.equal(decoder.detectFormat(data), "png");
  });

  it("detects JPEG format", () => {
    const data = loadFixture("gradient_64x64.jpeg");
    assert.equal(decoder.detectFormat(data), "jpeg");
  });

  it("decodes PNG with correct dimensions", () => {
    const data = loadFixture("gradient_64x64.png");
    const result = decoder.decode(data);
    assert.equal(result.info.width, 64);
    assert.equal(result.info.height, 64);
  });

  it("decodes all supported formats", () => {
    const cases = [
      ["gradient_64x64.png", "png"],
      ["gradient_64x64.jpeg", "jpeg"],
      ["gradient_64x64.webp", "webp"],
      ["gradient_64x64.gif", "gif"],
      ["gradient_64x64.bmp", "bmp"],
      ["gradient_64x64.tiff", "tiff"],
      ["gradient_64x64.qoi", "qoi"],
    ];

    for (const [name, expectedFormat] of cases) {
      const data = loadFixture(name);
      assert.equal(
        decoder.detectFormat(data),
        expectedFormat,
        `format detection failed for ${name}`
      );
      const img = decoder.decode(data);
      assert.equal(img.info.width, 64, `width mismatch for ${name}`);
      assert.equal(img.info.height, 64, `height mismatch for ${name}`);
    }
  });

  it("decodes as RGBA8", () => {
    const data = loadFixture("gradient_64x64.png");
    const result = decoder.decodeAs(data, "rgba8");
    assert.equal(result.info.format, "rgba8");
    assert.equal(result.pixels.length, 64 * 64 * 4);
  });

  it("lists supported formats", () => {
    const formats = decoder.supportedFormats();
    assert.ok(formats.includes("png"));
    assert.ok(formats.includes("jpeg"));
    assert.ok(formats.includes("webp"));
  });
});

// =============================================================================
// Encoder tests
// =============================================================================

describe("encoder", () => {
  it("PNG roundtrip is lossless", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const encoded = encoder.encode(decoded.pixels, decoded.info, "png", undefined);
    const reDecoded = decoder.decode(encoded);
    assert.deepEqual(decoded.pixels, reDecoded.pixels);
  });

  it("JPEG roundtrip has PSNR > 30dB", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decodeAs(data, "rgba8");
    const encoded = encoder.encode(decoded.pixels, decoded.info, "jpeg", 95);
    const reDecoded = decoder.decodeAs(encoded, "rgba8");
    const quality = psnr(decoded.pixels, reDecoded.pixels);
    assert.ok(quality > 30, `JPEG roundtrip PSNR too low: ${quality.toFixed(1)}dB`);
  });

  it("lists supported formats", () => {
    const formats = encoder.supportedFormats();
    assert.ok(formats.includes("png"));
    assert.ok(formats.includes("jpeg"));
  });
});

// =============================================================================
// Transform tests (vs ImageMagick reference)
// =============================================================================

describe("transform", () => {
  it("resize lanczos matches ImageMagick (MAE < 10)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [resizedPx, resizedInfo] = transform.resize(
      decoded.pixels, decoded.info, 32, 16, "lanczos3"
    );
    assert.equal(resizedInfo.width, 32);
    assert.equal(resizedInfo.height, 16);

    const refData = loadReference("resize_lanczos_32x16.png");
    const refDecoded = decoder.decode(refData);
    const mae = meanAbsoluteError(resizedPx, refDecoded.pixels);
    assert.ok(mae < 10, `resize MAE vs ImageMagick too high: ${mae.toFixed(2)}`);
  });

  it("crop matches ImageMagick (MAE < 1)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [croppedPx, croppedInfo] = transform.crop(
      decoded.pixels, decoded.info, 8, 8, 16, 16
    );

    const refData = loadReference("crop_16x16_8_8.png");
    const refDecoded = decoder.decode(refData);
    assert.equal(croppedInfo.width, refDecoded.info.width);
    assert.equal(croppedInfo.height, refDecoded.info.height);

    const mae = meanAbsoluteError(croppedPx, refDecoded.pixels);
    assert.ok(mae < 1, `crop MAE: ${mae.toFixed(2)}`);
  });

  it("rotate 90 matches ImageMagick (MAE < 1)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [rotatedPx, rotatedInfo] = transform.rotate(decoded.pixels, decoded.info, "r90");

    const refData = loadReference("rotate_90.png");
    const refDecoded = decoder.decode(refData);
    assert.equal(rotatedInfo.width, refDecoded.info.width);
    assert.equal(rotatedInfo.height, refDecoded.info.height);

    const mae = meanAbsoluteError(rotatedPx, refDecoded.pixels);
    assert.ok(mae < 1, `rotate 90 MAE: ${mae.toFixed(2)}`);
  });

  it("flip horizontal matches ImageMagick (MAE < 1)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [flippedPx] = transform.flip(decoded.pixels, decoded.info, "horizontal");

    const refData = loadReference("flip_horizontal.png");
    const refDecoded = decoder.decode(refData);

    const mae = meanAbsoluteError(flippedPx, refDecoded.pixels);
    assert.ok(mae < 1, `flip horizontal MAE: ${mae.toFixed(2)}`);
  });

  it("flip vertical matches ImageMagick (MAE < 1)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [flippedPx] = transform.flip(decoded.pixels, decoded.info, "vertical");

    const refData = loadReference("flip_vertical.png");
    const refDecoded = decoder.decode(refData);

    const mae = meanAbsoluteError(flippedPx, refDecoded.pixels);
    assert.ok(mae < 1, `flip vertical MAE: ${mae.toFixed(2)}`);
  });
});

// =============================================================================
// Filter tests
// =============================================================================

describe("filters", () => {
  it("grayscale matches ImageMagick (MAE < 5)", () => {
    const data = loadFixture("gradient_64x64.png");
    const decoded = decoder.decode(data);
    const [grayPx, grayInfo] = filters.grayscale(decoded.pixels, decoded.info);
    assert.equal(grayInfo.format, "gray8");
    assert.equal(grayInfo.width, 64);
    assert.equal(grayInfo.height, 64);

    const refData = loadReference("grayscale.png");
    const refDecoded = decoder.decodeAs(refData, "gray8");

    const mae = meanAbsoluteError(grayPx, refDecoded.pixels);
    assert.ok(mae < 5, `grayscale MAE: ${mae.toFixed(2)}`);
  });
});
