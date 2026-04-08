#!/bin/bash
# Generate ACES 2.0 reference vectors using Docker.
#
# This builds OCIO from source inside a container and generates
# binary reference files for validating the Rust port.
#
# Usage:
#   ./tests/aces2-vectors/generate.sh
#
# Output: tests/aces2-vectors/output/*.bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

mkdir -p "${OUTPUT_DIR}"

echo "Building ACES 2.0 reference generator container..."
docker build -t aces2-reference "${SCRIPT_DIR}/"

echo "Generating reference vectors..."
docker run --rm -v "${OUTPUT_DIR}:/out" aces2-reference

echo ""
echo "Reference vectors written to: ${OUTPUT_DIR}/"
ls -lh "${OUTPUT_DIR}"/*.bin 2>/dev/null || echo "(no .bin files yet)"
echo ""
echo "These files are used by: cargo test -p rasmcore-pipeline-v2 --lib aces2"
