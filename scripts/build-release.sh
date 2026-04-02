#!/usr/bin/env bash
set -euo pipefail

# Build librcimg for a single target triple.
#
# Usage:
#   ./scripts/build-release.sh <target>
#   ./scripts/build-release.sh aarch64-apple-darwin
#   ./scripts/build-release.sh x86_64-unknown-linux-gnu
#
# Outputs: dist/rcimg-<version>-<target>.tar.gz (or .zip for Windows)
#
# For cross-compilation to Linux/Windows from macOS, install cross-rs:
#   cargo install cross
#
# Supported targets:
#   x86_64-unknown-linux-gnu
#   aarch64-unknown-linux-gnu
#   x86_64-apple-darwin
#   aarch64-apple-darwin
#   x86_64-pc-windows-gnu
#   aarch64-pc-windows-gnu

TARGET="${1:?Usage: build-release.sh <target-triple>}"
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
DIST_DIR="dist"
STAGING="$DIST_DIR/rcimg-${VERSION}-${TARGET}"

echo "=== Building librcimg for ${TARGET} (v${VERSION}) ==="

# Determine build tool: use cross for non-native targets
ARCH=$(uname -m)
OS=$(uname -s)
NATIVE_TARGET=""
case "${OS}-${ARCH}" in
    Darwin-arm64)  NATIVE_TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64) NATIVE_TARGET="x86_64-apple-darwin" ;;
    Linux-x86_64)  NATIVE_TARGET="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64) NATIVE_TARGET="aarch64-unknown-linux-gnu" ;;
esac

if [ "$TARGET" = "$NATIVE_TARGET" ]; then
    BUILD_CMD="cargo"
else
    # cross-rs handles Docker-based cross-compilation
    if ! command -v cross &>/dev/null; then
        echo "ERROR: cross-rs not installed. Install with: cargo install cross"
        echo "       (needed for cross-compilation to ${TARGET})"
        exit 1
    fi
    BUILD_CMD="cross"
fi

# Ensure target is installed
rustup target add "$TARGET" 2>/dev/null || true

# Build
$BUILD_CMD build -p rasmcore-ffi --release --target "$TARGET"

# Determine library filenames
case "$TARGET" in
    *-apple-darwin)
        DYLIB="librasmcore_ffi.dylib"
        STATIC="librasmcore_ffi.a"
        ;;
    *-linux-*)
        DYLIB="librasmcore_ffi.so"
        STATIC="librasmcore_ffi.a"
        ;;
    *-windows-*)
        DYLIB="rasmcore_ffi.dll"
        STATIC="rasmcore_ffi.lib"
        ;;
esac

# Stage files
rm -rf "$STAGING"
mkdir -p "$STAGING/lib" "$STAGING/include"

RELEASE_DIR="target/${TARGET}/release"
[ -f "$RELEASE_DIR/$DYLIB" ] && cp "$RELEASE_DIR/$DYLIB" "$STAGING/lib/"
[ -f "$RELEASE_DIR/$STATIC" ] && cp "$RELEASE_DIR/$STATIC" "$STAGING/lib/"

# Copy header
cp crates/rasmcore-ffi/include/rcimg.h "$STAGING/include/"

# Generate pkg-config
mkdir -p "$STAGING/lib/pkgconfig"
sed -e "s|@PREFIX@|/usr/local|" \
    -e "s|@VERSION@|${VERSION}|" \
    crates/rasmcore-ffi/rcimg.pc.in > "$STAGING/lib/pkgconfig/rcimg.pc"

# Package
cd "$DIST_DIR"
ARCHIVE_NAME="rcimg-${VERSION}-${TARGET}"
case "$TARGET" in
    *-windows-*)
        zip -r "${ARCHIVE_NAME}.zip" "${ARCHIVE_NAME}/"
        echo "=== Created ${DIST_DIR}/${ARCHIVE_NAME}.zip ==="
        ;;
    *)
        tar czf "${ARCHIVE_NAME}.tar.gz" "${ARCHIVE_NAME}/"
        echo "=== Created ${DIST_DIR}/${ARCHIVE_NAME}.tar.gz ==="
        ;;
esac

# Summary
echo ""
echo "Contents:"
find "${ARCHIVE_NAME}/" -type f | sort | while read -r f; do
    echo "  $f ($(du -h "$f" | cut -f1))"
done
