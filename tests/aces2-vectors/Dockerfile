# ACES 2.0 Output Transform Reference Vector Generator
#
# Builds OCIO from source, compiles reference generators using
# OCIO's public FixedFunctionTransform API to test individual
# ACES2 pipeline stages.
#
# Usage:
#   docker build -t aces2-reference tests/aces2-vectors/
#   docker run --rm -v $(pwd)/tests/aces2-vectors/output:/out aces2-reference

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config ca-certificates \
    libyaml-cpp-dev libexpat1-dev zlib1g-dev \
    libimath-dev libpystring-dev libminizip-dev \
    python3 && \
    rm -rf /var/lib/apt/lists/*

# Clone OCIO at a known version
RUN git clone --depth 1 --branch v2.5.1 \
    https://github.com/AcademySoftwareFoundation/OpenColorIO.git /ocio

# Build OCIO (minimal — library only)
WORKDIR /ocio/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOCIO_BUILD_APPS=OFF \
    -DOCIO_BUILD_TESTS=OFF \
    -DOCIO_BUILD_GPU_TESTS=OFF \
    -DOCIO_BUILD_PYTHON=OFF \
    -DOCIO_BUILD_DOCS=OFF \
    -DOCIO_INSTALL_EXT_PACKAGES=ALL && \
    cmake --build . -j$(nproc) --target OpenColorIO

# Copy reference generator
COPY generate_step_reference.cpp /generators/generate_step_reference.cpp

# OCIO include/link paths
ENV OCIO_INC="-I/ocio/build/include -I/ocio/build/include/OpenColorIO -I/ocio/include -I/ocio/include/OpenColorIO"
ENV OCIO_LNK="-L/ocio/build/src/OpenColorIO -lOpenColorIO -Wl,-rpath,/ocio/build/src/OpenColorIO"

# Compile reference generator (uses OCIO public API only)
RUN c++ -std=c++17 -o /generators/gen_steps \
    /generators/generate_step_reference.cpp \
    $OCIO_INC $OCIO_LNK

# Generate all reference outputs
WORKDIR /out
CMD ["/bin/bash", "-c", "\
    echo 'Generating ACES 2.0 reference vectors...' && \
    /generators/gen_steps && \
    echo 'Done. Files:' && ls -lh /out/*.bin 2>/dev/null"]
