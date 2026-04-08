# ACES 2.0 Output Transform Reference Vector Generator
#
# Builds OCIO from source, compiles reference generators that call
# internal ACES2 functions directly, generates per-step binary
# reference files for validating the Rust port.
#
# Usage:
#   docker build -t aces2-reference tests/aces2-vectors/
#   docker run --rm -v $(pwd)/tests/aces2-vectors/output:/out aces2-reference

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libyaml-cpp-dev libexpat1-dev zlib1g-dev \
    python3 && \
    rm -rf /var/lib/apt/lists/*

# Clone OCIO at a known version
RUN git clone --depth 1 --branch v2.5.1 \
    https://github.com/AcademySoftwareFoundation/OpenColorIO.git /ocio

# Build OCIO (minimal — library only, no apps/tests/python)
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

# Copy reference generators into container
COPY generate_reference.cpp /generators/generate_reference.cpp
COPY generate_step_reference.cpp /generators/generate_step_reference.cpp

# Compile full-pipeline reference generator (links against installed OCIO)
RUN c++ -std=c++17 -o /generators/gen_full \
    /generators/generate_reference.cpp \
    -I/ocio/include -I/ocio/build/include \
    -L/ocio/build/src/OpenColorIO \
    -lOpenColorIO -Wl,-rpath,/ocio/build/src/OpenColorIO

# Compile per-step reference generator (links against OCIO internals)
RUN c++ -std=c++17 -o /generators/gen_steps \
    /generators/generate_step_reference.cpp \
    -I/ocio/src/OpenColorIO \
    -I/ocio/src/OpenColorIO/ops/fixedfunction/ACES2 \
    -I/ocio/include -I/ocio/build/include \
    -L/ocio/build/src/OpenColorIO \
    -lOpenColorIO -Wl,-rpath,/ocio/build/src/OpenColorIO

# Generate all reference outputs
WORKDIR /out
CMD ["/bin/bash", "-c", "\
    echo 'Generating ACES 2.0 reference vectors...' && \
    /generators/gen_full && \
    /generators/gen_steps && \
    echo 'Done. Files:' && ls -lh /out/*.bin /out/*.ref.json 2>/dev/null && \
    echo 'Copy from /out/ to tests/aces2-vectors/output/'"]
