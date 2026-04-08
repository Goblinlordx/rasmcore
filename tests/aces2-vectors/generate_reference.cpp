// ACES 2.0 OT Reference Vector Generator — Binary Format
// Generates thousands of input/output pairs from OCIO for comprehensive validation.
//
// Build:
//   c++ -std=c++17 -o generate_reference generate_reference.cpp \
//       $(pkg-config --cflags --libs OpenColorIO)
//
// Usage:
//   ./generate_reference
//   → generates .bin files for each configuration

#include <OpenColorIO/OpenColorIO.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

namespace OCIO = OCIO_NAMESPACE;

struct Header {
    char magic[4];        // "ACE2"
    uint32_t count;       // number of vector pairs
    float peak_luminance; // e.g., 100.0 or 1000.0
    uint32_t config_id;   // 0=sdr-rec709, 1=hdr-rec2020, etc.
};

struct VectorPair {
    float input[3];
    float output[3];
};

void generate_inputs(std::vector<VectorPair>& vectors) {
    std::mt19937 rng(42); // deterministic seed
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    // ─── Achromatic sweep (log-spaced, 200 values) ─────────────────────
    for (int i = 0; i < 200; i++) {
        float t = float(i) / 199.0f;
        float v = powf(10.0f, -4.0f + t * 7.0f); // 0.0001 to 1000
        vectors.push_back({{v, v, v}, {}});
    }

    // ─── Random uniform AP0 samples (5000 values) ──────────────────────
    for (int i = 0; i < 5000; i++) {
        float r = uniform(rng) * 2.0f; // 0 to 2
        float g = uniform(rng) * 2.0f;
        float b = uniform(rng) * 2.0f;
        vectors.push_back({{r, g, b}, {}});
    }

    // ─── HDR random (log-distributed, 2000 values) ─────────────────────
    for (int i = 0; i < 2000; i++) {
        float scale = powf(10.0f, uniform(rng) * 4.0f - 1.0f); // 0.1 to 1000
        float r = uniform(rng) * scale;
        float g = uniform(rng) * scale;
        float b = uniform(rng) * scale;
        vectors.push_back({{r, g, b}, {}});
    }

    // ─── Primary/secondary saturated at various levels (360 values) ────
    for (int h = 0; h < 360; h++) {
        float angle = float(h) * 3.14159265f / 180.0f;
        float r = 0.5f + 0.5f * cosf(angle);
        float g = 0.5f + 0.5f * cosf(angle - 2.094f);
        float b = 0.5f + 0.5f * cosf(angle + 2.094f);
        vectors.push_back({{r, g, b}, {}});
    }

    // ─── Edge cases (50 values) ────────────────────────────────────────
    vectors.push_back({{0.0f, 0.0f, 0.0f}, {}});        // black
    vectors.push_back({{0.18f, 0.18f, 0.18f}, {}});      // 18% grey
    vectors.push_back({{1.0f, 1.0f, 1.0f}, {}});          // white
    vectors.push_back({{-0.1f, 0.5f, 0.5f}, {}});         // negative R
    vectors.push_back({{0.5f, -0.1f, 0.5f}, {}});         // negative G
    vectors.push_back({{0.5f, 0.5f, -0.1f}, {}});         // negative B
    vectors.push_back({{100.0f, 100.0f, 100.0f}, {}});    // extreme bright
    vectors.push_back({{0.00001f, 0.00001f, 0.00001f}, {}}); // near-zero
    // Saturated primaries at reference luminance
    vectors.push_back({{1.0f, 0.0f, 0.0f}, {}});
    vectors.push_back({{0.0f, 1.0f, 0.0f}, {}});
    vectors.push_back({{0.0f, 0.0f, 1.0f}, {}});
    vectors.push_back({{1.0f, 1.0f, 0.0f}, {}});
    vectors.push_back({{0.0f, 1.0f, 1.0f}, {}});
    vectors.push_back({{1.0f, 0.0f, 1.0f}, {}});
    // Saturated at high luminance
    for (float scale : {5.0f, 10.0f, 50.0f}) {
        vectors.push_back({{scale, 0.0f, 0.0f}, {}});
        vectors.push_back({{0.0f, scale, 0.0f}, {}});
        vectors.push_back({{0.0f, 0.0f, scale}, {}});
        vectors.push_back({{scale, scale, 0.0f}, {}});
        vectors.push_back({{0.0f, scale, scale}, {}});
        vectors.push_back({{scale, 0.0f, scale}, {}});
    }
}

void write_bin(const char* path, const std::vector<VectorPair>& vectors,
               float peak_luminance, uint32_t config_id) {
    FILE* f = fopen(path, "wb");
    Header h;
    memcpy(h.magic, "ACE2", 4);
    h.count = (uint32_t)vectors.size();
    h.peak_luminance = peak_luminance;
    h.config_id = config_id;
    fwrite(&h, sizeof(Header), 1, f);
    fwrite(vectors.data(), sizeof(VectorPair), vectors.size(), f);
    fclose(f);
    printf("  Written %u vectors to %s (%.1f KB)\n",
           h.count, path, (sizeof(Header) + vectors.size() * sizeof(VectorPair)) / 1024.0f);
}

int main() {
    printf("ACES 2.0 OT Reference Generator — OpenColorIO %s\n", OCIO::GetVersion());

    struct Config {
        const char* style;
        const char* filename;
        float peak;
        uint32_t id;
    };
    Config configs[] = {
        {"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0",
         "sdr-100nit-rec709.bin", 100.0f, 0},
        {"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-1000nit-REC2020_2.0",
         "hdr-1000nit-rec2020.bin", 1000.0f, 1},
    };

    try {
        auto config = OCIO::Config::CreateRaw();

        for (auto& cfg : configs) {
            printf("\nGenerating: %s\n", cfg.style);

            auto bt = OCIO::BuiltinTransform::Create();
            bt->setStyle(cfg.style);
            auto proc = config->getProcessor(bt);
            auto cpu = proc->getDefaultCPUProcessor();

            std::vector<VectorPair> vectors;
            generate_inputs(vectors);

            // Process all inputs through OCIO
            for (auto& v : vectors) {
                float rgb[3] = {v.input[0], v.input[1], v.input[2]};
                cpu->applyRGB(rgb);
                v.output[0] = rgb[0];
                v.output[1] = rgb[1];
                v.output[2] = rgb[2];
            }

            write_bin(cfg.filename, vectors, cfg.peak, cfg.id);
        }

    } catch (const OCIO::Exception& e) {
        fprintf(stderr, "OCIO error: %s\n", e.what());
        return 1;
    }

    printf("\nDone.\n");
    return 0;
}
