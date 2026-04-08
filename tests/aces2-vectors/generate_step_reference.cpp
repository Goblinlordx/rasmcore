// ACES 2.0 OT Per-Step Reference Vector Generator
// Links against OCIO built from source to access internal ACES2 functions.
//
// Build:
//   c++ -std=c++17 -o generate_step_reference generate_step_reference.cpp \
//       -I/tmp/ocio-source/src/OpenColorIO \
//       -I/tmp/ocio-source/src/OpenColorIO/ops/fixedfunction/ACES2 \
//       -I/tmp/ocio-source/include \
//       -I/tmp/ocio-source/build/include \
//       -L/tmp/ocio-source/build/src/OpenColorIO \
//       -lOpenColorIO \
//       -Wl,-rpath,/tmp/ocio-source/build/src/OpenColorIO
//
// Usage:
//   ./generate_step_reference
//   → generates per-step .bin files

// OCIO internal headers
#include "Transform.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

using namespace OCIO_NAMESPACE::ACES2;

struct Header {
    char magic[4];     // "ACE2"
    uint32_t count;
    uint32_t step_id;  // which function this tests
    uint32_t reserved;
};

// Step IDs
enum StepId : uint32_t {
    STEP_CAM16_FWD = 1,     // RGB_to_JMh: input f32x3 (RGB AP0) → output f32x3 (J,M,h)
    STEP_CAM16_INV = 2,     // JMh_to_RGB: input f32x3 (J,M,h) → output f32x3 (RGB AP0)
    STEP_TONESCALE = 3,     // aces_tonescale: input f32 (Y_in) → output f32 (Y_out)
    STEP_CHROMA_NORM = 4,   // chroma_compress_norm: input f32 (hue_deg) → output f32 (norm)
    STEP_TOE_FWD = 5,       // toe_fwd: input f32x4 (x, limit, k1, k2) → output f32
    STEP_FULL_SDR = 10,     // full pipeline SDR 100 nit Rec.709
    STEP_FULL_HDR = 11,     // full pipeline HDR 1000 nit Rec.2020
};

struct Vec6 { float a[6]; }; // input[3] + output[3]
struct Vec2 { float a[2]; }; // input + output (scalar)
struct Vec5 { float a[5]; }; // input[4] + output[1] (toe)

template<typename T>
void write_bin(const char* path, uint32_t step_id, const std::vector<T>& data) {
    FILE* f = fopen(path, "wb");
    Header h;
    memcpy(h.magic, "ACE2", 4);
    h.count = (uint32_t)data.size();
    h.step_id = step_id;
    h.reserved = 0;
    fwrite(&h, sizeof(Header), 1, f);
    fwrite(data.data(), sizeof(T), data.size(), f);
    fclose(f);
    printf("  Written %u vectors to %s (%.1f KB)\n",
           h.count, path, (sizeof(Header) + data.size() * sizeof(T)) / 1024.0f);
}

int main() {
    printf("ACES 2.0 Per-Step Reference Generator\n");
    printf("Using OCIO internal ACES2 functions directly\n\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    // Initialize params (same as OCIO does for SDR 100 nit)
    const float peakLuminance = 100.0f;
    const auto& limitPrims = ACES_AP1::primaries; // Rec.709 would need actual Rec709 prims
    // For simplicity use AP1 as limiting — the point is testing the math functions

    JMhParams pIn = init_JMhParams(ACES_AP0::primaries);
    JMhParams pReach = init_JMhParams(ACES_AP1::primaries);
    ToneScaleParams tsParams = init_ToneScaleParams(peakLuminance);
    SharedCompressionParameters sharedParams = init_SharedCompressionParams(peakLuminance, pIn, pReach);
    ChromaCompressParams chromaParams = init_ChromaCompressParams(peakLuminance, tsParams);

    // ─── Step 1: CAM16 Forward (RGB AP0 → JMh) ────────────────────────
    {
        printf("Step 1: CAM16 Forward (RGB AP0 → JMh)\n");
        std::vector<Vec6> data;

        // Achromatic sweep
        for (int i = 0; i < 200; i++) {
            float t = float(i) / 199.0f;
            float v = powf(10.0f, -4.0f + t * 6.0f);
            f3 rgb = {v, v, v};
            f3 jmh = RGB_to_JMh(rgb, pIn);
            data.push_back({{rgb[0], rgb[1], rgb[2], jmh[0], jmh[1], jmh[2]}});
        }

        // Random colors
        for (int i = 0; i < 2000; i++) {
            float r = uniform01(rng) * 2.0f;
            float g = uniform01(rng) * 2.0f;
            float b = uniform01(rng) * 2.0f;
            f3 rgb = {r, g, b};
            f3 jmh = RGB_to_JMh(rgb, pIn);
            data.push_back({{rgb[0], rgb[1], rgb[2], jmh[0], jmh[1], jmh[2]}});
        }

        // Edge cases
        for (auto& rgb : std::vector<f3>{{0,0,0}, {0.18f,0.18f,0.18f}, {1,1,1}, {1,0,0}, {0,1,0}, {0,0,1}}) {
            f3 jmh = RGB_to_JMh(rgb, pIn);
            data.push_back({{rgb[0], rgb[1], rgb[2], jmh[0], jmh[1], jmh[2]}});
        }

        write_bin("cam16_fwd_ap0.bin", STEP_CAM16_FWD, data);
    }

    // ─── Step 2: CAM16 Inverse (JMh → RGB AP0) ────────────────────────
    {
        printf("Step 2: CAM16 Inverse (JMh → RGB AP0)\n");
        std::vector<Vec6> data;

        // Generate JMh values by forward-converting known RGBs, then verify inverse
        for (int i = 0; i < 2000; i++) {
            float r = uniform01(rng) * 2.0f;
            float g = uniform01(rng) * 2.0f;
            float b = uniform01(rng) * 2.0f;
            f3 rgb_in = {r, g, b};
            f3 jmh = RGB_to_JMh(rgb_in, pIn);
            f3 rgb_out = JMh_to_RGB(jmh, pIn);
            data.push_back({{jmh[0], jmh[1], jmh[2], rgb_out[0], rgb_out[1], rgb_out[2]}});
        }

        write_bin("cam16_inv_ap0.bin", STEP_CAM16_INV, data);
    }

    // ─── Step 3: Tonescale (Y → Y_ts) ─────────────────────────────────
    {
        printf("Step 3: Tonescale (Y → Y_ts)\n");
        std::vector<Vec2> data;

        // Dense log sweep
        for (int i = 0; i < 5000; i++) {
            float t = float(i) / 4999.0f;
            float y_in = powf(10.0f, -5.0f + t * 8.0f); // 1e-5 to 1000
            float y_out = aces_tonescale<false>(y_in, tsParams);
            data.push_back({{y_in, y_out}});
        }

        write_bin("tonescale_100nit.bin", STEP_TONESCALE, data);
    }

    // ─── Step 3b: Tonescale HDR ────────────────────────────────────────
    {
        printf("Step 3b: Tonescale HDR 1000 nit\n");
        ToneScaleParams tsHdr = init_ToneScaleParams(1000.0f);
        std::vector<Vec2> data;

        for (int i = 0; i < 5000; i++) {
            float t = float(i) / 4999.0f;
            float y_in = powf(10.0f, -5.0f + t * 8.0f);
            float y_out = aces_tonescale<false>(y_in, tsHdr);
            data.push_back({{y_in, y_out}});
        }

        write_bin("tonescale_1000nit.bin", STEP_TONESCALE, data);
    }

    // ─── Step 4: Chroma Compress Norm ──────────────────────────────────
    {
        printf("Step 4: Chroma compress norm\n");
        std::vector<Vec2> data;

        for (int deg = 0; deg < 360; deg++) {
            float h_rad = float(deg) * 3.14159265f / 180.0f;
            float cos_h = cosf(h_rad);
            float sin_h = sinf(h_rad);
            float norm = chroma_compress_norm(cos_h, sin_h, chromaParams.chroma_compress_scale);
            data.push_back({{float(deg), norm}});
        }

        write_bin("chroma_norm.bin", STEP_CHROMA_NORM, data);
    }

    // ─── Step 5: Toe function ──────────────────────────────────────────
    {
        printf("Step 5: Toe function\n");
        std::vector<Vec5> data;

        float limits[] = {0.5f, 1.0f, 2.0f, 5.0f};
        float k1s[] = {0.1f, 0.5f, 1.0f, 2.0f};
        float k2s[] = {0.05f, 0.2f, 0.5f, 1.0f};

        for (float limit : limits) {
            for (float k1 : k1s) {
                for (float k2 : k2s) {
                    for (int i = 0; i < 50; i++) {
                        float x = float(i) / 49.0f * limit * 1.5f;
                        float y = toe_fwd(x, limit, k1, k2);
                        data.push_back({{x, limit, k1, k2, y}});
                    }
                }
            }
        }

        write_bin("toe_fwd.bin", STEP_TOE_FWD, data);
    }

    // ─── Step 10: Full pipeline SDR ────────────────────────────────────
    {
        printf("Step 10: Full pipeline SDR 100 nit (via public OCIO API)\n");
        // This uses the installed OCIO library's public API
        // (already generated by the other script)
        printf("  → Use generate_reference for full pipeline vectors\n");
    }

    printf("\nDone.\n");
    return 0;
}
