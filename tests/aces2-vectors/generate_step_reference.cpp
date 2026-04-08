// ACES 2.0 Per-Step Reference Vector Generator
// Uses OCIO's public FixedFunctionTransform API to run individual
// ACES2 pipeline stages and generate reference vectors for each.

#include <OpenColorIO/OpenColorIO.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

namespace OCIO = OCIO_NAMESPACE;

struct BinHeader {
    char magic[4];
    uint32_t count;
    uint32_t step_id;
    uint32_t reserved;
};
struct Vec6 { float a[6]; };

template<typename T>
void write_bin(const char* path, uint32_t step_id, const std::vector<T>& data) {
    FILE* f = fopen(path, "wb");
    BinHeader h;
    memcpy(h.magic, "ACE2", 4);
    h.count = (uint32_t)data.size();
    h.step_id = step_id;
    h.reserved = 0;
    fwrite(&h, sizeof(BinHeader), 1, f);
    fwrite(data.data(), sizeof(T), data.size(), f);
    fclose(f);
    printf("  Written %u vectors to %s (%.1f KB)\n",
           h.count, path, (sizeof(BinHeader) + data.size() * sizeof(T)) / 1024.0f);
}

// ACES AP0 primaries as parameter array for FixedFunctionTransform
const std::vector<double> AP0_PARAMS = {
    0.7347, 0.2653,  // red
    0.0,    1.0,     // green
    0.0001, -0.077,  // blue
    0.32168, 0.33767 // white
};

OCIO::ConstCPUProcessorRcPtr make_processor(OCIO::FixedFunctionStyle style,
                                              const std::vector<double>& params) {
    auto config = OCIO::Config::CreateRaw();
    auto ff = OCIO::FixedFunctionTransform::Create(style, params.data(), params.size());
    auto proc = config->getProcessor(ff);
    return proc->getDefaultCPUProcessor();
}

int main() {
    printf("ACES 2.0 Per-Step Reference Generator (OCIO %s)\n\n", OCIO::GetVersion());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    try {
        auto config = OCIO::Config::CreateRaw();

        // ─── Step 1: RGB_TO_JMH (CAM16 forward) ───────────────────────
        {
            printf("Step 1: RGB_TO_JMH (CAM16 forward, AP0 input)\n");
            // Params: 8 primaries (rx, ry, gx, gy, bx, by, wx, wy) for AP0
            auto cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_RGB_TO_JMH_20, AP0_PARAMS);

            std::vector<Vec6> data;
            // Achromatic sweep
            for (int i = 0; i < 500; i++) {
                float v = powf(10.0f, -4.0f + float(i)/499.0f * 6.0f);
                float rgb[3] = {v, v, v};
                float in0=v, in1=v, in2=v;
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            // Random colors
            for (int i = 0; i < 5000; i++) {
                float rgb[3] = {uniform01(rng)*2, uniform01(rng)*2, uniform01(rng)*2};
                float in0=rgb[0], in1=rgb[1], in2=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            write_bin("/out/cam16_fwd_ap0.bin", 1, data);
        }

        // ─── Step 2: TONESCALE_COMPRESS (tonescale + chroma compress) ──
        {
            printf("Step 2: TONESCALE_COMPRESS (SDR 100 nit)\n");
            // Input: JMh, Output: tonemapped+chroma-compressed JMh
            // Params: peakLuminance
            auto cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_TONESCALE_COMPRESS_20, {100.0});

            // First get JMh values by running RGB_TO_JMH
            auto cam_cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_RGB_TO_JMH_20, AP0_PARAMS);

            std::vector<Vec6> data;
            // Generate JMh inputs by converting known RGB values
            for (int i = 0; i < 500; i++) {
                float v = powf(10.0f, -4.0f + float(i)/499.0f * 6.0f);
                float rgb[3] = {v, v, v};
                cam_cpu->applyRGB(rgb); // now rgb = JMh
                float j0=rgb[0], m0=rgb[1], h0=rgb[2];
                cpu->applyRGB(rgb); // now rgb = compressed JMh
                data.push_back({{j0, m0, h0, rgb[0], rgb[1], rgb[2]}});
            }
            for (int i = 0; i < 5000; i++) {
                float rgb[3] = {uniform01(rng)*2, uniform01(rng)*2, uniform01(rng)*2};
                cam_cpu->applyRGB(rgb);
                float j0=rgb[0], m0=rgb[1], h0=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{j0, m0, h0, rgb[0], rgb[1], rgb[2]}});
            }
            write_bin("/out/tonescale_compress_100nit.bin", 2, data);
        }

        // ─── Step 3: GAMUT_COMPRESS ────────────────────────────────────
        {
            printf("Step 3: GAMUT_COMPRESS (SDR 100 nit Rec.709)\n");
            // Input: tonemapped JMh, Output: gamut-compressed JMh
            // Params: peakLuminance + 8 Rec.709 limiting primaries
            const std::vector<double> gamut_params = {
                100.0,            // peak luminance
                0.64, 0.33,       // Rec.709 red
                0.30, 0.60,       // Rec.709 green
                0.15, 0.06,       // Rec.709 blue
                0.3127, 0.3290    // D65 white
            };
            auto cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_GAMUT_COMPRESS_20, gamut_params);

            auto cam_cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_RGB_TO_JMH_20, AP0_PARAMS);
            auto tc_cpu = make_processor(OCIO::FIXED_FUNCTION_ACES_TONESCALE_COMPRESS_20, {100.0});

            std::vector<Vec6> data;
            for (int i = 0; i < 500; i++) {
                float v = powf(10.0f, -4.0f + float(i)/499.0f * 6.0f);
                float rgb[3] = {v, v, v};
                cam_cpu->applyRGB(rgb);
                tc_cpu->applyRGB(rgb);
                float j0=rgb[0], m0=rgb[1], h0=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{j0, m0, h0, rgb[0], rgb[1], rgb[2]}});
            }
            for (int i = 0; i < 5000; i++) {
                float rgb[3] = {uniform01(rng)*2, uniform01(rng)*2, uniform01(rng)*2};
                cam_cpu->applyRGB(rgb);
                tc_cpu->applyRGB(rgb);
                float j0=rgb[0], m0=rgb[1], h0=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{j0, m0, h0, rgb[0], rgb[1], rgb[2]}});
            }
            write_bin("/out/gamut_compress_100nit.bin", 3, data);
        }

        // ─── Step 10: Full pipeline SDR ────────────────────────────────
        {
            printf("Step 10: Full pipeline SDR 100 nit Rec.709\n");
            auto bt = OCIO::BuiltinTransform::Create();
            bt->setStyle("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0");
            auto proc = config->getProcessor(bt);
            auto cpu = proc->getDefaultCPUProcessor();

            std::vector<Vec6> data;
            for (int i = 0; i < 500; i++) {
                float v = powf(10.0f, -4.0f + float(i)/499.0f * 7.0f);
                float rgb[3] = {v, v, v};
                float in0=v, in1=v, in2=v;
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            for (int i = 0; i < 7000; i++) {
                float s = powf(10.0f, uniform01(rng)*4 - 1);
                float rgb[3] = {uniform01(rng)*s, uniform01(rng)*s, uniform01(rng)*s};
                float in0=rgb[0], in1=rgb[1], in2=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            write_bin("/out/full_sdr_100nit_rec709.bin", 10, data);
        }

        // ─── Step 11: Full pipeline HDR ────────────────────────────────
        {
            printf("Step 11: Full pipeline HDR 1000 nit Rec.2020\n");
            auto bt = OCIO::BuiltinTransform::Create();
            bt->setStyle("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-1000nit-REC2020_2.0");
            auto proc = config->getProcessor(bt);
            auto cpu = proc->getDefaultCPUProcessor();

            std::vector<Vec6> data;
            for (int i = 0; i < 500; i++) {
                float v = powf(10.0f, -4.0f + float(i)/499.0f * 7.0f);
                float rgb[3] = {v, v, v};
                float in0=v, in1=v, in2=v;
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            for (int i = 0; i < 7000; i++) {
                float s = powf(10.0f, uniform01(rng)*4 - 1);
                float rgb[3] = {uniform01(rng)*s, uniform01(rng)*s, uniform01(rng)*s};
                float in0=rgb[0], in1=rgb[1], in2=rgb[2];
                cpu->applyRGB(rgb);
                data.push_back({{in0, in1, in2, rgb[0], rgb[1], rgb[2]}});
            }
            write_bin("/out/full_hdr_1000nit_rec2020.bin", 11, data);
        }

    } catch (const OCIO::Exception& e) {
        fprintf(stderr, "OCIO error: %s\n", e.what());
        return 1;
    }

    printf("\nDone.\n");
    return 0;
}
