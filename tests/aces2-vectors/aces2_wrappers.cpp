// ACES2 internal function wrappers — extern C for use by reference generator.
// This file #includes Transform.cpp to access inline functions,
// then provides C-linkage wrappers for each function we need to test.

#include "Transform.cpp"

namespace OCIO_NAMESPACE {
namespace ACES2 {

extern "C" {

float aces2_tonescale_fwd_c(float y, float peak) {
    ToneScaleParams p = init_ToneScaleParams(peak);
    return aces_tonescale<false>(y, p);
}

float aces2_toe_fwd_c(float x, float limit, float k1, float k2) {
    return toe_fwd(x, limit, k1, k2);
}

float aces2_chroma_norm_c(float cos_h, float sin_h, float scale) {
    return chroma_compress_norm(cos_h, sin_h, scale);
}

void aces2_rgb_to_jmh_c(const float* rgb_in, float* jmh_out, float /*peak*/) {
    JMhParams p = init_JMhParams(ACES_AP0::primaries);
    f3 rgb;
    rgb[0] = rgb_in[0];
    rgb[1] = rgb_in[1];
    rgb[2] = rgb_in[2];
    f3 jmh = RGB_to_JMh(rgb, p);
    jmh_out[0] = jmh[0];
    jmh_out[1] = jmh[1];
    jmh_out[2] = jmh[2];
}

} // extern "C"

} // namespace ACES2
} // namespace OCIO_NAMESPACE
