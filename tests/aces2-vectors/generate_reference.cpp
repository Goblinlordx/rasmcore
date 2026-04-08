// ACES 2.0 OT Reference Vector Generator
// Compiles against libOpenColorIO to produce ground-truth outputs.
//
// Build:
//   c++ -std=c++17 -o generate_reference generate_reference.cpp \
//       $(pkg-config --cflags --libs OpenColorIO)
//
// Usage:
//   ./generate_reference sdr-100nit-rec709.json
//   → writes sdr-100nit-rec709.ref.json with expected outputs

#include <OpenColorIO/OpenColorIO.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

namespace OCIO = OCIO_NAMESPACE;

// Minimal JSON parsing (just enough for our vector format)
struct TestVector {
    float input[3];
    std::string label;
    float output[3]; // filled by OCIO
};

struct TestFile {
    std::string description;
    std::string ocio_style;
    float tolerance;
    std::vector<TestVector> vectors;
};

// Very basic JSON number extraction
float parse_float(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '[')) pos++;
    size_t start = pos;
    while (pos < s.size() && (s[pos] == '-' || s[pos] == '.' || (s[pos] >= '0' && s[pos] <= '9') || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+')) pos++;
    return std::stof(s.substr(start, pos - start));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vectors.json>" << std::endl;
        return 1;
    }

    // Read input file
    std::ifstream ifs(argv[1]);
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    // Extract OCIO style from JSON
    std::string ocio_style;
    auto style_pos = json_str.find("\"ocio_style\"");
    if (style_pos != std::string::npos) {
        auto q1 = json_str.find('"', style_pos + 13);
        auto q2 = json_str.find('"', q1 + 1);
        ocio_style = json_str.substr(q1 + 1, q2 - q1 - 1);
    }
    std::cout << "OCIO style: " << ocio_style << std::endl;

    // Parse vectors
    std::vector<TestVector> vectors;
    size_t search_pos = 0;
    while (true) {
        auto input_pos = json_str.find("\"input\"", search_pos);
        if (input_pos == std::string::npos) break;

        TestVector tv;
        size_t p = input_pos + 8;
        tv.input[0] = parse_float(json_str, p);
        tv.input[1] = parse_float(json_str, p);
        tv.input[2] = parse_float(json_str, p);

        auto label_pos = json_str.find("\"label\"", p);
        if (label_pos != std::string::npos && label_pos < p + 200) {
            auto q1 = json_str.find('"', label_pos + 8);
            auto q2 = json_str.find('"', q1 + 1);
            tv.label = json_str.substr(q1 + 1, q2 - q1 - 1);
        }

        vectors.push_back(tv);
        search_pos = p;
    }

    std::cout << "Loaded " << vectors.size() << " test vectors" << std::endl;

    // Setup OCIO transform
    try {
        auto config = OCIO::Config::CreateRaw();
        auto bt = OCIO::BuiltinTransform::Create();
        bt->setStyle(ocio_style.c_str());

        auto proc = config->getProcessor(bt);
        auto cpu = proc->getDefaultCPUProcessor();

        // Process each vector
        for (auto& tv : vectors) {
            float rgb[3] = { tv.input[0], tv.input[1], tv.input[2] };
            cpu->applyRGB(rgb);
            tv.output[0] = rgb[0];
            tv.output[1] = rgb[1];
            tv.output[2] = rgb[2];
            printf("  %-30s [%8.5f, %8.5f, %8.5f] -> [%8.5f, %8.5f, %8.5f]\n",
                   tv.label.c_str(),
                   tv.input[0], tv.input[1], tv.input[2],
                   tv.output[0], tv.output[1], tv.output[2]);
        }

        // Write output JSON
        std::string out_path = std::string(argv[1]);
        auto dot = out_path.rfind('.');
        std::string ref_path = out_path.substr(0, dot) + ".ref.json";

        std::ofstream ofs(ref_path);
        ofs << "{\n  \"source\": \"OpenColorIO " << OCIO::GetVersion() << "\",\n";
        ofs << "  \"ocio_style\": \"" << ocio_style << "\",\n";
        ofs << "  \"vectors\": [\n";
        for (size_t i = 0; i < vectors.size(); i++) {
            auto& tv = vectors[i];
            ofs << "    { \"input\": [" << tv.input[0] << ", " << tv.input[1] << ", " << tv.input[2] << "], ";
            ofs << "\"output\": [" << tv.output[0] << ", " << tv.output[1] << ", " << tv.output[2] << "], ";
            ofs << "\"label\": \"" << tv.label << "\" }";
            if (i + 1 < vectors.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]\n}\n";
        ofs.close();

        std::cout << "Reference written to: " << ref_path << std::endl;

    } catch (const OCIO::Exception& e) {
        std::cerr << "OCIO error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
