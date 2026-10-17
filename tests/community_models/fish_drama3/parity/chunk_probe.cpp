// Local-only probe: print text chunks as JSON lines. Not part of the PR.
#include "engine/framework/text/chunking.h"
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc, char ** argv) {
    const std::string mode_name = argv[1];
    const int64_t budget = std::stoll(argv[2]);
    const auto mode = mode_name == "tag_aware" ? engine::text::TextChunkMode::TagAware : engine::text::TextChunkMode::Default;
    std::ifstream in(argv[3]);
    std::stringstream buffer;
    buffer << in.rdbuf();
    for (const auto & chunk : engine::text::split_text_chunks(buffer.str(), budget, mode)) {
        std::string escaped;
        for (char c : chunk) {
            if (c == '"' || c == '\\') escaped += '\\';
            if (c == '\n') { escaped += "\\n"; continue; }
            escaped += c;
        }
        std::cout << "\"" << escaped << "\"\n";
    }
}
