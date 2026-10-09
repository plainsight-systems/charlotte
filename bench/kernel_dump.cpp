// Every kernel a model's program compiles, as the browser receives it:
// charlotte_dump_kernels <model.gguf> > kernels.json
//
// A DIAGNOSTIC tool, for finding which kernel a browser's shader compiler
// refuses where the page reports only that the kernels did not build: WebKit
// names no kernel and passes on no compiler error. It plans the model as a
// load does, at WebGPU's default limits and the default load policy, builds
// its graph, and writes each distinct kernel's composition — the WGSL, its
// entry point and its override constants (kernels/program.h's compose) — as
// a JSON array, in the order the graph first launches it. A browser page
// then compiles each on its own (the probe in docs/research's phone note).
//
// It runs no GPU: planning and the graph need only the file's index.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SL.io.50  Avoid endl — the output is written once, with '\n'.
//     E.27      Use error codes systematically — every failure is named on
//               stderr, with a nonzero exit.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/gpu/device_requirements.h"
#include "core/kernels/program.h"
#include "core/policy/policy.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"

using namespace bllm;

namespace {

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "charlotte_dump_kernels: %s\n", message.c_str());
    std::exit(1);
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += c;
        }
    }
    return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) die("usage: charlotte_dump_kernels <model.gguf>");
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) die(std::string("cannot open ") + argv[1]);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    gguf::MemoryByteSource source{std::as_bytes(std::span{bytes}), bytes.size()};
    gguf::TensorIndex index;
    if (gguf::read_index(source, index).error != gguf::ReadError::Ok) die("the file's index does not read");

    const residency::DeviceLimits limits{gpu::kRequirements.max_buffer_size,
                                         gpu::kRequirements.max_storage_buffer_binding_size, 256};
    model::ModelDescription description;
    residency::ResidencyPlan plan;
    if (const std::string stop = preflight::plan_load(index, limits, policy::LoadPolicy{}, description, plan);
        !stop.empty()) {
        die("the model does not plan: " + stop);
    }
    std::string_view architecture;
    (void)index.read_string("general.architecture", architecture);
    const formats::Format* cache_format = capability::find_format(plan.cache_type);
    if (cache_format == nullptr) die("the cache's format is not one this build writes");
    std::vector<kernels::Launch> launches;
    const graph::GraphResult built =
        capability::find_architecture(architecture)->graph(description, plan, *cache_format, launches);
    if (!built.ok()) die("the graph does not build: " + built.subject);

    std::set<std::tuple<std::string, std::string_view, std::vector<std::pair<std::string_view, double>>>> seen;
    std::string out = "[\n";
    for (const kernels::Launch& launch : launches) {
        kernels::Composed composed = kernels::compose(launch);
        if (!seen.emplace(composed.source, composed.entry_point, composed.constants).second) continue;
        if (seen.size() > 1) out += ",\n";
        out += "{\"entryPoint\":" + json_string(composed.entry_point) + ",\"constants\":{";
        for (std::size_t i = 0; i < composed.constants.size(); ++i) {
            char value[32];
            std::snprintf(value, sizeof(value), "%.17g", composed.constants[i].second);
            out += (i == 0 ? "" : ",") + json_string(composed.constants[i].first) + ":" + value;
        }
        out += "},\"source\":" + json_string(composed.source) + "}";
    }
    out += "\n]\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fprintf(stderr, "%zu launches, %zu distinct kernels\n", launches.size(), seen.size());
    return 0;
}
