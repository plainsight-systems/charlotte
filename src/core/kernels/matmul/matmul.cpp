#include "core/kernels/matmul/matmul.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "bllm/shaders_generated.h"
#include "core/capability/capability.h"
#include "core/gguf/types.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kDecodeSets = 2;        // a decode workgroup's sets of 32
constexpr std::uint32_t kDecodePairs = 4;       // or gate-and-up pairs
// The workgroups a decode product should make at least: in the profile,
// products of 512 and more read their weights at 260 to 383 GB/s, those of
// 128 at 107 to 180 (matmul.h, decode).
constexpr std::uint32_t kDecodeWorkgroups = 512;
// Prefill's tiles, by tokens, and the steps each takes: the narrowest tile
// that holds the step whole, or the widest when none does (matmul.h,
// prefill) (GDSA.6).
struct Tile {
    std::uint32_t tokens;
    TokenRange steps;
};
constexpr std::array<Tile, 3> kTiles{{{8, {2, 8}}, {16, {9, 16}}, {32, {17, UINT32_MAX}}}};
constexpr std::uint32_t kTileOutputs = 64;      // and outputs
constexpr std::uint32_t kTilePairs = 32;        // or gate-and-up pairs
// A prefill invocation's micro-tile: kMicro tokens × kMicro outputs, so a
// tile of τ tokens × kTileOutputs takes (τ / kMicro) × (kTileOutputs /
// kMicro) invocations, 4τ (matmul.h, prefill).
constexpr std::uint32_t kMicro = 4;

// Binding 1, as matmul.wgsl's Matmul lays it out: for each member its first
// word in the binding, its blocks, its first output row and its rows.
struct Member {
    std::uint32_t base, blocks, first_row, rows;
};
struct Constants {
    std::array<Member, 3> members;
    // GatedByInput's input: its row stride and first column, in floats.
    std::uint32_t gate_input[4];
    // The options' words in the weights binding: the bounds, and the bias.
    std::uint32_t clip[4];
    std::uint32_t bias[4];
};
static_assert(sizeof(Constants) == 96);

// Whether the product takes any of the options (matmul.h).
bool has_options(const MatmulLaunch& m) {
    return m.bounds[0] != nullptr || m.bias != nullptr || m.activated;
}

double activation_of(model::FeedForwardActivation a) {
    switch (a) {
        case model::FeedForwardActivation::SiLU: return 0.0;
        case model::FeedForwardActivation::GeluTanh: return 1.0;
        case model::FeedForwardActivation::Sigmoid: return 2.0;
    }
    return 0.0;
}

// The split form's one entry point, for every epilogue it takes: it writes
// range sums alone (matmul.h, split along K).
constexpr std::string_view kSplitEntry = "split_ranges";
// The split form's workgroups a tiled one's: a range each.
constexpr std::uint32_t kRanges = 32;

struct Entry {
    std::string_view decode, prefill, combine;
};

// GatedActivation has no combine: it never splits (split_tokens).
Entry entry_of(Epilogue e) {
    switch (e) {
        case Epilogue::Write: return {"decode_write", "prefill_write", "combine_write"};
        case Epilogue::QKV: return {"decode_qkv", "prefill_qkv", "combine_qkv"};
        case Epilogue::GatedActivation: return {"decode_gated", "prefill_gated", {}};
        case Epilogue::GatedByInput: return {"decode_gated_input", "prefill_gated_input", "combine_gated_input"};
    }
    return {"decode_write", "prefill_write", "combine_write"};
}

std::uint32_t ceil_div(std::uint64_t a, std::uint32_t b) { return static_cast<std::uint32_t>((a + b - 1) / b); }

// A Write or QKV decode set's rows: the most, of 4 down to 1, that still
// makes kDecodeWorkgroups workgroups of kDecodeSets sets, or 1.
std::uint32_t set_rows_for(std::uint64_t rows) {
    for (std::uint32_t s = 4; s > 1; --s) {
        if (ceil_div(rows, kDecodeSets * s) >= kDecodeWorkgroups) return s;
    }
    return 1;
}

// A weight as the product reads it: rows of every dimension but the last
// (formats/format.h), so a convolution's (16, 16, 3, 768) is 768 rows of
// 768 — a vision encoder's patch embedding — and a matrix of K × N is N rows
// of K, as it always was.
std::uint64_t columns_of(const gguf::TensorShape& shape) {
    if (shape.dimension_count < 2) return shape.dimensions[0];
    const std::uint64_t last = shape.dimensions[shape.dimension_count - 1];
    return last == 0 ? 0 : shape.element_count / last;
}

std::uint64_t outputs_of(const gguf::TensorShape& shape) {
    return shape.dimension_count < 2 ? 1 : shape.dimensions[shape.dimension_count - 1];
}

// The product over one binding: its members, the launches that read them —
// decode, and each tile width's tiled and split launches with the split's
// combine — or the head's one launch.
void add_launches(const MatmulLaunch& m, const Binding& weights, const Constants& members, std::uint64_t rows,
                  SplitGate gate, const residency::BufferRange& range_sums, std::vector<Launch>& out) {
    const residency::WeightView& head = *m.weights[0];
    const std::uint64_t columns = columns_of(head.shape());
    const bool gated = m.epilogue == Epilogue::GatedActivation;
    const auto bytes = std::as_bytes(std::span(&members, 1));
    std::vector<Binding> bindings{weights, {m.input.buffer, m.input.offset, m.input.length}};
    const std::uint32_t outputs = m.epilogue == Epilogue::QKV ? 3 : 1;
    for (std::uint32_t o = 0; o < outputs; ++o) {
        bindings.push_back({m.outputs[o].buffer, m.outputs[o].offset, m.outputs[o].length});
    }
    if (m.epilogue == Epilogue::GatedByInput) {
        bindings.push_back({m.gate_input.buffer, m.gate_input.offset, m.gate_input.length});
    }
    std::vector<Override> overrides{
        {"columns", static_cast<double>(columns)},
        {"out_width", static_cast<double>(outputs_of(head.shape()))},
        {"activation", activation_of(m.activation)}};
    if (m.bounds[0] != nullptr) overrides.push_back({"clipped", 1.0});
    if (m.bias != nullptr) overrides.push_back({"biased", 1.0});
    if (m.activated) overrides.push_back({"activated", 1.0});
    const Entry entry = entry_of(m.epilogue);
    const formats::Format* format = capability::find_format(head.format());

    // Decode: a workgroup two sets of set_rows rows, or kDecodePairs pairs.
    const std::uint32_t set_rows = gated ? 4 : set_rows_for(rows);
    std::vector<Override> decode_overrides = overrides;
    decode_overrides.push_back({"set_rows", static_cast<double>(set_rows)});
    // x_tile's length, set on decode too, which never touches it: WebKit
    // sizes no workgroup array by an override's default (matmul.wgsl).
    decode_overrides.push_back({"x_tile_vec4s", static_cast<double>(8 * kTiles.back().tokens)});
    Launch decode{shaders::matmul,
                  format,
                  std::vector<std::byte>(bytes.begin(), bytes.end()),
                  bindings,
                  ceil_div(gated ? rows / 2 : rows, gated ? kDecodePairs : kDecodeSets * set_rows) * kWorkgroupSize,
                  kWorkgroupSize,
                  m.rows,
                  std::move(decode_overrides)};
    decode.entry_point = entry.decode;
    if (m.rows == Rows::LastToken) {
        out.push_back(std::move(decode));   // the head: every regime
        return;
    }
    decode.tokens = tokens_of(Regime::Decode);
    out.push_back(std::move(decode));

    // Prefill: for each tile width, a tiled launch, each workgroup a token
    // tile an output tile, of a micro-tile an invocation; and where the gate
    // splits the product, a split launch of kRanges workgroups a tiled one's,
    // each launch in its part of the width's steps (matmul.h, split along K).
    // The gate's steps, held to those whose range sums fit the buffer.
    const std::uint64_t fits = range_sums.length / 4 / (rows * kRanges);
    // A product with options never splits: the combine binds no weights.
    const std::uint32_t split = has_options(m) ? 1u : static_cast<std::uint32_t>(
        std::min<std::uint64_t>(split_tokens(m.epilogue, rows, columns, gate), fits));
    const std::uint32_t out_tiles = gated ? ceil_div(rows / 2, kTilePairs) : ceil_div(rows, kTileOutputs);
    for (const auto [tile, steps] : kTiles) {
        std::vector<Override> with_tile = overrides;
        with_tile.push_back({"tile_tokens", static_cast<double>(tile)});
        with_tile.push_back({"x_tile_vec4s", static_cast<double>(8 * tile)});   // matmul.wgsl's x_tile
        const std::uint32_t invocations = (tile / kMicro) * (kTileOutputs / kMicro);
        const TokenRange tiled_steps{std::max(steps.least, split + 1), steps.most};
        if (tiled_steps.least <= tiled_steps.most) {
            Launch prefill{shaders::matmul,
                           format,
                           std::vector<std::byte>(bytes.begin(), bytes.end()),
                           bindings,
                           out_tiles * invocations / tile,
                           invocations,
                           Rows::EveryToken,
                           with_tile};
            prefill.entry_point = entry.prefill;
            prefill.rows_per_tile = tile;
            prefill.tokens = tiled_steps;
            out.push_back(std::move(prefill));
        }
        const TokenRange split_steps{steps.least, std::min(steps.most, split)};
        if (split_steps.least <= split_steps.most) {
            Launch ranges{shaders::matmul,
                          format,
                          std::vector<std::byte>(bytes.begin(), bytes.end()),
                          {weights, bindings[1], {range_sums.buffer, range_sums.offset, range_sums.length}},
                          out_tiles * kRanges * invocations / tile,
                          invocations,
                          Rows::EveryToken,
                          std::move(with_tile)};
            ranges.entry_point = kSplitEntry;
            ranges.rows_per_tile = tile;
            ranges.tokens = split_steps;
            out.push_back(std::move(ranges));
        }
    }
    if (split < 2) return;

    // The combine: an invocation an output of each token, in every step the
    // product splits; its bindings start at range_sums (matmul.wgsl).
    std::vector<Binding> combine_bindings{{range_sums.buffer, range_sums.offset, range_sums.length}};
    for (std::size_t b = 2; b < bindings.size(); ++b) combine_bindings.push_back(bindings[b]);
    std::vector<Override> combine_overrides = overrides;
    combine_overrides.push_back({"x_tile_vec4s", static_cast<double>(8 * kTiles.back().tokens)});
    Launch combine{shaders::matmul,
                   format,
                   std::vector<std::byte>(bytes.begin(), bytes.end()),
                   std::move(combine_bindings),
                   static_cast<std::uint32_t>(rows),
                   kWorkgroupSize,
                   Rows::EveryToken,
                   std::move(combine_overrides)};
    combine.entry_point = entry.combine;
    combine.tokens = {2, split};
    out.push_back(std::move(combine));
}

}  // namespace

std::uint32_t split_tokens(Epilogue epilogue, std::uint64_t rows, std::uint64_t columns, SplitGate gate) noexcept {
    if (epilogue == Epilogue::GatedActivation) return 1;
    // W, the tiled form's workgroups, is its output tiles × token tiles; the
    // gate takes W × denominator ≤ G × numerator and W ≤ max_workgroups.
    const std::uint64_t groups = columns / 32;
    const std::uint64_t out_tiles = (rows + kTileOutputs - 1) / kTileOutputs;
    const std::uint64_t by_ratio = groups * gate.numerator / (std::uint64_t{gate.denominator} * out_tiles);
    const std::uint64_t by_cap = gate.max_workgroups / out_tiles;
    const std::uint64_t token_tiles = std::min(by_ratio, by_cap);
    if (token_tiles == 0) return 1;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(token_tiles * kTiles.back().tokens, residency::kPrefillBlock));
}

std::uint64_t range_sum_floats(Epilogue epilogue, std::uint64_t rows, std::uint64_t columns, SplitGate gate) noexcept {
    const std::uint32_t split = split_tokens(epilogue, rows, columns, gate);
    return split < 2 ? 0 : std::uint64_t{split} * rows * kRanges;
}

std::vector<Launch> matmul_launches(const MatmulLaunch& m, SplitGate gate, residency::BufferRange range_sums) {
    const residency::WeightView& head = *m.weights[0];
    const std::uint64_t columns = columns_of(head.shape());
    const std::uint64_t block = gguf::format_layout(head.format())->block_elements;
    std::vector<Launch> out;

    if (m.span != nullptr) {
        // A product with options: one weight, one piece, the binding its
        // product span, the bounds and the bias read from it by word.
        const residency::BufferRange& span = *m.span;
        const residency::WeightPiece& p = head.pieces().front();
        const auto word = [&](const residency::WeightView& view) {
            return static_cast<std::uint32_t>((view.pieces().front().offset - span.offset) / 4);
        };
        const auto base = static_cast<std::uint32_t>((p.offset - span.offset) / 4);
        Constants members{};
        if (m.halves) {
            // Both members the whole weight, its streams laid out across
            // all its rows (device_layout.h): the gate from the second
            // half's first row, up from row 0 (matmul.wgsl, gated).
            const std::uint64_t half = p.row_count / 2;
            const auto blocks = static_cast<std::uint32_t>(p.row_count * columns / block);
            members.members[0] = {base, blocks, static_cast<std::uint32_t>(half), static_cast<std::uint32_t>(half)};
            members.members[1] = {base, blocks, 0, static_cast<std::uint32_t>(half)};
        } else {
            members.members[0] = {base, static_cast<std::uint32_t>(p.row_count * columns / block),
                                  static_cast<std::uint32_t>(p.first_row), static_cast<std::uint32_t>(p.row_count)};
        }
        if (m.bounds[0] != nullptr) {
            for (std::size_t i = 0; i < 4; ++i) members.clip[i] = word(*m.bounds[i]);
        }
        if (m.bias != nullptr) members.bias[0] = word(*m.bias);
        add_launches(m, {span.buffer, span.offset, span.length}, members, p.row_count, gate, range_sums, out);
        return out;
    }

    if (m.members == 1) {
        // One weight: a launch pair a piece, each binding its piece.
        for (const residency::WeightPiece& p : head.pieces()) {
            const Constants members{{Member{0, static_cast<std::uint32_t>(p.row_count * columns / block),
                                            static_cast<std::uint32_t>(p.first_row),
                                            static_cast<std::uint32_t>(p.row_count)},
                                     Member{}, Member{}},
                                    {m.gate_input_stride, m.gate_input_column, 0, 0},
                                    {},
                                    {}};
            add_launches(m, {p.buffer, p.offset, p.length}, members, p.row_count, gate, range_sums, out);
        }
        return out;
    }

    // A fused group: one binding spanning its members, each one piece in one
    // buffer, read from its own first word.
    std::uint64_t begin = UINT64_MAX, end = 0;
    for (std::uint32_t i = 0; i < m.members; ++i) {
        const residency::WeightPiece& p = m.weights[i]->pieces().front();
        begin = std::min(begin, p.offset);
        end = std::max(end, p.offset + p.length);
    }
    Constants members{};
    std::uint64_t rows = 0;
    for (std::uint32_t i = 0; i < m.members; ++i) {
        const residency::WeightPiece& p = m.weights[i]->pieces().front();
        // A gated product's members each read from their own row 0
        // (matmul.wgsl); every other's z is its first output row.
        const bool gated = m.epilogue == Epilogue::GatedActivation;
        members.members[i] = {static_cast<std::uint32_t>((p.offset - begin) / 4),
                              static_cast<std::uint32_t>(p.row_count * columns / block),
                              gated ? 0u : static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(p.row_count)};
        rows += p.row_count;
    }
    add_launches(m, {head.pieces().front().buffer, begin, end - begin}, members, rows, gate, range_sums, out);
    return out;
}

}  // namespace bllm::kernels
